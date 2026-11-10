//! lock-free monotonic snowflake generation
//!
//! state packs the last timestamp and sequence into one atomic value so each
//! id claims a unique pair with compare-and-swap
//!
//! sequence exhaustion waits for the wall clock unless the clock moved
//! backwards, when logical time advances to preserve monotonicity
//!
//! restart safety assumes the clock does not move backwards across restarts

use std::sync::atomic::{AtomicU64, Ordering};
use std::thread::sleep;
use std::time::Duration;

const TIMESTAMP_SHIFT: u8 = 22;

/// pause while the current millisecond's sequence is exhausted; sleeping
/// instead of spinning stops a saturated worker from burning its cpu quota
/// while it waits for the clock
const SEQUENCE_EXHAUSTED_WAIT: Duration = Duration::from_micros(50);

/// largest relative timestamp that keeps the assembled id positive
pub const MAX_TIMESTAMP_MILLIS: i64 = (1 << 41) - 1;

/// injectable unix millisecond clock
pub trait Clock: Send + Sync {
    fn now_unix_millis(&self) -> i64;
}

/// lock-free monotonic snowflake id generator
pub struct SnowflakeGenerator<C: Clock> {
    epoch_unix_millis: i64,
    machine_bits: i64,
    sequence_bits: u8,
    max_sequence: u32,
    state: AtomicU64,
    clock: C,
}

impl<C: Clock> SnowflakeGenerator<C> {
    /// validates the machine layout and ids, then constructs a generator
    pub fn new(
        data_center_id: u64,
        worker_id: u32,
        epoch_unix_millis: i64,
        clock: C,
        mpid_bit_size: u8,
        dc_bit_size: u8,
    ) -> Self {
        assert!(
            mpid_bit_size <= 21,
            "MPID_BIT_SIZE must be between 0 and 21"
        );
        assert!(
            dc_bit_size <= mpid_bit_size / 2,
            "DC_BIT_SIZE must be between 0 and {}",
            mpid_bit_size / 2
        );
        let worker_bits = mpid_bit_size - dc_bit_size;
        let max_worker_id = (1u32 << worker_bits) - 1;
        let max_data_center_id = (1u64 << dc_bit_size) - 1;
        assert!(
            worker_id <= max_worker_id,
            "WORKER_ID must be at most {max_worker_id}"
        );
        assert!(
            dc_bit_size == 0 || data_center_id <= max_data_center_id,
            "DATA_CENTER_ID must be at most {max_data_center_id}"
        );

        let sequence_bits = TIMESTAMP_SHIFT - mpid_bit_size;
        let max_sequence = (1u32 << sequence_bits) - 1;
        let data_center_id = if dc_bit_size == 0 { 0 } else { data_center_id };
        let machine_bits = ((data_center_id as i64) << (sequence_bits + worker_bits))
            | ((worker_id as i64) << sequence_bits);
        let initial = clock.now_unix_millis() - epoch_unix_millis;
        SnowflakeGenerator {
            epoch_unix_millis,
            machine_bits,
            sequence_bits,
            max_sequence,
            // waiting one millisecond avoids reusing ids after a restart
            state: AtomicU64::new(((initial.max(0) as u64) << sequence_bits) | max_sequence as u64),
            clock,
        }
    }

    /// emits the next unique increasing snowflake id
    pub fn generate(&self) -> i64 {
        loop {
            let now = self.relative_now();
            let current = self.state.load(Ordering::Acquire);
            let (last_millis, last_sequence) = self.unpack(current);

            if now > last_millis {
                // a clock step between this read and the cas may briefly put ids ahead
                if self.try_claim(current, now, 0) {
                    return self.assemble(now, 0);
                }
            } else if last_sequence < self.max_sequence {
                // keep logical time monotonic when the clock moves backwards
                if self.try_claim(current, last_millis, last_sequence + 1) {
                    return self.assemble(last_millis, last_sequence + 1);
                }
            } else if now == last_millis {
                // wait instead of borrowing from the future under load
                sleep(SEQUENCE_EXHAUSTED_WAIT);
            } else {
                // keep serving after a backward clock step exhausts the sequence
                if self.try_claim(current, last_millis + 1, 0) {
                    return self.assemble(last_millis + 1, 0);
                }
            }
        }
    }

    /// emits `count` snowflake ids
    pub fn generate_batch(&self, count: usize) -> Vec<i64> {
        let mut ids = Vec::with_capacity(count);
        for _ in 0..count {
            ids.push(self.generate());
        }
        ids
    }

    fn try_claim(&self, current: u64, millis: i64, sequence: u32) -> bool {
        self.state
            .compare_exchange_weak(
                current,
                ((millis as u64) << self.sequence_bits) | sequence as u64,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_ok()
    }

    fn relative_now(&self) -> i64 {
        self.clock.now_unix_millis() - self.epoch_unix_millis
    }

    fn assemble(&self, millis: i64, sequence: u32) -> i64 {
        // crossing the 41-bit range would emit a negative id
        assert!(
            (0..=MAX_TIMESTAMP_MILLIS).contains(&millis),
            "timestamp {millis} out of 41-bit range; epoch is too old"
        );
        (millis << TIMESTAMP_SHIFT) | self.machine_bits | sequence as i64
    }

    fn unpack(&self, packed: u64) -> (i64, u32) {
        (
            (packed >> self.sequence_bits) as i64,
            (packed & self.max_sequence as u64) as u32,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    use std::sync::atomic::{AtomicBool, AtomicI64};
    use std::sync::Arc;

    const MAX_SEQUENCE: u32 = (1 << 12) - 1;

    struct MockClock {
        millis: AtomicI64,
    }

    impl MockClock {
        fn new(millis: i64) -> Self {
            Self {
                millis: AtomicI64::new(millis),
            }
        }

        fn set(&self, millis: i64) {
            self.millis.store(millis, Ordering::SeqCst);
        }

        fn advance(&self, delta: i64) {
            self.millis.fetch_add(delta, Ordering::SeqCst);
        }
    }

    impl Clock for MockClock {
        fn now_unix_millis(&self) -> i64 {
            self.millis.load(Ordering::SeqCst)
        }
    }

    impl Clock for Arc<MockClock> {
        fn now_unix_millis(&self) -> i64 {
            self.millis.load(Ordering::SeqCst)
        }
    }

    // lets overflow tests progress past a frozen millisecond
    fn with_advancing_clock<R>(start: i64, body: impl FnOnce(Arc<MockClock>) -> R) -> R {
        let clock = Arc::new(MockClock::new(start));
        let stop = Arc::new(AtomicBool::new(false));
        let ticker = {
            let clock = Arc::clone(&clock);
            let stop = Arc::clone(&stop);
            std::thread::spawn(move || {
                while !stop.load(Ordering::Relaxed) {
                    clock.advance(1);
                    std::thread::yield_now();
                }
            })
        };
        let result = body(Arc::clone(&clock));
        stop.store(true, Ordering::Relaxed);
        ticker.join().unwrap();
        result
    }

    fn decode(id: i64) -> (i64, i64, i64, u32) {
        let timestamp = id >> TIMESTAMP_SHIFT;
        let data_center = (id >> 17) & 0x1F;
        let worker = (id >> 12) & 0x1F;
        let sequence = (id & MAX_SEQUENCE as i64) as u32;
        (timestamp, data_center, worker, sequence)
    }

    #[test]
    fn every_supported_layout_encodes_machine_ids_without_touching_timestamp() {
        for mpid_bits in 0..=21 {
            for dc_bits in 0..=mpid_bits / 2 {
                let worker_bits = mpid_bits - dc_bits;
                let worker_id = (1u32 << worker_bits) - 1;
                let data_center_id = (1u64 << dc_bits) - 1;
                let clock = Arc::new(MockClock::new(1_000));
                let gen = SnowflakeGenerator::new(
                    if dc_bits == 0 {
                        u64::MAX
                    } else {
                        data_center_id
                    },
                    worker_id,
                    0,
                    Arc::clone(&clock),
                    mpid_bits,
                    dc_bits,
                );

                clock.set(1_001);
                let sequence_bits = 22 - mpid_bits;
                for sequence in 0..2 {
                    let id = gen.generate();
                    assert_eq!(id >> 22, 1_001);
                    assert_eq!(
                        (id >> sequence_bits) & ((1 << mpid_bits) - 1),
                        ((data_center_id as i64) << worker_bits) | worker_id as i64,
                        "layout MPID_BIT_SIZE={mpid_bits}, DC_BIT_SIZE={dc_bits}"
                    );
                    assert_eq!(id & ((1 << sequence_bits) - 1), sequence);
                }
            }
        }
    }

    #[test]
    fn sequence_capacity_scales_with_machine_bits() {
        for mpid_bits in [0, 5, 10, 21] {
            let clock = Arc::new(MockClock::new(2_000));
            let gen = SnowflakeGenerator::new(0, 0, 0, Arc::clone(&clock), mpid_bits, 0);
            clock.set(2_001);
            for sequence in 0..(1 << (22 - mpid_bits)) {
                assert_eq!(gen.generate(), (2_001 << 22) | sequence);
            }
            clock.set(2_002);
            assert_eq!(gen.generate(), 2_002 << 22);
        }
    }

    #[test]
    fn one_bit_sequence_waits_after_two_ids_and_survives_clock_rollback() {
        let clock = Arc::new(MockClock::new(3_000));
        let gen = SnowflakeGenerator::new(0, 0, 0, Arc::clone(&clock), 21, 0);
        clock.set(3_001);
        assert_eq!(gen.generate(), 3_001 << 22);
        assert_eq!(gen.generate(), (3_001 << 22) | 1);

        let (sender, receiver) = std::sync::mpsc::channel();
        let handle = std::thread::spawn(move || {
            sender.send(gen.generate()).unwrap();
            gen
        });
        let exhausted = receiver.recv_timeout(Duration::from_millis(20));
        clock.set(3_002);
        let next = receiver.recv_timeout(Duration::from_secs(2)).unwrap();
        let gen = handle.join().unwrap();
        assert_eq!(exhausted, Err(std::sync::mpsc::RecvTimeoutError::Timeout));
        assert_eq!(next, 3_002 << 22);

        clock.set(2_900);
        assert_eq!(gen.generate(), (3_002 << 22) | 1);
        assert_eq!(gen.generate(), 3_003 << 22);
    }

    #[test]
    fn rejects_invalid_layouts_and_machine_ids() {
        for (mpid_bits, dc_bits, data_center_id, worker_id) in [
            (22, 0, 0, 0),
            (0, 1, 0, 0),
            (1, 1, 0, 0),
            (9, 5, 0, 0),
            (21, 11, 0, 0),
            (0, 0, 0, 1),
            (10, 0, 0, 1_024),
            (21, 0, 0, 2_097_152),
            (10, 1, 2, 0),
            (21, 10, 1_024, 0),
        ] {
            assert!(
                std::panic::catch_unwind(|| SnowflakeGenerator::new(
                    data_center_id,
                    worker_id,
                    0,
                    MockClock::new(1_000),
                    mpid_bits,
                    dc_bits,
                ))
                .is_err(),
                "accepted invalid configuration: {mpid_bits}, {dc_bits}, {data_center_id}, {worker_id}"
            );
        }
    }

    #[test]
    fn generates_positive_id_with_expected_bit_layout() {
        let clock = Arc::new(MockClock::new(1_000));
        let gen = SnowflakeGenerator::new(1, 2, 0, Arc::clone(&clock), 10, 5);

        // the first id waits for the next millisecond after startup
        clock.set(1_001);
        let id = gen.generate();
        assert!(id > 0, "id should be positive, got {id}");

        let (timestamp, data_center, worker, _sequence) = decode(id);
        assert_eq!(timestamp, 1_001, "timestamp bits");
        assert_eq!(data_center, 1, "data center bits");
        assert_eq!(worker, 2, "worker bits");
    }

    #[test]
    fn honors_custom_epoch_in_timestamp_bits() {
        let clock = Arc::new(MockClock::new(1_420_070_500_000));
        let gen = SnowflakeGenerator::new(0, 0, 1_420_070_400_000, Arc::clone(&clock), 10, 5);

        clock.set(1_420_070_500_001);
        let (timestamp, _, _, _) = decode(gen.generate());
        assert_eq!(timestamp, 100_001, "timestamp should be relative to epoch");
    }

    #[test]
    fn ids_are_strictly_increasing_within_a_millisecond() {
        let clock = Arc::new(MockClock::new(5_000));
        let gen = SnowflakeGenerator::new(0, 0, 0, Arc::clone(&clock), 10, 5);

        // stay below the sequence limit so the frozen clock cannot stall
        clock.set(5_001);
        let mut previous = gen.generate();
        for _ in 0..(MAX_SEQUENCE as usize - 1) {
            let next = gen.generate();
            assert!(next > previous, "expected {next} > {previous}");
            let (timestamp, _, _, _) = decode(next);
            assert_eq!(timestamp, 5_001, "all ids share the frozen millisecond");
            previous = next;
        }
    }

    #[test]
    fn exhausted_sequence_waits_for_wall_clock_instead_of_drifting() {
        let clock = Arc::new(MockClock::new(7_000));
        let gen = SnowflakeGenerator::new(0, 0, 0, Arc::clone(&clock), 10, 5);

        clock.set(7_001);
        for _ in 0..=MAX_SEQUENCE {
            gen.generate();
        }

        clock.set(7_002);
        let (timestamp, _, _, sequence) = decode(gen.generate());
        assert_eq!(timestamp, 7_002, "must consume exactly one real wall tick");
        assert_eq!(sequence, 0, "a fresh millisecond resets the sequence");
    }

    #[test]
    fn crossing_millisecond_boundaries_stays_unique_and_ordered() {
        with_advancing_clock(8_000, |clock| {
            let gen = SnowflakeGenerator::new(0, 0, 0, Arc::clone(&clock), 10, 5);

            let count = MAX_SEQUENCE as usize * 8;
            let mut previous = gen.generate();
            let mut seen = HashSet::new();
            seen.insert(previous);
            let mut max_timestamp = previous >> TIMESTAMP_SHIFT;

            for _ in 0..count {
                let next = gen.generate();
                assert!(
                    next > previous,
                    "strictly increasing across ms: {next} > {previous}"
                );
                assert!(seen.insert(next), "ids must be unique across ms boundaries");
                max_timestamp = max_timestamp.max(next >> TIMESTAMP_SHIFT);
                previous = next;
            }

            assert!(
                max_timestamp <= clock.now_unix_millis(),
                "the generator must never emit a timestamp ahead of the wall clock"
            );
        });
    }

    #[test]
    fn backward_clock_step_never_duplicates_or_regresses() {
        let clock = Arc::new(MockClock::new(9_000));
        let gen = SnowflakeGenerator::new(3, 4, 0, Arc::clone(&clock), 10, 5);

        clock.set(9_001);
        let mut seen = HashSet::new();
        let mut previous = gen.generate();
        seen.insert(previous);

        clock.set(8_950);

        for _ in 0..1_000 {
            let next = gen.generate();
            assert!(
                next > previous,
                "monotonic across backward step: {next} > {previous}"
            );
            assert!(seen.insert(next), "no duplicates across backward step");
            previous = next;
        }
    }

    #[test]
    fn concurrent_generation_yields_unique_ids() {
        with_advancing_clock(11_000, |clock| {
            let gen = Arc::new(SnowflakeGenerator::new(0, 0, 0, Arc::clone(&clock), 10, 5));

            let per_thread = 20_000;
            let threads = 8;
            let mut handles = Vec::new();
            for _ in 0..threads {
                let gen = Arc::clone(&gen);
                handles.push(std::thread::spawn(move || {
                    (0..per_thread).map(|_| gen.generate()).collect::<Vec<_>>()
                }));
            }

            let mut all = HashSet::new();
            for handle in handles {
                for id in handle.join().unwrap() {
                    assert!(all.insert(id), "concurrent ids must be globally unique");
                }
            }
            assert_eq!(all.len(), threads * per_thread);
        });
    }

    #[test]
    fn generate_batch_returns_requested_count_uniquely() {
        with_advancing_clock(15_000, |clock| {
            let gen = SnowflakeGenerator::new(0, 0, 0, clock, 10, 5);

            let ids = gen.generate_batch(10_000);
            assert_eq!(ids.len(), 10_000);
            assert_eq!(
                ids.iter().collect::<HashSet<_>>().len(),
                10_000,
                "batch ids must be unique"
            );
        });
    }

    #[test]
    fn timestamp_at_max_bound_stays_positive() {
        // seed below the bound so the first id lands exactly on it
        let clock = Arc::new(MockClock::new(MAX_TIMESTAMP_MILLIS - 1));
        let gen = SnowflakeGenerator::new(0, 0, 0, Arc::clone(&clock), 10, 5);

        clock.set(MAX_TIMESTAMP_MILLIS);
        let id = gen.generate();
        assert_eq!(
            id >> TIMESTAMP_SHIFT,
            MAX_TIMESTAMP_MILLIS,
            "lands on the bound"
        );
        assert!(
            id > 0,
            "ids at the 41-bit timestamp bound must stay positive"
        );
    }

    #[test]
    #[should_panic(expected = "out of 41-bit range")]
    fn emitting_past_the_max_bound_panics_instead_of_corrupting() {
        // one millisecond past the bound would produce a negative id
        let clock = Arc::new(MockClock::new(MAX_TIMESTAMP_MILLIS));
        let gen = SnowflakeGenerator::new(0, 0, 0, Arc::clone(&clock), 10, 5);

        clock.set(MAX_TIMESTAMP_MILLIS + 1);
        gen.generate();
    }

    #[test]
    fn first_id_after_construction_waits_past_the_construction_millisecond() {
        // a prior process may have used the construction millisecond
        let clock = Arc::new(MockClock::new(20_000));
        let gen = SnowflakeGenerator::new(0, 0, 0, Arc::clone(&clock), 10, 5);

        clock.set(20_001);
        let (timestamp, _, _, _) = decode(gen.generate());
        assert!(
            timestamp > 20_000,
            "first id must land after the construction millisecond, got {timestamp}"
        );
    }
}
