//  Copyright 2026 Robert Zavalczki
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

use std::ops::{BitXor, Shr};
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::Duration;

/// Enable/disable the memory locking feature
const USE_MEM_LOCKS: bool = false;
/// the number of locks, MEM_LOCK_COUNT must be a power of 2
const MEM_LOCK_COUNT: usize = 64;

pub struct MemBlockLocker {
    flags: [AtomicBool; MEM_LOCK_COUNT],
}

impl Default for MemBlockLocker {
    fn default() -> Self {
        Self::new()
    }
}

impl MemBlockLocker {
    pub fn new() -> MemBlockLocker {
        MemBlockLocker {
            flags: std::array::from_fn(|_| AtomicBool::new(false)),
        }
    }

    /// multiplicative hashing, https://en.wikipedia.org/wiki/Hash_function
    fn mhash(k: usize) -> usize {
        const M_BITS: u32 = MEM_LOCK_COUNT.ilog2();
        const WORD_BITS: usize = std::mem::size_of::<usize>() * 8;
        const SHIFT: usize = WORD_BITS - M_BITS as usize;
        k.bitxor(k.shr(SHIFT)).wrapping_mul(9143527).shr(SHIFT)
    }

    pub fn lock(&self, pos: usize) {
        if !USE_MEM_LOCKS {
            return;
        }

        let mut poll_count: u32 = 0;
        let bucket = Self::mhash(pos);
        let flag = &self.flags[bucket];
        loop {
            if flag
                .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
                .is_ok()
            {
                return;
            }

            while flag.load(Ordering::Relaxed) {
                if poll_count < 10 {
                    thread::yield_now();
                } else {
                    thread::sleep(Duration::from_micros(poll_count as u64));
                }
                poll_count = poll_count.wrapping_add(1);
            }
        }
    }

    pub fn unlock(&self, pos: usize) {
        if !USE_MEM_LOCKS {
            return;
        }
        let bucket = Self::mhash(pos);
        self.flags[bucket].store(false, Ordering::Release);
    }
}
