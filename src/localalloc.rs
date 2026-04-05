//!
//! Values allocated from Temp and Local Allocators must be freed by the same thread that allocated them, or the program will abort.
//!
//! Temp is for heap allocations that last a short time.
//!
//! Local is for heap allocations that last a longer time ( but not longer than the thread ).
//!
//! Example
//! ```
//! use pstd::{Box,Vec,localalloc::Local};
//! let b = Box::new_in(98, Local::new());
//! assert!(*b == 98);
//! type LBox<T> = Box::<T,Local>; // Locally allocated Box.
//! let b = LBox::auto(99); // Alternative to using new_in.
//! assert!(*b == 99);
//! type LVec<T> = Vec<T,Local>; // Locally allocated Vec.
//! let mut v = LVec::with_capacity_auto(4); // Pre-allocate space for 4 values.
//! v.push("Hello");
//! assert!(v.pop() == Some("Hello"));
//! ```

use crate::alloc::{AllocError, Allocator, Global};

use std::{
    alloc::Layout,
    cell::RefCell,
    marker::PhantomData,
    mem,
    ptr::{NonNull, null, slice_from_raw_parts_mut},
};

thread_local! {
    static TA: RefCell<BumpAllocator> = RefCell::new(BumpAllocator::new());
    static LA: RefCell<ChainAllocator> = RefCell::new(ChainAllocator::new());
}

const NOT_MIRI: bool = !cfg!(miri);
const K: usize = 1024;
const MAX_ALIGN: usize = 128;
const BSIZE: usize = 1024 * K; // 20 bits
const MAX_SIZE: usize = BSIZE / 16; // 16 bits
const MAX_SC: usize = 14;
const MIN_SIZE: usize = 2 * mem::size_of::<FreeMem>(); // 16 for 64-bit system.
const L2_MIN_SIZE: usize = MIN_SIZE.ilog2() as usize;

/// Temp [`Allocator`].
#[derive(Default, Clone)]
pub struct Temp {
    pd: PhantomData<NonNull<()>>, // To make Temp !Send and !Sync
}

impl Temp {
    /// Create a Temp allocator
    pub const fn new() -> Self {
        Self { pd: PhantomData }
    }
}

unsafe impl Allocator for Temp {
    fn allocate(&self, lay: Layout) -> Result<NonNull<[u8]>, AllocError> {
        TA.with_borrow_mut(|a| a.allocate(lay))
    }

    unsafe fn deallocate(&self, p: NonNull<u8>, lay: Layout) {
        TA.with_borrow_mut(|a| a.deallocate(p, lay));
    }
}

/// Local [`Allocator`].
#[derive(Default, Clone)]
pub struct Local {
    pd: PhantomData<NonNull<()>>, // To make Local !Send and !Sync
}

impl Local {
    /// Create a Local allocator
    pub const fn new() -> Self {
        Self { pd: PhantomData }
    }
}

unsafe impl Allocator for Local {
    fn allocate(&self, lay: Layout) -> Result<NonNull<[u8]>, AllocError> {
        LA.with_borrow_mut(|a| a.allocate(lay))
    }

    unsafe fn deallocate(&self, p: NonNull<u8>, lay: Layout) {
        LA.with_borrow_mut(|a| a.deallocate(p, lay));
    }
}

struct Block(NonNull<u8>);

impl Block {
    fn new() -> Self {
        let lay = Layout::from_size_align(BSIZE, MAX_ALIGN).unwrap();
        let p = Global::allocate(&Global, lay).unwrap();
        let p = p.as_ptr().cast::<u8>();
        let nn = unsafe { NonNull::new_unchecked(p) };
        Self(nn)
    }

    fn alloc(&self, i: usize, n: usize) -> NonNull<[u8]> {
        let p = unsafe { self.0.as_ptr().add(i) };
        let p = slice_from_raw_parts_mut(p, n);
        unsafe { NonNull::new_unchecked(p) }
    }

    fn drop(&mut self) {
        if self.0 != NonNull::dangling() {
            let lay = Layout::from_size_align(BSIZE, MAX_ALIGN).unwrap();
            unsafe { Global::deallocate(&Global, self.0, lay) }
            self.0 = NonNull::dangling();
        }
    }
}

struct BumpAllocator {
    alloc_count: u64,               // Number of current allocations
    idx: usize,                     // Current bytes allocated from cur
    cur: Block,                     // Current block for allocation
    overflow: std::vec::Vec<Block>, // List of used up blocks
    _alloc_bytes: usize,            // Rest are only for diagnostic purposes.
    _max_alloc: usize,
    _reset_count: usize,
    _total_count: usize,
    _total_alloc: usize,
}

impl BumpAllocator {
    fn new() -> Self {
        Self {
            alloc_count: 0,
            idx: 0,
            cur: Block::new(),
            overflow: Vec::new(),
            _alloc_bytes: 0,
            _max_alloc: 0,
            _reset_count: 0,
            _total_count: 0,
            _total_alloc: 0,
        }
    }

    fn allocate(&mut self, lay: Layout) -> Result<NonNull<[u8]>, AllocError> {
        self.alloc_count += 1;
        let (n, m) = (lay.size(), lay.align());
        if NOT_MIRI && n <= MAX_SIZE && m <= MAX_ALIGN {
            #[cfg(feature = "log-bump")]
            {
                self._alloc_bytes += n;
                self._total_count += 1;
                self._total_alloc += n;
            }
            let mut i = self.idx.checked_next_multiple_of(m).unwrap();
            let e = i + n;
            // Make a new block if necessary.
            if e >= BSIZE && (e > BSIZE || n == 0) {
                let old = mem::replace(&mut self.cur, Block::new());
                self.overflow.push(old);
                i = 0;
            }
            self.idx = i + n;
            Ok(self.cur.alloc(i, n))
        } else {
            Global::allocate(&Global, lay)
        }
    }

    fn deallocate(&mut self, p: NonNull<u8>, lay: Layout) {
        self.alloc_count -= 1;
        let (n, m) = (lay.size(), lay.align());
        if NOT_MIRI && n <= MAX_SIZE && m <= MAX_ALIGN {
            if self.alloc_count == 0 {
                #[cfg(feature = "log-bump")]
                {
                    self._reset_count += 1;
                    self._max_alloc = std::cmp::max(self._max_alloc, self._alloc_bytes);
                    self._alloc_bytes = 0;
                }
                self.idx = 0;
                self.reset_overflow();
            }
        } else {
            unsafe {
                Global::deallocate(&Global, p, lay);
            }
        }
    }

    fn reset_overflow(&mut self) {
        while let Some(mut b) = self.overflow.pop() {
            b.drop();
        }
    }
}

impl Drop for BumpAllocator {
    fn drop(&mut self) {
        #[cfg(feature = "log-bump")]
        println!(
            "BumpAllocator Dropping total_count={} total_alloc={} max_alloc={} reset_count={}",
            self._total_count, self._total_alloc, self._max_alloc, self._reset_count
        );

        if self.alloc_count != 0 {
            println!(
                "BumpAllocator has {} outstanding allocations, aborting",
                self.alloc_count,
            );
            std::process::abort();
        }

        self.cur.drop();
        self.reset_overflow();
    }
}

struct FreeMem {
    next: *const FreeMem, // May be null
}

struct ChainAllocator {
    alloc_count: u64,               // Number of current allocations
    idx: usize,                     // Current bytes allocated from cur
    cur: Block,                     // Current block for allocation
    free: [*const FreeMem; MAX_SC], // Address of first free allocation for each size class.
    overflow: std::vec::Vec<Block>, // List of used up blocks
    _alloc_bytes: usize,            // Rest are only for diagnostic purposes.
    _max_alloc: usize,
    _reset_count: usize,
    _total_count: usize,
    _total_alloc: usize,
}

impl ChainAllocator {
    fn new() -> Self {
        Self {
            free: [null(); MAX_SC],
            alloc_count: 0,
            idx: 0,
            cur: Block::new(),
            overflow: Vec::new(),
            _alloc_bytes: 0,
            _max_alloc: 0,
            _reset_count: 0,
            _total_count: 0,
            _total_alloc: 0,
        }
    }

    /// Calculates size class index and size for that class.
    const fn sc(mut n: usize) -> (usize, usize) {
        if n < MIN_SIZE {
            n = MIN_SIZE;
        }
        let sc = ((n - 1).ilog2() + 1) as usize;
        (sc - L2_MIN_SIZE, 2 << sc)
    }

    fn allocate(&mut self, lay: Layout) -> Result<NonNull<[u8]>, AllocError> {
        self.alloc_count += 1;
        let (n,m) = (lay.size(), lay.align());
        if NOT_MIRI && n <= MAX_SIZE && m <= MIN_SIZE {
            #[cfg(feature = "log-bump")]
            {
                self._alloc_bytes += n;
                self._total_count += 1;
                self._total_alloc += n;
            };
            let (sc, xn) = Self::sc(n);
            let p = self.free[sc];
            if !p.is_null() {
                let next = unsafe { (*p).next };
                self.free[sc] = next;
                let p = p as *mut u8;
                let p = slice_from_raw_parts_mut(p, n);
                Ok(unsafe { NonNull::new_unchecked(p) })
            } else {
                let mut i = self.idx;
                let e = i + xn;
                // Make a new block if necessary.
                if e > BSIZE {
                    let old = mem::replace(&mut self.cur, Block::new());
                    self.overflow.push(old);
                    i = 0;
                }
                self.idx = i + xn;
                Ok(self.cur.alloc(i, n)) // Ought to be able to return xn, but that causes problems!
            }
        } else {
            println!("Using Global n={n} m={m}");
            Global::allocate(&Global, lay)
        }
    }

    fn deallocate(&mut self, p: NonNull<u8>, lay: Layout) {
        self.alloc_count -= 1;
        let (n,m) = (lay.size(), lay.align());
        if NOT_MIRI && n <= MAX_SIZE && m <= MIN_SIZE {
            if self.alloc_count == 0 {
                #[cfg(feature = "log-bump")]
                {
                    self._reset_count += 1;
                    self._max_alloc = std::cmp::max(self._max_alloc, self._alloc_bytes);
                    self._alloc_bytes = 0;
                }
                self.idx = 0;
                self.reset_overflow();
                self.free = [null(); MAX_SC];
            } else {
                // Put freed storage on free list.
                let sc = Self::sc(n).0;
                let p = p.as_ptr() as *mut FreeMem;
                unsafe {
                    (*p).next = self.free[sc];
                }
                self.free[sc] = p;
            }
        } else {
            unsafe {
                Global::deallocate(&Global, p, lay);
            }
        }
    }

    fn reset_overflow(&mut self) {
        while let Some(mut b) = self.overflow.pop() {
            b.drop();
        }
    }
}

impl Drop for ChainAllocator {
    fn drop(&mut self) {
        #[cfg(feature = "log-bump")]
        println!(
            "ChainAllocator Dropping total_count={} total_alloc={} max_alloc={} reset_count={}",
            self._total_count, self._total_alloc, self._max_alloc, self._reset_count
        );

        if self.alloc_count != 0 {
            println!(
                "ChainAllocator has {} outstanding allocations, aborting",
                self.alloc_count,
            );
            std::process::abort();
        }

        self.cur.drop();
        self.reset_overflow();
    }
}

#[test]
fn test_alloc() {
    let x = crate::Box::new_in(99, Local::new());
    assert!(*x == 99);
    {
        let x = crate::Box::new_in(99, Local::new());
        assert!(*x == 99);
    }
    let x = crate::Box::new_in(99, Local::new());
    assert!(*x == 99);
}
