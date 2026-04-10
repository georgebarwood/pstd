//!
//! [`Temp`](localalloc::Temp) is for heap allocations that last a short time.
//!
//! [`Local`](localalloc::Local) is for heap allocations that last a longer time ( but not longer than the thread ).
//!
//! [`Perm`](localalloc::Perm) is for heap allocations that are expected to last longer than a thread.
//!
//! Temp and Local are not Send or Sync, so values allocated from them cannot be accessed by other threads.
//!
//! Temp uses "bump" allocation, deallocate just decreases a count of outstanding allocations.
//! This means there is no minimum allocation internally.
//! Allocations larger than 64K bytes, or having more than 128 byte alignment, are routed to [`Global`](alloc::Global).
//!
//! Local has an array of free lists, one for each size class 16,32,64...64K.
//! The minimum internal allocation (on a 64-bit system) is 16 bytes, which is also the maximum alignment.
//! Allocations larger than 64K bytes, or having more than 16 byte alignment, are routed to [`Global`](alloc::Global).
//!
//! Perm is similar to Local, except there is a single instance per process. It implements both Allocator and GlobalAlloc
//! so can be used to set a Global Allocator using the `#[global_allocator]` attribute.
//!
//! Example
//! ```
//! use pstd::{BoxA,VecA,localalloc::{Local,Temp}};
//! let b = BoxA::new_in(98, Local::new());
//! assert!(*b == 98);
//! type TBox<T> = BoxA::<T,Temp>; // Temp allocated Box.
//! let b = TBox::new(99);
//! assert!(*b == 99);
//! type LVec<T> = VecA<T,Local>; // Locally allocated Vec.
//! let mut v = LVec::with_capacity(4); // Pre-allocate space for 4 values.
//! v.push("Hello");
//! assert!(v.pop() == Some("Hello"));
//! ```

use crate::alloc::{AllocError, Allocator, System, GlobalAlloc};

use std::{
    alloc::Layout,
    cell::RefCell,
    marker::PhantomData,
    mem,
    ptr::{NonNull, null, slice_from_raw_parts_mut},
    sync::LazyLock,
};

thread_local! {
    static TA: RefCell<BumpAllocator> = RefCell::new(BumpAllocator::new());
    static LA: RefCell<ChainAllocator> = RefCell::new(ChainAllocator::new());
}

static mut PA: LazyLock<ChainAllocator> = LazyLock::new(|| { ChainAllocator::new() } );

const MIRI: bool = cfg!(miri);
const K: usize = 1024;
const MAX_ALIGN: usize = 128;
const BSIZE: usize = 1024 * K; // 20 bits = 1MB
const MAX_SIZE: usize = BSIZE / 16; // 16 bits = 64K
const MAX_SC: usize = 1 + (MAX_SIZE.ilog2() as usize) - L2_MIN_SIZE;
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

/// Perm [`Allocator`].
#[derive(Default, Clone)]
pub struct Perm;

impl Perm {
    /// Create a Perm allocator
    pub const fn new() -> Self {
        Self { }
    }
}

unsafe impl Allocator for Perm {
    fn allocate(&self, lay: Layout) -> Result<NonNull<[u8]>, AllocError> {
        let a = &raw mut PA;
        let a = unsafe{ LazyLock::<ChainAllocator>::force_mut(&mut (*a)) };
        a.allocate(lay)
    }

    unsafe fn deallocate(&self, p: NonNull<u8>, lay: Layout) {
        let a = &raw mut PA;
        let a = unsafe{ LazyLock::<ChainAllocator>::force_mut(&mut (*a)) };
        a.deallocate(p, lay);
    }
}

unsafe impl GlobalAlloc for Perm {
    unsafe fn alloc(&self, lay: Layout) -> *mut u8 {
        let nn =self.allocate(lay).unwrap();
        let p : * mut [u8] = nn.as_ptr();
        let p : * mut u8 = p.cast::<u8>();
        p
    }

    unsafe fn dealloc(&self, p: *mut u8, lay: Layout) {
        unsafe {
            let nn = NonNull::new_unchecked(p);
            self.deallocate(nn, lay);
        }
    }
}

struct Block(NonNull<u8>);

impl Block {
    fn new() -> Self {
        let lay = Layout::from_size_align(BSIZE, MAX_ALIGN).unwrap();
        let p = System::allocate(&System, lay).unwrap();
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
            unsafe { System::deallocate(&System, self.0, lay) }
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
        if MIRI || n > MAX_SIZE || m > MAX_ALIGN {
            System::allocate(&System, lay)
        } else {
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
        }
    }

    fn deallocate(&mut self, p: NonNull<u8>, lay: Layout) {
        let (n, m) = (lay.size(), lay.align());
        if MIRI || n > MAX_SIZE || m > MAX_ALIGN {
            unsafe {
                System::deallocate(&System, p, lay);
            }
        } else {
            self.alloc_count -= 1;
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
            "BumpAllocator Dropping alloc_count={} total_count={} total_alloc={} max_alloc={} reset_count={}",
            self.alloc_count,
            self._total_count,
            self._total_alloc,
            self._max_alloc,
            self._reset_count
        );
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
            alloc_count: 0,
            idx: 0,
            cur: Block::new(),
            free: [null(); MAX_SC],
            overflow: Vec::new(),
            _alloc_bytes: 0,
            _max_alloc: 0,
            _reset_count: 0,
            _total_count: 0,
            _total_alloc: 0,
        }
    }

    /// Calculate size class index and size for n-byte storage request.
    const fn size_class(mut n: usize) -> (usize, usize) {
        if n < MIN_SIZE {
            n = MIN_SIZE;
        }
        let sc = ((n - 1).ilog2() + 1) as usize;
        (sc - L2_MIN_SIZE, 2 << (sc - 1))
    }

    fn allocate(&mut self, lay: Layout) -> Result<NonNull<[u8]>, AllocError> {
        let (n, m) = (lay.size(), lay.align());
        if MIRI || n > MAX_SIZE || m > MIN_SIZE {
            System::allocate(&System, lay)
        } else {
            self.alloc_count += 1;
            #[cfg(feature = "log-bump")]
            {
                self._alloc_bytes += n;
                self._total_count += 1;
                self._total_alloc += n;
            };
            let (sc, xn) = Self::size_class(n);

            let p = self.free[sc];

            // println!("ChainAllocator::allocate n={n} m={m} sc={sc} xn={xn} ix={} p={}", self.idx, p as usize);

            if !p.is_null() {
                // Remove p from free list and return it.
                let next = unsafe { (*p).next };
                self.free[sc] = next;
                let p = p as *mut u8;
                let p: *mut [u8] = slice_from_raw_parts_mut(p, xn);
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
                Ok(self.cur.alloc(i, xn))
            }
        }
    }

    fn deallocate(&mut self, p: NonNull<u8>, lay: Layout) {
        let (n, m) = (lay.size(), lay.align());
        if MIRI || n > MAX_SIZE || m > MIN_SIZE {
            unsafe {
                System::deallocate(&System, p, lay);
            }
        } else {
            self.alloc_count -= 1;
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
                let (sc, _xn) = Self::size_class(n);
                let p = p.as_ptr() as *mut FreeMem;

                // println!("ChainAllocator::deallocate n={n} m={m} sc={sc} xn={_xn} ix={} p={}", self.idx, p as usize);

                unsafe {
                    (*p).next = self.free[sc];
                }
                self.free[sc] = p;
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
            "ChainAllocator Dropping alloc_count={} total_count={} total_alloc={} max_alloc={} reset_count={}",
            self.alloc_count,
            self._total_count,
            self._total_alloc,
            self._max_alloc,
            self._reset_count
        );
        self.cur.drop();
        self.reset_overflow();
    }
}

unsafe impl Send for ChainAllocator{}
unsafe impl Sync for ChainAllocator{}

#[test]
fn test_alloc() {
    let x = crate::BoxA::new_in(99, Local::new());
    assert!(*x == 99);
    {
        let x = crate::BoxA::new_in(99, Local::new());
        assert!(*x == 99);
    }
    let x = crate::BoxA::new_in(99, Local::new());
    assert!(*x == 99);
}

#[test]
fn test_perm_alloc() {
    type PBox<T> = crate::BoxA<T,Perm>;
    let x = PBox::new(99);
    assert!( *x == 99 );
}
