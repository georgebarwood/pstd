//!
//! [`Local`](localalloc::Local) is for thread-local allocations.
//!
//! [`Temp`](localalloc::Temp) is for thread-local allocations that last a short time.
//!
//! [`Perm`](localalloc::Perm) is for non-thread-local allocations.
//!
//! [`GTemp`](localalloc::GTemp) is for non-thread-local allocations that do not persist permanently.
//!
//! Example
//! ```
//! use pstd::{BoxA,VecA,localalloc::{Local,Temp}};
//! type TBox<T> = BoxA::<T,Temp>; // Temp allocated Box.
//! let b = TBox::new(99);
//! assert!(*b == 99);
//! type LVec<T> = VecA<T,Local>; // Locally allocated Vec.
//! let mut v = LVec::with_capacity(4); // Pre-allocate space for 4 values.
//! v.push("Hello");
//! assert!(v.pop() == Some("Hello"));
//! ```

use crate::{
    VecA,
    alloc::{AllocError, Allocator, GlobalAlloc, System},
};

use std::{
    alloc::Layout,
    cell::RefCell,
    marker::PhantomData,
    mem,
    ptr::{NonNull, null, slice_from_raw_parts_mut},
    sync::Mutex,
};

const MIRI: bool = cfg!(miri);

const L2_MIN_SIZE: usize = 1 + mem::size_of::<FreeMem>().ilog2() as usize;

/// Maximum alignment for [`Temp`].
pub const MAX_ALIGN: usize = 128;

/// Size of the blocks fetched from [`System`] allocator.
pub const BLOCK_SIZE: usize = 1 << 18;

/// Minimum size allocation for [`Local`] and [`Perm`] = 16 for a 64-bit system.
pub const MIN_SIZE: usize = 1 << L2_MIN_SIZE;

/// Maximum size allocation.
pub const MAX_SIZE: usize = BLOCK_SIZE / 16;

/// Number of size classes.
pub const NUM_SC: usize = 1 + (MAX_SIZE.ilog2() as usize) - L2_MIN_SIZE;

/// Allocator state info.
#[non_exhaustive]
#[derive(Debug)]
pub struct Info {
    /// Number of outstanding allocations.
    pub alloc_count: u64,
    /// Allocation index for current block.
    pub idx: usize,
    /// Number of overflow blocks.
    pub overflow_len: usize,
    /// Length of free chain for each size class.
    pub free_len: [usize; NUM_SC],
}

thread_local! {
    static TA: RefCell<BumpAllocator> = RefCell::new(BumpAllocator::new());
    static LA: RefCell<ChainAllocator> = RefCell::new(ChainAllocator::new());
}

/// Temp is for thread-local allocations that last a short time.
/// Temp uses "bump" allocation, deallocate just decreases a count of outstanding allocations.
/// This means there is no minimum allocation internally.
/// Allocations larger than [`MAX_SIZE`] bytes, or having more than [`MAX_ALIGN`] alignment, are routed to [`System`].
#[derive(Default, Clone, Debug)]
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

/// Local is for thread-local allocations.
/// Local has an array of free lists, one for each size class [`MIN_SIZE`]..[`MAX_SIZE`].
/// Allocations larger than [`MAX_SIZE`] bytes, or having more than [`MIN_SIZE`] alignment, are routed to [`System`].
#[derive(Default, Clone, Debug)]
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

static PA: Mutex<Option<ChainAllocator>> = Mutex::new(None);

/// Perm is for non-thread-local allocations.
/// Perm is similar to [Local], except there is a single instance per process. It implements both [`Allocator`] and [`GlobalAlloc`]
/// so can be used to set a Global Allocator using the `#[global_allocator]` attribute.
#[derive(Default, Clone, Debug)]
pub struct Perm;

impl Perm {
    /// Create a Perm allocator
    pub const fn new() -> Self {
        Self {}
    }

    /// Get the number of outstanding allocations.
    pub fn alloc_count() -> u64 {
        let a = PA.lock().unwrap();
        if a.is_none() {
            return 0;
        }
        let a = a.as_ref().unwrap();
        a.alloc_count
    }

    /// Get allocator state info.
    ///
    /// # Example
    /// ```
    /// use pstd::{BoxA,localalloc::Perm};
    /// let b = BoxA::<_,Perm>::new(99);
    /// println!( "Perm::info = {:?}", Perm::info() );
    ///```
    pub fn info() -> Option<Info> {
        PA.lock().unwrap().as_ref().map(|a| a.info())
    }
}

unsafe impl Allocator for Perm {
    fn allocate(&self, lay: Layout) -> Result<NonNull<[u8]>, AllocError> {
        let mut a = PA.lock().unwrap();
        if a.is_none() {
            *a = Some(ChainAllocator::new());
        }
        let a = a.as_mut().unwrap();
        a.allocate(lay)
    }

    unsafe fn deallocate(&self, p: NonNull<u8>, lay: Layout) {
        let mut a = PA.lock().unwrap();
        let a = a.as_mut().unwrap();
        a.deallocate(p, lay);
    }
}

unsafe impl GlobalAlloc for Perm {
    unsafe fn alloc(&self, lay: Layout) -> *mut u8 {
        let nn = self.allocate(lay).unwrap();
        let p: *mut [u8] = nn.as_ptr();
        let p: *mut u8 = p.cast::<u8>();
        p
    }

    unsafe fn dealloc(&self, p: *mut u8, lay: Layout) {
        unsafe {
            let nn = NonNull::new_unchecked(p);
            self.deallocate(nn, lay);
        }
    }

    // ToDo : implement realloc rather than use the supplied function.
}

static GTA: Mutex<Option<ChainAllocator>> = Mutex::new(None);

/// GTemp is for non-thread-local allocations that do not persist permanently.
#[derive(Default, Clone, Debug)]
pub struct GTemp;

impl GTemp {
    /// Create a GTemp allocator
    pub const fn new() -> Self {
        Self {}
    }

    /// Get the number of outstanding allocations.
    pub fn alloc_count() -> u64 {
        let a = GTA.lock().unwrap();
        if a.is_none() {
            return 0;
        }
        let a = a.as_ref().unwrap();
        a.alloc_count
    }

    /// Get allocator state info.
    ///
    /// # Example
    /// ```
    /// use pstd::{BoxA,localalloc::GTemp};
    /// let b = BoxA::<_,GTemp>::new(99);
    /// println!( "GTemp::info = {:?}", GTemp::info() );
    ///```
    pub fn info() -> Option<Info> {
        GTA.lock().unwrap().as_ref().map(|a| a.info())
    }
}

unsafe impl Allocator for GTemp {
    fn allocate(&self, lay: Layout) -> Result<NonNull<[u8]>, AllocError> {
        let mut a = GTA.lock().unwrap();
        if a.is_none() {
            *a = Some(ChainAllocator::new());
        }
        let a = a.as_mut().unwrap();
        a.allocate(lay)
    }

    unsafe fn deallocate(&self, p: NonNull<u8>, lay: Layout) {
        let mut a = GTA.lock().unwrap();
        let a = a.as_mut().unwrap();
        a.deallocate(p, lay);
    }
}

struct Block(NonNull<u8>);

impl Block {
    fn alloc(&self, i: usize, n: usize) -> NonNull<[u8]> {
        let p = unsafe { self.0.as_ptr().add(i) };
        let p = slice_from_raw_parts_mut(p, n);
        unsafe { NonNull::new_unchecked(p) }
    }
}

impl Default for Block {
    fn default() -> Self {
        let lay = Layout::from_size_align(BLOCK_SIZE, MAX_ALIGN).unwrap();
        let p = System::allocate(&System, lay).unwrap();
        let p = p.as_ptr().cast::<u8>();
        let nn = unsafe { NonNull::new_unchecked(p) };
        Self(nn)
    }
}

impl Drop for Block {
    fn drop(&mut self) {
        let lay = Layout::from_size_align(BLOCK_SIZE, MAX_ALIGN).unwrap();
        unsafe { System::deallocate(&System, self.0, lay) }
    }
}

type SVec<T> = VecA<T, System>;

#[derive(Default)]
struct BumpAllocator {
    alloc_count: u64,      // Number of current allocations
    idx: usize,            // Current bytes allocated from cur
    cur: Block,            // Current block for allocation
    overflow: SVec<Block>, // List of used up blocks
    _alloc_bytes: usize,   // Rest are only for diagnostic purposes.
    _max_alloc: usize,
    _reset_count: usize,
    _total_count: usize,
    _total_alloc: usize,
}

impl BumpAllocator {
    fn new() -> Self {
        Self::default()
    }

    fn allocate(&mut self, lay: Layout) -> Result<NonNull<[u8]>, AllocError> {
        let (n, m) = (lay.size(), lay.align());
        if MIRI || n > MAX_SIZE || m > MAX_ALIGN {
            System::allocate(&System, lay)
        } else {
            self.alloc_count += 1;
            #[cfg(feature = "log-alloc")]
            {
                self._alloc_bytes += n;
                self._total_count += 1;
                self._total_alloc += n;
            }
            let mut i = self.idx.checked_next_multiple_of(m).unwrap();
            let e = i + n;
            // Make a new block if necessary.
            if e >= BLOCK_SIZE && (e > BLOCK_SIZE || n == 0) {
                let old = mem::take(&mut self.cur);
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
                #[cfg(feature = "log-alloc")]
                {
                    self._reset_count += 1;
                    self._max_alloc = std::cmp::max(self._max_alloc, self._alloc_bytes);
                    self._alloc_bytes = 0;
                }
                self.idx = 0;
                self.overflow.clear();
            }
        }
    }
}

impl Drop for BumpAllocator {
    fn drop(&mut self) {
        #[cfg(feature = "log-alloc")]
        println!(
            "BumpAllocator Dropping alloc_count={} total_count={} total_alloc={} max_alloc={} reset_count={}",
            self.alloc_count,
            self._total_count,
            self._total_alloc,
            self._max_alloc,
            self._reset_count
        );
    }
}

struct FreeMem {
    next: *const FreeMem, // May be null
}

#[derive(Default)]
struct ChainAllocator {
    alloc_count: u64,               // Number of current allocations
    idx: usize,                     // Current bytes allocated from cur
    cur: Block,                     // Current block for allocation
    free: [*const FreeMem; NUM_SC], // Address of first free allocation for each size class.
    overflow: SVec<Block>,          // List of used up blocks
    _alloc_bytes: usize,            // Rest are only for diagnostic purposes.
    _max_alloc: usize,
    _reset_count: usize,
    _total_count: usize,
    _total_alloc: usize,
}

impl ChainAllocator {
    fn new() -> Self {
        Self::default()
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
            #[cfg(feature = "log-alloc")]
            {
                self._alloc_bytes += n;
                self._total_count += 1;
                self._total_alloc += n;
            }
            let (sc, xn) = Self::size_class(n);
            let p = self.free[sc];
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
                if e > BLOCK_SIZE {
                    let old = mem::take(&mut self.cur);
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
                #[cfg(feature = "log-alloc")]
                {
                    self._reset_count += 1;
                    self._max_alloc = std::cmp::max(self._max_alloc, self._alloc_bytes);
                    self._alloc_bytes = 0;
                }
                self.idx = 0;
                self.overflow.clear();
                self.free = [null(); NUM_SC];
            } else {
                // Put freed storage on free list.
                let (sc, _xn) = Self::size_class(n);
                let p = p.as_ptr() as *mut FreeMem;
                unsafe {
                    (*p).next = self.free[sc];
                }
                self.free[sc] = p;
            }
        }
    }

    fn info(&self) -> Info {
        let mut free_len = [0; NUM_SC];
        for (i, n) in free_len.iter_mut().enumerate() {
            let mut p = self.free[i];
            while !p.is_null() {
                *n += 1;
                p = unsafe { (*p).next };
            }
        }
        Info {
            alloc_count: self.alloc_count,
            idx: self.idx,
            overflow_len: self.overflow.len(),
            free_len,
        }
    }
}

impl Drop for ChainAllocator {
    fn drop(&mut self) {
        #[cfg(feature = "log-alloc")]
        println!(
            "ChainAllocator Dropping alloc_count={} total_count={} total_alloc={} max_alloc={} reset_count={}",
            self.alloc_count,
            self._total_count,
            self._total_alloc,
            self._max_alloc,
            self._reset_count
        );
    }
}

unsafe impl Send for ChainAllocator {}
unsafe impl Sync for ChainAllocator {}

#[test]
fn test_box_local_alloc() {
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
    type PBox<T> = crate::BoxA<T, Perm>;
    let x = PBox::new(99);
    assert!(*x == 99);
}

#[test]
fn test_alloc_free() {
    let mut v = VecA::<_, Perm>::new();
    for _ in 0..1000 {
        v.push(0);
    }
    println!("v len={}", v.len());
    println!("Perm::info = {:?}", Perm::info());
}
