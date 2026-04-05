use crate::alloc::{Allocator, Global};
use std::{
    alloc::Layout,
    borrow::Borrow,
    cmp::Ordering,
    fmt,
    hash::{Hash, Hasher},
    marker::PhantomData,
    mem::MaybeUninit,
    ops::Deref,
    ptr,
    ptr::NonNull,
};

/// A single-threaded reference-counting pointer. ‘Rc’ stands for ‘Reference Counted’.
///
/// Note: does not currently support DSTs as this would require various unstable library features
/// but [`RcSlice`] and [`RcStr`] may be used instead. Dyn values must be boxed.
pub struct Rc<T, A: Allocator = Global> {
    nn: NonNull<RcInner<T, A>>,
}

struct RcInner<T, A: Allocator> {
    cc: usize, // Clone count
    a: A,
    v: T,
}

impl<T, A: Allocator> Rc<T, A> {
    /// Allocate a new Rc and move v into it.
    pub fn auto(v: T) -> Self
    where
        A: Default,
    {
        Self::new_in(v, A::default())
    }

    /// Allocate a new Rc in specified allocator and move v into it.
    pub fn new_in(v: T, a: A) -> Self {
        unsafe {
            let layout = Layout::new::<RcInner<T, A>>();
            let nn = a.allocate(layout).unwrap();
            let p = nn.as_ptr().cast::<RcInner<T, A>>();

            let inner = RcInner { cc: 0, a, v };
            ptr::write(p, inner);
            let nn = NonNull::new_unchecked(p);
            Self { nn }
        }
    }

    /// Returns a reference to the underlying allocator.
    pub fn allocator(this: &Self) -> &A {
        let p = this.nn.as_ptr();
        unsafe { &(*p).a }
    }

    /// Provides a raw pointer to the data.
    pub fn as_ptr(this: &Self) -> *const T {
        let p = this.nn.as_ptr();
        unsafe { &(*p).v }
    }
}

impl<T, A: Allocator> Clone for Rc<T, A> {
    fn clone(&self) -> Rc<T, A> {
        unsafe {
            let p = self.nn.as_ptr();
            (*p).cc += 1;
            let nn = NonNull::new_unchecked(p);
            Self { nn }
        }
    }
}

impl<T, A: Allocator> Drop for Rc<T, A> {
    fn drop(&mut self) {
        unsafe {
            let p = self.nn.as_ptr();
            if (*p).cc == 0 {
                ptr::drop_in_place(&mut (*p).v);
                let a = ptr::read(&(*p).a);
                // let layout = Layout::for_value(&*p); // Would need this if T was ?Sized
                let layout = Layout::new::<RcInner<T, A>>();
                let nn = NonNull::new_unchecked(p.cast::<u8>());
                a.deallocate(nn, layout)
            } else {
                (*p).cc -= 1;
            }
        }
    }
}

impl<T, A: Allocator> Deref for Rc<T, A> {
    type Target = T;
    fn deref(&self) -> &T {
        let p = self.nn.as_ptr();
        unsafe { &(*p).v }
    }
}

impl<T, A: Allocator> Borrow<T> for Rc<T, A> {
    fn borrow(&self) -> &T {
        self.deref()
    }
}

impl<T: Eq, A: Allocator> Eq for Rc<T, A> {}

impl<T: PartialEq, A: Allocator> PartialEq for Rc<T, A> {
    fn eq(&self, other: &Self) -> bool {
        PartialEq::eq(self.deref(), other.deref())
    }
}

impl<T: Ord, A: Allocator> Ord for Rc<T, A> {
    fn cmp(&self, other: &Self) -> Ordering {
        Ord::cmp(self.deref(), other.deref())
    }
}

impl<T: PartialOrd, A: Allocator> PartialOrd for Rc<T, A> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        PartialOrd::partial_cmp(self.deref(), other.deref())
    }
}

impl<T: Hash, A: Allocator> Hash for Rc<T, A> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.deref().hash(state);
    }
}

impl<T: fmt::Display, A: Allocator> fmt::Display for Rc<T, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self.deref(), f)
    }
}

impl<T: fmt::Debug, A: Allocator> fmt::Debug for Rc<T, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self.deref(), f)
    }
}

////////////////////////////////////////////

/// Reference-counted slice.
pub struct RcSlice<T, A: Allocator = Global> {
    nn: NonNull<RcSliceInner<A>>,
    pd: PhantomData<T>,
}

struct RcSliceInner<A: Allocator> {
    cc: usize,   // Clone count
    slen: usize, // Length of slice
    a: A,
}

impl<T, A: Allocator> RcSlice<T, A> {
    /// Create a RcSlice from s in specified allocator.
    pub fn new_in(s: &[T], a: A) -> Self
    where
        T: Clone,
    {
        unsafe {
            let slen = s.len();
            let (layout, off) = Self::layout(slen);

            let nn = a.allocate(layout).unwrap();
            let p = nn.as_ptr().cast::<RcSliceInner<A>>();

            let inner = RcSliceInner { cc: 0, slen, a };
            ptr::write(p, inner);

            // Initialise the slice of allocated memory.
            let to = (p as *mut u8).add(off) as *mut MaybeUninit<T>;
            let to = std::slice::from_raw_parts_mut(to, slen);
            to.write_clone_of_slice(s);

            Self {
                nn: NonNull::new_unchecked(p),
                pd: PhantomData,
            }
        }
    }

    /// Get the layout for the inner fields followed by a slice of size slen, and the offset of the slice.
    fn layout(slen: usize) -> (Layout, usize) {
        unsafe {
            Layout::new::<RcSliceInner<A>>()
                .extend(Layout::array::<T>(slen).unwrap_unchecked())
                .unwrap_unchecked()
        }
    }

    /// Get a NonNull pointer to the slice.
    fn slice(&self) -> NonNull<[T]> {
        unsafe {
            let p = self.nn.as_ptr();
            let slen = (*p).slen;
            let (_layout, off) = Self::layout(slen);
            let p = (p as *mut u8).add(off) as *mut T;
            let nn = NonNull::new_unchecked(p);
            NonNull::slice_from_raw_parts(nn, slen)
        }
    }
}

impl<T, A: Allocator> Clone for RcSlice<T, A> {
    fn clone(&self) -> Self {
        unsafe {
            let p = self.nn.as_ptr();
            (*p).cc += 1;
        }
        Self {
            nn: self.nn,
            pd: PhantomData,
        }
    }
}

impl<T, A: Allocator> Drop for RcSlice<T, A> {
    fn drop(&mut self) {
        unsafe {
            let p = self.nn.as_ptr();
            if (*p).cc == 0 {
                self.slice().drop_in_place();
                let slen = (*p).slen;
                let a = ptr::read(&(*p).a);
                let (layout, _off) = Self::layout(slen);
                let nn = NonNull::new_unchecked(p as *mut u8);
                a.deallocate(nn, layout)
            } else {
                (*p).cc -= 1;
            }
        }
    }
}

impl<T, A: Allocator> Deref for RcSlice<T, A> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        unsafe { self.slice().as_ref() }
    }
}

////////////////////////////////////////////////////////////
/// Reference-counted String.
#[derive(Clone)]
pub struct RcStr<A: Allocator = Global> {
    inner: RcSlice<u8, A>,
}

impl<A: Allocator> RcStr<A> {
    /// Create a RcStr from s
    pub fn auto(s: &str) -> Self
    where
        A: Default,
    {
        Self::new_in(s, A::default())
    }

    /// Create a RcStr from s in specified allocator.
    pub fn new_in(s: &str, a: A) -> Self {
        let inner = RcSlice::new_in(s.as_bytes(), a);
        Self { inner }
    }
}

impl<A: Allocator> Deref for RcStr<A> {
    type Target = str;
    fn deref(&self) -> &str {
        let b = self.inner.deref();
        unsafe { str::from_utf8_unchecked(b) }
    }
}

impl<A: Allocator> Borrow<str> for RcStr<A> {
    fn borrow(&self) -> &str {
        self.deref()
    }
}

impl<A: Allocator> Ord for RcStr<A> {
    fn cmp(&self, other: &Self) -> Ordering {
        Ord::cmp(self.deref(), other.deref())
    }
}

impl<A: Allocator> PartialOrd for RcStr<A> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<A: Allocator> Eq for RcStr<A> {}

impl<A: Allocator> PartialEq for RcStr<A> {
    fn eq(&self, other: &Self) -> bool {
        PartialEq::eq(self.deref(), other.deref())
    }
}

impl<A: Allocator> Hash for RcStr<A> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.deref().hash(state);
    }
}

impl<A: Allocator> fmt::Display for RcStr<A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self.deref(), f)
    }
}

impl<A: Allocator> fmt::Debug for RcStr<A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self.deref(), f)
    }
}

#[test]
fn rc_test() {
    use crate::localalloc::*;
    use crate::*;
    let mut m = collections::HashMap::new_in(Local::new());
    let x = RcStr::<Global>::auto("George");
    m.insert(x.clone(), 99);
    assert!(m.get("George").is_some());
    println!("x={}", x);

    let rs = RcSlice::new_in(b"George", Local::new());
    let rs1 = rs.clone();

    assert!(rs.deref() == b"George");
    assert!(rs1.deref() == b"George");

    #[derive(Clone)]
    struct D;
    impl Drop for D {
        fn drop(&mut self) {
            println!("Dropped D");
        }
    }

    let data = [D, D, D];
    let _rs = RcSlice::new_in(&data, Local::new());
}
