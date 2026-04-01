use crate::alloc::{Allocator, Global};
use std::{
    alloc::Layout,
    cmp::Ordering,
    fmt,
    hash::{Hash, Hasher},
    marker::PhantomData,
    ops::Deref,
    ptr::NonNull,
};

/// A single-threaded reference-counting pointer. ‘Rc’ stands for ‘Reference Counted’.
///
/// Note: does not currently support DSTs as this would require various unstable library features
/// but [`RcSlice`] and [`RcStr`] may be used instead. Dyn values must be boxed.
pub struct Rc<T, A: Allocator = Global> {
    nn: NonNull<Inner<T, A>>,
}

struct Inner<T, A: Allocator> {
    n: usize, // Clone count
    a: A,
    v: T,
}

impl<T, A: Allocator> Rc<T, A> {
    /// Allocate a new Rc in specified allocator and move v into it.
    pub fn new_in(v: T, a: A) -> Rc<T, A> {
        unsafe {
            let layout = Layout::new::<Inner<T, A>>();
            let nn = a.allocate(layout).unwrap();
            let p = nn.as_ptr().cast::<Inner<T, A>>();

            let inner = Inner { n: 0, a, v };
            std::ptr::write(p, inner);
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
        let ptr: *mut Inner<T, A> = this.nn.as_ptr();
        unsafe { &raw mut (*ptr).v }
    }
}

impl<T, A: Allocator> Clone for Rc<T, A> {
    fn clone(&self) -> Rc<T, A> {
        unsafe {
            let p = self.nn.as_ptr();
            (*p).n += 1;
            let nn = NonNull::new_unchecked(p);
            Self { nn }
        }
    }
}

impl<T, A: Allocator> Drop for Rc<T, A> {
    fn drop(&mut self) {
        unsafe {
            let p = self.nn.as_ptr();
            if (*p).n == 0 {
                self.nn.drop_in_place();
                let layout = Layout::for_value(&*self.nn.as_ptr());
                let dp = NonNull::new_unchecked(p.cast::<u8>());
                (*p).a.deallocate(dp, layout)
            } else {
                (*p).n -= 1;
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

use std::borrow::Borrow;
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
pub struct RcSlice<T, A: Allocator> {
    nn: NonNull<RcSliceInner<A>>,
    pd: PhantomData<T>,
}

struct RcSliceInner<A: Allocator> {
    n: usize,    // Clone count
    slen: usize, // Length of slice
    a: A,
}

impl<T, A: Allocator> RcSlice<T, A> {
    /// Create a RcSlice from s in specified allocator.
    pub fn from_slice_in(s: &[T], a: A) -> Self {
        let slen = s.len();
        unsafe {
            let layout = Layout::new::<RcSliceInner<A>>();
            let (layout, off) = layout
                .extend(Layout::array::<T>(slen).unwrap_unchecked())
                .unwrap_unchecked();

            let nn = a.allocate(layout).unwrap();
            let p = nn.as_ptr().cast::<RcSliceInner<A>>();

            let inner = RcSliceInner { n: 0, slen, a };
            std::ptr::write(p, inner);

            // Copy the slice into the allocated storage.
            let to = p as *mut u8;
            let to = to.add(off) as *mut T;
            let from: *const T = s.as_ptr();
            std::ptr::copy_nonoverlapping(from, to, slen);

            let nn = NonNull::new_unchecked(p);
            Self {
                nn,
                pd: PhantomData,
            }
        }
    }

    fn slice(&self) -> NonNull<[T]>
    {
        unsafe {
            let p = self.nn.as_ptr();
            let slen = (*p).slen;
            let layout = Layout::new::<RcSliceInner<A>>();
            let (_layout, off) = layout
                .extend(Layout::array::<T>(slen).unwrap_unchecked())
                .unwrap_unchecked();
            let p = p as *mut u8;
            let p = p.add(off) as *mut T;

            let nn = NonNull::new_unchecked(p);
            NonNull::slice_from_raw_parts(nn, slen)
        }
    }
}

impl<T, A: Allocator> Deref for RcSlice<T, A> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        unsafe {
            let snn = self.slice();
            &*snn.as_ptr()
        }
    }
}

impl<T, A: Allocator> Clone for RcSlice<T, A> {
    fn clone(&self) -> RcSlice<T, A> {
        unsafe {
            let p = self.nn.as_ptr();
            (*p).n += 1;
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
            let x = &mut (*p);
            if x.n == 0 {

                let snn = self.slice();
                snn.drop_in_place();
                
                let slen = x.slen;
                let layout = Layout::new::<RcSliceInner<A>>();
                let (layout, _off) = layout
                    .extend(Layout::array::<T>(slen).unwrap_unchecked())
                    .unwrap_unchecked();
                let nn = NonNull::new_unchecked(p as *mut u8);
                x.a.deallocate(nn, layout)
            } else {
                x.n -= 1;
            }
        }
    }
}

////////////////////////////////////////////////////////////
/// Reference-counted String.
#[derive(Clone)]
pub struct RcStr<A: Allocator> {
    inner: RcSlice<u8, A>,
}

impl<A: Allocator + Clone> RcStr<A> {
    /// Create a RcStr from s in specified allocator.
    pub fn from_str_in(s: &str, a: A) -> RcStr<A> {
        let inner = RcSlice::from_slice_in(s.as_bytes(), a);
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
    let mut m = lhashmap();
    let x = RcStr::from_str_in("George", Local::new());
    m.insert(x.clone(), 99);
    assert!(m.get("George").is_some());
    println!("x={}", x);

    let rs = RcSlice::from_slice_in(b"George", Local::new());
    let rs1 = rs.clone();

    assert!(rs.deref() == b"George");
    assert!(rs1.deref() == b"George");

    struct D;
    impl Drop for D{ fn drop(&mut self){ println!("Dropped D"); } }


    let data = [ D, D, D ];
    let _rs = RcSlice::from_slice_in( &data, Local::new() );
    
}
