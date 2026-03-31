use crate::Box;
use crate::alloc::{Allocator, Global};
use std::{
    alloc::Layout,
    cmp::Ordering,
    fmt,
    hash::{Hash, Hasher},
    ops::Deref,
    ptr::{NonNull},
};

/// A single-threaded reference-counting pointer. ‘Rc’ stands for ‘Reference Counted’.
///
/// Note: does not currently support DSTs as this would require various unstable library features.
pub struct Rc<T, A: Allocator = Global> {
    nn: NonNull<Inner<T, A>>,
}

struct Inner<T, A: Allocator> {
    n: usize,
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
}

impl<T, A: Allocator> Rc<T, A> {
    /// Returns a reference to the underlying allocator.
    pub fn allocator(this: &Self) -> &A {
        let p = this.nn.as_ptr();
        unsafe { &(*p).a }
    }

    /// Provides a raw pointer to the data.
    pub fn as_ptr(this: &Self) -> *const T {
        let ptr: *mut Inner<T,A> = this.nn.as_ptr();
        unsafe { &raw mut (*ptr).v }
    }
}

impl<T, A: Allocator> Deref for Rc<T, A> {
    type Target = T;
    fn deref(&self) -> &T {
        let p = self.nn.as_ptr();
        unsafe { &(*p).v }
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
                let dp = NonNull::new(p.cast::<u8>()).unwrap();
                (*p).a.deallocate(dp, layout)
            } else {
                (*p).n -= 1;
            }
        }
    }
}

impl<T: Hash, A: Allocator> Hash for Rc<T, A> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (**self).hash(state);
    }
}

impl<T: Eq, A: Allocator> Eq for Rc<T, A> {}

impl<T: PartialEq, A: Allocator> PartialEq for Rc<T, A> {
    fn eq(&self, other: &Self) -> bool {
        PartialEq::eq(&**self, &**other)
    }
}

impl<T: Ord, A: Allocator> Ord for Rc<T, A> {
    fn cmp(&self, other: &Self) -> Ordering {
        Ord::cmp(&**self, &**other)
    }
}

impl<T: PartialOrd, A: Allocator> PartialOrd for Rc<T, A> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        PartialOrd::partial_cmp(&**self, &**other)
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

use std::borrow::Borrow;
impl<T, A: Allocator> Borrow<T> for Rc<T, A> {
    fn borrow(&self) -> &T {
        self.deref()
    }
}

////////////////////////////////////////////////////////////
/// Reference-counted String.
#[derive(Clone)]
pub struct RcStr<A: Allocator> {
    inner: Rc<Box<str, A>, A>, // The Box could be eliminated eventually when various library features stabilise.
}

impl<A: Allocator + Clone> RcStr<A> {
    /// Create a RcStr from s in specified allocator.
    pub fn from_str_in(s: &str, a: A) -> RcStr<A> {
        let s = Box::<str, A>::from_str_in(s, a.clone());
        let inner = Rc::new_in(s, a);
        RcStr::<A> { inner }
    }
}

impl<A: Allocator> Deref for RcStr<A> {
    type Target = str;
    fn deref(&self) -> &str {
        // let b : &Box<str,A> = self.inner.deref();
        // b.deref()
        self.inner.deref().deref()
    }
}

impl<A: Allocator> Hash for RcStr<A> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.deref().hash(state);
    }
}

impl<A: Allocator> PartialOrd for RcStr<A> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<A: Allocator> Ord for RcStr<A> {
    fn cmp(&self, other: &Self) -> Ordering {
        Ord::cmp(&**self, &**other)
    }
}

impl<A: Allocator> Eq for RcStr<A> {}

impl<A: Allocator> PartialEq for RcStr<A> {
    fn eq(&self, other: &Self) -> bool {
        PartialEq::eq(&**self, &**other)
    }
}

impl<A: Allocator> Borrow<str> for RcStr<A> {
    fn borrow(&self) -> &str {
        self.deref()
    }
}

#[test]
fn rc_test() {
    use crate::localalloc::*;
    let mut m = lhashmap();
    let x = RcStr::from_str_in("George", Local::new());
    m.insert(x.clone(), 99);
    assert!(m.get("George").is_some())
}
