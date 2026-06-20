use crate::alloc::{Allocator, Global};
use std::{
    alloc::Layout,
    borrow::Borrow,
    cmp::Ordering,
    fmt,
    hash::{Hash, Hasher},
    ops::Deref,
    ptr,
    ptr::NonNull,
};
use std::sync::atomic;

/// Arc allocated from Global.
pub type Arc<T> = ArcA<T, Global>;

/// A multi-threaded reference-counting pointer. 
pub struct ArcA<T, A: Allocator> {
    nn: NonNull<ArcInner<T, A>>,
}

struct ArcInner<T, A: Allocator> {
    cc: atomic::AtomicUsize, // Clone count
    a: A,
    v: T,
}

impl<T, A: Allocator> ArcA<T, A> {
    /// Allocate a new Arc and move v into it.
    pub fn new(v: T) -> Self
    where
        A: Default,
    {
        Self::new_in(v, A::default())
    }

    /// Allocate a new Arc in specified allocator and move v into it.
    pub fn new_in(v: T, a: A) -> Self {
        unsafe {
            let layout = Layout::new::<ArcInner<T, A>>();
            let nn = a.allocate(layout).unwrap();
            let p = nn.as_ptr().cast::<ArcInner<T, A>>();

            let inner = ArcInner { cc: 1.into(), a, v };
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

    /// Returns a mutable reference to the contained value if data is not shared.
    ///
    /// Otherwise returns None.
    ///
    pub fn get_mut(rc: &mut Self) -> Option<&mut T> {
        unsafe {
            let p = rc.nn.as_ptr();
            let cc = (*p).cc.load(atomic::Ordering::Relaxed);
            atomic::fence(atomic::Ordering::Acquire);
            if cc == 1 {
                Some(&mut (*p).v)
            } else {
                None
            }
        }
    }
}

unsafe impl<T: Sync + Send> Send for Arc<T> {}
unsafe impl<T: Sync + Send> Sync for Arc<T> {}

impl<T, A: Allocator> Clone for ArcA<T, A> {
    fn clone(&self) -> Self {
        unsafe {
            let p = self.nn.as_ptr();
            (*p).cc.fetch_add(1, atomic::Ordering::Relaxed);
            let nn = NonNull::new_unchecked(p);
            Self { nn }
        }
    }
}

impl<T, A: Allocator> Drop for ArcA<T, A> {
    fn drop(&mut self) {
        unsafe {
            let p = self.nn.as_ptr();
            if (*p).cc.fetch_sub(1, atomic::Ordering::Release) == 1 {
                atomic::fence(atomic::Ordering::Acquire);
                ptr::drop_in_place(&mut (*p).v);
                let a = ptr::read(&(*p).a);
                // let layout = Layout::for_value(&*p); // Would need this if T was ?Sized
                let layout = Layout::new::<ArcInner<T, A>>();
                let nn = NonNull::new_unchecked(p.cast::<u8>());
                a.deallocate(nn, layout)
            }
        }
    }
}

impl<T, A: Allocator> Deref for ArcA<T, A> {
    type Target = T;
    fn deref(&self) -> &T {
        let p = self.nn.as_ptr();
        unsafe { &(*p).v }
    }
}

impl<T, A: Allocator> Borrow<T> for ArcA<T, A> {
    fn borrow(&self) -> &T {
        self.deref()
    }
}

impl<T: Eq, A: Allocator> Eq for ArcA<T, A> {}

impl<T: PartialEq, A: Allocator> PartialEq for ArcA<T, A> {
    fn eq(&self, other: &Self) -> bool {
        PartialEq::eq(self.deref(), other.deref())
    }
}

impl<T: Ord, A: Allocator> Ord for ArcA<T, A> {
    fn cmp(&self, other: &Self) -> Ordering {
        Ord::cmp(self.deref(), other.deref())
    }
}

impl<T: PartialOrd, A: Allocator> PartialOrd for ArcA<T, A> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        PartialOrd::partial_cmp(self.deref(), other.deref())
    }
}

impl<T: Hash, A: Allocator> Hash for ArcA<T, A> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.deref().hash(state);
    }
}

impl<T: fmt::Display, A: Allocator> fmt::Display for ArcA<T, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self.deref(), f)
    }
}

impl<T: fmt::Debug, A: Allocator> fmt::Debug for ArcA<T, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self.deref(), f)
    }
}

