use crate::alloc::{Allocator, Global};
use std::{
    alloc::Layout,
    cmp::Ordering,
    fmt,
    hash::{Hash, Hasher},
    ops::{Deref, DerefMut},
    ptr,
    ptr::NonNull,
};

/// A pointer type that uniquely owns a heap allocation of type `T`.
///
/// Boxing dyn values requires the dynbox feature to be enabled ( which currently requires the nightly toolchain ).
pub struct BoxA<T: ?Sized, A: Allocator = Global> {
    pub(crate) nn: NonNull<T>,
    pub(crate) a: A,
}

/// Box allocated from Global.
pub type Box<T> = BoxA<T, Global>;

impl<T, A: Allocator> BoxA<T, A> {
    /// Allocates memory then places t into it.
    #[must_use]
    pub fn new(t: T) -> Self
    where
        A: Default,
    {
        Self::new_in(t, A::default())
    }

    /// Allocates memory in the given allocator then places t into it.
    pub fn new_in(t: T, a: A) -> Self {
        let layout = Layout::new::<T>();
        let nn = a.allocate(layout).unwrap();
        let nn = unsafe { NonNull::<T>::new_unchecked(nn.as_ptr().cast::<T>()) };
        unsafe {
            ptr::write(nn.as_ptr(), t);
        }
        Self { nn, a }
    }

    /// Allocates memory in the given allocator then clones s into it.
    pub fn from_slice_in(s: &[T], a: A) -> BoxA<[T], A>
    where
        T: Clone,
    {
        let n = s.len();
        let layout = Layout::array::<T>(n).unwrap();
        let nn = a.allocate(layout).unwrap();
        let p = nn.as_ptr().cast::<T>();
        for (i, e) in s.iter().enumerate() {
            unsafe {
                ptr::write(p.add(i), e.clone());
            }
        }
        let nn = unsafe { NonNull::new_unchecked(p) };
        let nn = NonNull::slice_from_raw_parts(nn, n);
        BoxA::<[T], A> { nn, a }
    }
}

impl<T: ?Sized, A: Allocator> BoxA<T, A> {
    /// Allocates memory then copies s into it.
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> BoxA<str, A>
    where
        A: Default,
    {
        BoxA::<str, A>::from_str_in(s, A::default())
    }
}

impl<T: ?Sized, A: Allocator> BoxA<T, A> {
    /// Allocates memory in the given allocator then copies s into it.
    ///
    /// Note: there is currently no equivalent in the standard library.
    pub fn from_str_in(s: &str, a: A) -> BoxA<str, A> {
        let n = s.len();
        let layout = Layout::array::<u8>(n).unwrap();
        let nn: NonNull<[u8]> = a.allocate(layout).unwrap();
        let p: *mut u8 = nn.as_ptr().cast::<u8>();

        unsafe {
            ptr::copy_nonoverlapping(s.as_ptr(), p, n);
        }

        // Need to trim any over-allocation!
        let p = unsafe { std::slice::from_raw_parts_mut(p, n) };
        let nn: NonNull<[u8]> = unsafe { NonNull::new_unchecked(p) };
        let p: *mut str = nn.as_ptr() as *mut str;
        let nn: NonNull<str> = unsafe { NonNull::new_unchecked(p) };

        BoxA::<str, A> { nn, a }
    }
}

impl<T: ?Sized, A: Allocator> BoxA<T, A> {
    fn r(&self) -> &T {
        unsafe { &*self.nn.as_ptr() }
    }
}

impl<A: Allocator + Clone> Clone for BoxA<str, A> {
    fn clone(&self) -> BoxA<str, A> {
        BoxA::<str, A>::from_str_in(self.r(), self.a.clone())
    }
}

impl<T: ?Sized + Hash, A: Allocator> Hash for BoxA<T, A> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (**self).hash(state);
    }
}

impl<T: ?Sized + Eq, A: Allocator> Eq for BoxA<T, A> {}

impl<T: ?Sized + PartialEq, A: Allocator> PartialEq for BoxA<T, A> {
    fn eq(&self, other: &Self) -> bool {
        PartialEq::eq(&**self, &**other)
    }
}

impl<T: ?Sized + Ord, A: Allocator> Ord for BoxA<T, A> {
    fn cmp(&self, other: &Self) -> Ordering {
        Ord::cmp(&**self, &**other)
    }
}

impl<T: ?Sized + PartialOrd, A: Allocator> PartialOrd for BoxA<T, A> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        PartialOrd::partial_cmp(&**self, &**other)
    }
}

unsafe impl<T: Send, A: Allocator + Send> Send for BoxA<T, A> {}
unsafe impl<T: Sync, A: Allocator + Send> Sync for BoxA<T, A> {}

impl<T: ?Sized, A: Allocator> Drop for BoxA<T, A> {
    fn drop(&mut self) {
        unsafe {
            let layout = Layout::for_value(&*self.nn.as_ptr());
            self.nn.drop_in_place();
            let p = NonNull::new(self.nn.as_ptr().cast::<u8>()).unwrap();
            self.a.deallocate(p, layout);
        }
    }
}

impl<T: ?Sized, A: Allocator> Deref for BoxA<T, A> {
    type Target = T;

    fn deref(&self) -> &T {
        self.r()
    }
}

impl<T: ?Sized, A: Allocator> DerefMut for BoxA<T, A> {
    fn deref_mut(&mut self) -> &mut T {
        unsafe { &mut *self.nn.as_ptr() }
    }
}

impl<T: ?Sized + fmt::Display, A: Allocator> fmt::Display for BoxA<T, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self.r(), f)
    }
}

impl<T: ?Sized + fmt::Debug, A: Allocator> fmt::Debug for BoxA<T, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self.r(), f)
    }
}

use std::borrow::Borrow;
impl<T: ?Sized, A: Allocator> Borrow<T> for BoxA<T, A> {
    fn borrow(&self) -> &T {
        self
    }
}

use std::borrow::BorrowMut;
impl<T: ?Sized, A: Allocator> BorrowMut<T> for BoxA<T, A> {
    fn borrow_mut(&mut self) -> &mut T {
        self
    }
}

#[cfg(feature = "dynbox")]
use std::{marker::Unsize, ops::CoerceUnsized};

#[cfg(feature = "dynbox")]
impl<T: ?Sized + Unsize<U>, U: ?Sized, A: Allocator> CoerceUnsized<BoxA<U, A>> for BoxA<T, A> {}
