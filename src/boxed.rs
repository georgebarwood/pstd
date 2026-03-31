use crate::alloc::{Allocator, Global};
use std::alloc::Layout;
use std::cmp::Ordering;
use std::fmt;
use std::hash::Hash;
use std::hash::Hasher;
use std::ops::Deref;
use std::ops::DerefMut;
use std::ptr;
use std::ptr::NonNull;

/// A pointer type that uniquely owns a heap allocation of type `T`.
///
/// Boxing dyn values requires the dynbox feature to be enabled ( which currently requires the nightly toolchain ).
pub struct Box<T: ?Sized, A: Allocator = Global> {
    pub(crate) nn: NonNull<T>,
    pub(crate) a: A,
}

impl<T> Box<T> {
    /// Allocates memory on the heap and then places x into it.
    ///
    /// ```
    /// use pstd::Box;
    /// let b = Box::new(99);
    /// assert_eq!( *b, 99 );
    /// ```
    pub fn new(t: T) -> Self {
        Self::new_in(t, Global)
    }
}

impl<T, A: Allocator> Box<T, A> {
    /// Allocates memory in the given allocator then places x into it.
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
    pub fn from_slice_in(s: &[T], a: A) -> Box<[T], A>
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
        Box::<[T], A> { nn, a }
    }
}

impl<T: ?Sized, A: Allocator> Box<T, A> {
    /// Allocates memory in the given allocator then copies s into it.
    ///
    /// Note: there is currently no equivalent in the standard library.
    pub fn from_str_in(s: &str, a: A) -> Box<str, A> {
        let n = s.len();
        let layout = Layout::array::<u8>(n).unwrap();
        let nn: NonNull<[u8]> = a.allocate(layout).unwrap();
        let p: *mut u8 = nn.as_ptr().cast::<u8>();

        unsafe {
            ptr::copy_nonoverlapping(s.as_ptr(), p, n);
        }

        let p: *mut str = nn.as_ptr() as *mut str;
        let nn: NonNull<str> = unsafe { NonNull::new_unchecked(p) };
        Box::<str, A> { nn, a }
    }
}

impl<T: ?Sized, A: Allocator> Box<T, A> {
    fn r(&self) -> &T {
        unsafe { &*self.nn.as_ptr() }
    }
}

impl<A: Allocator + Clone> Clone for Box<str, A> {
    fn clone(&self) -> Box<str, A> {
        Box::<str, A>::from_str_in(self.r(), self.a.clone())
    }
}

impl<T: ?Sized + Hash, A: Allocator> Hash for Box<T, A> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (**self).hash(state);
    }
}

impl<T: ?Sized + Eq, A: Allocator> Eq for Box<T, A> {}

impl<T: ?Sized + PartialEq, A: Allocator> PartialEq for Box<T, A> {
    fn eq(&self, other: &Self) -> bool {
        PartialEq::eq(&**self, &**other)
    }
}

impl<T: ?Sized + Ord, A: Allocator> Ord for Box<T, A> {
    fn cmp(&self, other: &Self) -> Ordering {
        Ord::cmp(&**self, &**other)
    }
}

impl<T: ?Sized + PartialOrd, A: Allocator> PartialOrd for Box<T, A> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        PartialOrd::partial_cmp(&**self, &**other)
    }
}

unsafe impl<T: Send, A: Allocator + Send> Send for Box<T, A> {}
unsafe impl<T: Sync, A: Allocator + Send> Sync for Box<T, A> {}

impl<T: ?Sized, A: Allocator> Drop for Box<T, A> {
    fn drop(&mut self) {
        unsafe {
            let layout = Layout::for_value(&*self.nn.as_ptr());
            self.nn.drop_in_place();
            let p = NonNull::new(self.nn.as_ptr().cast::<u8>()).unwrap();
            self.a.deallocate(p, layout);
        }
    }
}

impl<T: ?Sized, A: Allocator> Deref for Box<T, A> {
    type Target = T;

    fn deref(&self) -> &T {
        self.r()
    }
}

impl<T: ?Sized, A: Allocator> DerefMut for Box<T, A> {
    fn deref_mut(&mut self) -> &mut T {
        unsafe { &mut *self.nn.as_ptr() }
    }
}

impl<T: ?Sized + fmt::Display, A: Allocator> fmt::Display for Box<T, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self.r(), f)
    }
}

impl<T: ?Sized + fmt::Debug, A: Allocator> fmt::Debug for Box<T, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self.r(), f)
    }
}

use std::borrow::Borrow;
impl<T: ?Sized, A: Allocator> Borrow<T> for Box<T, A> {
    fn borrow(&self) -> &T {
        self
    }
}

use std::borrow::BorrowMut;
impl<T: ?Sized, A: Allocator> BorrowMut<T> for Box<T, A> {
    fn borrow_mut(&mut self) -> &mut T {
        self
    }
}

#[cfg(feature = "dynbox")]
use std::{marker::Unsize, ops::CoerceUnsized};

#[cfg(feature = "dynbox")]
impl<T: ?Sized + Unsize<U>, U: ?Sized, A: Allocator> CoerceUnsized<Box<U, A>> for Box<T, A> {}

#[test]
fn test_boxed() {
    struct D(usize, usize, usize);
    impl Drop for D {
        fn drop(&mut self) {
            println!("Dropping D value={} {} {}", self.0, self.1, self.2);
        }
    }

    {
        let b = Box::new(D(99, 1, 2));
        assert_eq!(b.0, 99);
        println!("b.0={}", b.0);
    }

    {
        let b = Box::<str>::from_str_in("Hello There", Global);
        assert_eq!("Hello There", &*b);
    }

    {
        use crate::localalloc::*;
        let mut m = lhashmap();
        let s = lboxstr("Hello George");
        m.insert(s, 1);
        let v = m.get("Hello George");
        println!("v={:?}", v);
        assert_eq!(v, Some(&1));
    }

    {
        use crate::localalloc::*;
        let mut m = lbtreemap();
        let s = lboxstr("Hello George");
        m.insert(s, 1);
        let v = m.get("Hello George");
        println!("v={:?}", v);
        assert_eq!(v, Some(&1));
    }

    #[cfg(feature = "dynbox")]
    {
        trait E {
            fn say_hello(&self);
        }

        impl E for D {
            fn say_hello(&self) {
                println!("Hello");
            }
        }

        type Ep = Box<dyn E>;
        let x: Ep = Box::new(D(999, 1000, 1001));
        x.say_hello();
    }
}
