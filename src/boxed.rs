use crate::alloc::{Allocator, Global};
use std::alloc::Layout;
use std::fmt;
use std::mem;
use std::ops::Deref;
use std::ops::DerefMut;
use std::ptr;
use std::ptr::NonNull;

/// A pointer type that uniquely owns a heap allocation of type `T`.
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
        let nn = if mem::size_of::<T>() > 0 {
            let layout = Layout::new::<T>();
            let nn = a.allocate(layout).unwrap();
            let nn = NonNull::<T>::new(nn.as_ptr().cast::<T>()).unwrap();
            unsafe {
                ptr::write(nn.as_ptr(), t);
            }
            nn
        } else {
            NonNull::<T>::dangling()
        };
        Self { nn, a }
    }
}

impl<T: ?Sized, A: Allocator> Box<T, A> {
    fn r(&self) -> &T {
        unsafe { &*self.nn.as_ptr() }
    }
}

unsafe impl<T: Send, A: Allocator + Send> Send for Box<T, A> {}
unsafe impl<T: Sync, A: Allocator + Send> Sync for Box<T, A> {}

impl<T: ?Sized, A: Allocator> Drop for Box<T, A> {
    fn drop(&mut self) {
        let layout = unsafe { Layout::for_value(&*self.nn.as_ptr()) };
        unsafe {
            self.nn.drop_in_place();
        }
        if layout.size() != 0 {
            let p = NonNull::new(self.nn.as_ptr().cast::<u8>()).unwrap();
            unsafe { self.a.deallocate(p, layout) }
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
