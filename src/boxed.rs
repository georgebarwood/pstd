use crate::alloc::{Allocator, Global};
use std::alloc::Layout;
use std::mem;
use std::ops::Deref;
use std::ops::DerefMut;
use std::ptr;
use std::ptr::NonNull;

/// A pointer type that uniquely owns a heap allocation of type `T`.
pub struct Box<T, A: Allocator = Global> {
    nn: NonNull<T>,
    a: A,
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

impl<T, A: Allocator> Drop for Box<T, A> {
    fn drop(&mut self) {
        if mem::size_of::<T>() > 0 {
            let layout = Layout::new::<T>();
            let p = NonNull::new(self.nn.as_ptr().cast::<u8>()).unwrap();
            unsafe { self.a.deallocate(p, layout) }
        }
    }
}

impl<T, A: Allocator> Deref for Box<T, A> {
    type Target = T;

    fn deref(&self) -> &T {
        unsafe { &*self.nn.as_ptr() }
    }
}

impl<T, A: Allocator> DerefMut for Box<T, A> {
    fn deref_mut(&mut self) -> &mut T {
        unsafe { &mut *self.nn.as_ptr() }
    }
}
