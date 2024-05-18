use std::alloc::Layout;
use std::error::Error;
use std::fmt;
use std::ptr;
use std::ptr::NonNull;

// use std::alloc::LayoutError;

/// The `AllocError` error indicates an allocation failure
/// that may be due to resource exhaustion or to
/// something wrong when combining the given input arguments with this
/// allocator.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct AllocError;

impl Error for AllocError {}

impl fmt::Display for AllocError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("memory allocation failed")
    }
}

/// Memory allocation.
///
/// # Safety
///
/// ToDo...
pub unsafe trait Allocator {
    /// Attempts to allocate a block of memory.
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError>;

    /// Behaves like `allocate`, but also ensures that the returned memory is zero-initialized.
    /// # Safety
    ///
    /// ToDo...
    fn allocate_zeroed(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        let ptr = self.allocate(layout)?;
        let len = ptr.len();
        unsafe {
            ptr.as_ptr().cast::<u8>().write_bytes(0, len);
        }

        Ok(ptr)
    }

    /// Deallocates the memory referenced by `ptr`.
    /// # Safety
    ///
    /// ToDo...
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout);

    /// Attempts to extend the memory block.
    /// # Safety
    ///
    /// ToDo...
    unsafe fn grow(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        debug_assert!(
            new_layout.size() >= old_layout.size(),
            "`new_layout.size()` must be greater than or equal to `old_layout.size()`"
        );

        let new_ptr = self.allocate(new_layout)?;

        let len = old_layout.size();

        ptr::copy_nonoverlapping(
            ptr.as_ptr().cast::<u8>(),
            new_ptr.as_ptr().cast::<u8>(),
            len,
        );
        self.deallocate(ptr, old_layout);

        Ok(new_ptr)
    }

    /// Behaves like `grow`, but also ensures that the new contents are set to zero before being
    /// returned.
    /// # Safety
    ///
    /// ToDo...
    unsafe fn grow_zeroed(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        debug_assert!(
            new_layout.size() >= old_layout.size(),
            "`new_layout.size()` must be greater than or equal to `old_layout.size()`"
        );

        let new_ptr = self.allocate_zeroed(new_layout)?;

        let len = old_layout.size();

        ptr::copy_nonoverlapping(
            ptr.as_ptr().cast::<u8>(),
            new_ptr.as_ptr().cast::<u8>(),
            len,
        );
        self.deallocate(ptr, old_layout);

        Ok(new_ptr)
    }

    /// Attempts to shrink the memory block.
    /// # Safety
    ///
    /// ToDo...
    unsafe fn shrink(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        debug_assert!(
            new_layout.size() <= old_layout.size(),
            "`new_layout.size()` must be smaller than or equal to `old_layout.size()`"
        );

        let new_ptr = self.allocate(new_layout)?;
        let len = new_layout.size();
        ptr::copy_nonoverlapping(
            ptr.as_ptr().cast::<u8>(),
            new_ptr.as_ptr().cast::<u8>(),
            len,
        );
        self.deallocate(ptr, old_layout);
        Ok(new_ptr)
    }

    /// Creates a "by reference" adapter for this instance of `Allocator`.
    #[inline(always)]
    fn by_ref(&self) -> &Self
    where
        Self: Sized,
    {
        self
    }
}

unsafe impl<A> Allocator for &A
where
    A: Allocator + ?Sized,
{
    #[inline]
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        (**self).allocate(layout)
    }

    #[inline]
    fn allocate_zeroed(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        (**self).allocate_zeroed(layout)
    }

    #[inline]
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        // SAFETY: the safety contract must be upheld by the caller
        unsafe { (**self).deallocate(ptr, layout) }
    }

    #[inline]
    unsafe fn grow(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        // SAFETY: the safety contract must be upheld by the caller
        unsafe { (**self).grow(ptr, old_layout, new_layout) }
    }

    #[inline]
    unsafe fn grow_zeroed(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        // SAFETY: the safety contract must be upheld by the caller
        unsafe { (**self).grow_zeroed(ptr, old_layout, new_layout) }
    }

    #[inline]
    unsafe fn shrink(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        // SAFETY: the safety contract must be upheld by the caller
        unsafe { (**self).shrink(ptr, old_layout, new_layout) }
    }
}

/// Default implementation of Allocator.
#[derive(Clone, Default)]
pub struct Global {}
unsafe impl Allocator for Global {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        match layout.size() {
            0 => Ok(NonNull::slice_from_raw_parts(NonNull::dangling(), 0)),
            size => unsafe {
                let raw_ptr = std::alloc::alloc(layout);
                let ptr = NonNull::new(raw_ptr).ok_or(AllocError)?;
                Ok(NonNull::slice_from_raw_parts(ptr, size))
            },
        }
    }
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        std::alloc::dealloc(ptr.as_ptr(), layout);
    }
}
