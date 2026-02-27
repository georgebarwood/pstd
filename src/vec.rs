//! Not possible under stable Rust: Vec::const_make_global, Vec::into_boxed_slice (?)
//!
//! ToDo : peek_mut (may not do that one),  various trait impls.
//!
//! Ideas : have features which allow exclusion of unstable features, methods which can panic.
//!
//! What about more non-panic methods: try_insert, index, index_mut

use crate::alloc::{AllocError, Allocator, Global};

use std::{
    alloc::Layout,
    cmp,
    // cmp::Ordering,
    fmt,
    fmt::Debug,
    iter::FusedIterator,
    mem,
    mem::ManuallyDrop,
    mem::MaybeUninit,
    ops::{Bound, Deref, DerefMut, RangeBounds},
    ptr,
    ptr::NonNull,
    slice,
};

/// A vector that grows as elements are pushed onto it similar to similar to [`std::vec::Vec`].
///
/// Implementation sections: [Construct](#construct-with-default-allocator)
/// [Basic](#basic-methods)
/// [Advanced](#advanced-methods)
/// [Allocation](#allocation-methods)
/// [Conversion](#conversion-methods)
/// [Non-Panic](#non-panic-methods)
pub struct Vec<T, A: Allocator = Global> {
    len: usize,
    cap: usize,
    nn: NonNull<T>,
    alloc: A,
}

/// # Construct with default allocator
impl<T> Vec<T> {
    /// Create a new Vec.
    ///
    /// # Example
    ///
    /// ```
    /// use pstd::vec::Vec;
    /// let mut v = Vec::new();
    /// v.push("England");
    /// v.push("France");
    /// assert!( v.len() == 2 );
    /// for s in &v { println!("s={}",s); }
    /// ```
    #[must_use]
    pub const fn new() -> Vec<T> {
        Vec::new_in(Global)
    }
}

/// # Basic methods
///
/// Properties / methods that operate on one element at a time.
impl<T, A: Allocator> Vec<T, A> {
    /// Returns the number of elements.
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the vector contains no elements.
    ///
    /// # Example
    ///
    /// ```
    /// use pstd::vec::Vec;
    /// let mut v = Vec::new();
    /// assert!(v.is_empty());
    ///
    /// v.push(1);
    /// assert!(!v.is_empty());
    /// ```
    pub const fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Push a value onto the end of the vec.
    pub fn push(&mut self, value: T) {
        if self.cap == self.len {
            let nc = if self.cap == 0 { 4 } else { self.cap * 2 };
            self.set_capacity(nc).unwrap();
        }
        unsafe {
            self.set(self.len, value);
        }
        self.len += 1;
    }

    /// Pop a value from the end of the vec.
    pub fn pop(&mut self) -> Option<T> {
        if self.len == 0 {
            None
        } else {
            self.len -= 1;
            unsafe { Some(self.get(self.len)) }
        }
    }

    /// Removes and returns the last element from a vector if the predicate
    /// returns `true`, or [`None`] if the predicate returns false or the vector
    /// is empty (the predicate will not be called in that case).
    pub fn pop_if(&mut self, predicate: impl FnOnce(&mut T) -> bool) -> Option<T> {
        let last = self.last_mut()?;
        if predicate(last) { self.pop() } else { None }
    }

    /// Insert value at index, after moving elements up to make a space.
    /// # Panics
    ///
    /// Panics if `index` > len().
    ///
    pub fn insert(&mut self, index: usize, value: T) {
        assert!(index <= self.len);
        if self.cap == self.len {
            let na = if self.cap == 0 { 4 } else { self.cap * 2 };
            self.set_capacity(na).unwrap();
        }
        unsafe {
            if index < self.len {
                ptr::copy(self.ixp(index), self.ixp(index + 1), self.len - index);
            }
            self.set(index, value);
        }
        self.len += 1;
    }

    /// Insert value at index, after moving elements up to make a space, returning mut ref to inserted element.
    /// # Panics
    ///
    /// Panics if `index` > len().
    ///
    pub fn insert_mut(&mut self, index: usize, value: T) -> &mut T {
        assert!(index <= self.len);
        if self.cap == self.len {
            let na = if self.cap == 0 { 4 } else { self.cap * 2 };
            self.set_capacity(na).unwrap();
        }
        unsafe {
            if index < self.len {
                ptr::copy(self.ixp(index), self.ixp(index + 1), self.len - index);
            }
            self.set(index, value);
        }
        self.len += 1;
        unsafe { &mut *self.ixp(index) }
    }

    /// Remove the value at index, elements are moved down to fill the space.
    /// # Panics
    ///
    /// Panics if `index` >= len().
    ///
    pub fn remove(&mut self, index: usize) -> T {
        assert!(index < self.len);
        unsafe {
            let result = self.get(index);
            ptr::copy(self.ixp(index + 1), self.ixp(index), self.len() - index - 1);
            self.len -= 1;
            result
        }
    }

    /// Removes an element from the vector and returns it.
    ///
    /// The removed element is replaced by the last element of the vector.
    pub fn swap_remove(&mut self, index: usize) -> T {
        assert!(index < self.len);
        unsafe {
            let result = self.get(index);
            self.len -= 1;
            if index != self.len {
                let last = self.get(self.len);
                self.set(index, last);
            }
            result
        }
    }
}

/// # Advanced methods
///
/// These operate on multiple elements.
impl<T, A: Allocator> Vec<T, A> {
    /// Move all the elements of `other` into `self`, leaving `other` empty.
    /// # Example
    ///
    /// ```
    /// use pstd::vec;
    /// let mut v = vec![1,2,3];
    /// let mut v2 = vec![4,5,6];
    /// v.append(&mut v2);
    /// assert_eq!(v, [1, 2, 3, 4, 5, 6]);
    /// ```
    pub fn append(&mut self, other: &mut Self) {
        self.reserve(other.len);
        unsafe {
            ptr::copy_nonoverlapping(other.ixp(0), self.ixp(self.len), other.len);
        }
        self.len += other.len;
        other.len = 0;
    }

    /// Clones and appends all elements in a slice to the `Vec`.
    pub fn extend_from_slice(&mut self, other: &[T])
    where
        T: Clone,
    {
        for e in other {
            self.push(e.clone());
        }
    }

    /// Given a range `src`, clones a slice of elements in that range and appends it to the end.
    pub fn extend_from_within<R>(&mut self, src: R)
    where
        T: Clone,
        R: RangeBounds<usize>,
    {
        /*
        let start = match src.start_bound() {
            Bound::Included(x) => *x,
            Bound::Excluded(x) => *x + 1,
            Bound::Unbounded => 0,
        };
        let end = match src.end_bound() {
            Bound::Included(x) => *x + 1,
            Bound::Excluded(x) => *x,
            Bound::Unbounded => self.len,
        };
        */
        let (start, end) = self.get_range(src);

        for i in start..end {
            let e = self[i].clone();
            self.push(e);
        }
    }

    /// Creates a splicing iterator that replaces the specified range in the vector
    /// with the given `replace_with` iterator and yields the removed items.
    /// `replace_with` does not need to be the same length as `range`.
    ///
    /// `range` is removed even if the `Splice` iterator is not consumed before it is dropped.
    ///
    /// It is unspecified how many elements are removed from the vector
    /// if the `Splice` value is leaked.
    ///
    /// The input iterator `replace_with` is only consumed when the `Splice` value is dropped.
    ///
    /// If the iterator yields more values than the size of the removed range
    /// a temporary vector is allocated to hold any elements after the removed range.
    ///
    /// # Panics
    ///
    /// Panics if the range has `start_bound > end_bound`, or, if the range is
    /// bounded on either end and past the length of the vector.
    ///
    pub fn splice<R, I>(&mut self, range: R, replace_with: I) -> Splice<'_, I::IntoIter, A>
    where
        R: RangeBounds<usize>,
        I: IntoIterator<Item = T>,
        A: Clone,
    {
        Splice {
            drain: self.drain(range),
            replace_with: replace_with.into_iter(),
        }
    }

    /// Clears the vector, removing all values.
    ///
    /// This method has no effect on the allocated capacity of the vector.
    ///
    pub fn clear(&mut self) {
        while self.len > 0 {
            self.pop();
        }
    }

    /// Shortens the vector, keeping the first `len` elements and dropping
    /// the rest.
    ///
    /// If `len` is greater or equal to the vector's current length, this has
    /// no effect.
    ///
    /// This method has no effect on the allocated capacity
    /// of the vector.
    ///
    pub fn truncate(&mut self, len: usize) {
        while self.len > len {
            self.pop();
        }
    }

    /// Resizes the `Vec` in-place so that `len` is equal to `new_len`.
    ///
    /// If `new_len` is greater than `len`, the `Vec` is extended by the
    /// difference, with each additional slot filled with `value`.
    /// If `new_len` is less than `len`, the `Vec` is simply truncated.
    ///
    pub fn resize(&mut self, new_len: usize, value: T)
    where
        T: Clone,
    {
        if new_len > self.len {
            while self.len < new_len {
                self.push(value.clone());
            }
        } else {
            self.truncate(new_len);
        }
    }

    /// Resizes the `Vec` in-place so that `len` is equal to `new_len`.
    ///
    /// If `new_len` is greater than `len`, the `Vec` is extended by the
    /// difference, with each additional slot filled with the result of
    /// calling the closure `f`. The return values from `f` will end up
    /// in the `Vec` in the order they have been generated.
    ///
    /// If `new_len` is less than `len`, the `Vec` is simply truncated.
    ///
    /// This method uses a closure to create new values on every push. If
    /// you'd rather [`Clone`] a given value, use [`Vec::resize`]. If you
    /// want to use the [`Default`] trait to generate values, you can
    /// pass [`Default::default`] as the second argument.
    pub fn resize_with<F>(&mut self, new_len: usize, f: F)
    where
        F: FnMut() -> T,
    {
        let mut f = f;
        if new_len > self.len {
            while self.len < new_len {
                self.push(f());
            }
        } else {
            self.truncate(new_len);
        }
    }

    /// Split the collection into two at the given index.
    pub fn split_off(&mut self, at: usize) -> Self
    where
        A: Clone,
    {
        assert!(at <= self.len);

        let other_len = self.len - at;
        let mut other = Vec::with_capacity_in(other_len, self.alloc.clone());

        unsafe {
            self.len = at;
            other.len = other_len;
            ptr::copy_nonoverlapping(self.ixp(at), other.ixp(0), other_len);
        }
        other
    }

    /// Retains only the elements specified by the predicate.
    pub fn retain<F>(&mut self, f: F)
    where
        F: FnMut(&T) -> bool,
    {
        Gap::new(self, 0).retain(f);
    }

    /// Retains only the elements specified by the predicate, passing a mutable reference to it.
    pub fn retain_mut<F>(&mut self, f: F)
    where
        F: FnMut(&mut T) -> bool,
    {
        Gap::new(self, 0).retain_mut(f);
    }

    /// Creates an iterator which removes a range.
    pub fn drain<R>(&mut self, range: R) -> Drain<'_, T, A>
    where
        R: RangeBounds<usize>,
    {
        Drain::new(self, range)
    }

    /// Creates an iterator which uses a closure to determine if an element in the range should be removed.
    pub fn extract_if<F, R>(&mut self, range: R, filter: F) -> ExtractIf<'_, T, F, A>
    where
        F: FnMut(&mut T) -> bool,
        R: RangeBounds<usize>,
    {
        ExtractIf::new(self, filter, range)
    }

    /// Removes consecutive repeated elements in the vector according to the
    /// [`PartialEq`] trait implementation.
    ///
    /// If the vector is sorted, this removes all duplicates.
    ///
    pub fn dedup(&mut self)
    where
        T: PartialEq,
    {
        self.dedup_by(|a, b| a == b);
    }

    /// Removes all but the first of consecutive elements in the vector satisfying a given equality
    /// relation.
    ///
    /// The `same_bucket` function is passed references to two elements from the vector and
    /// must determine if the elements compare equal. The elements are passed in opposite order
    /// from their order in the slice, so if `same_bucket(a, b)` returns `true`, `a` is removed.
    ///
    /// If the vector is sorted, this removes all duplicates.
    ///
    pub fn dedup_by<F>(&mut self, same_bucket: F)
    where
        F: FnMut(&mut T, &mut T) -> bool,
    {
        Gap::new(self, 0).dedup_by(same_bucket);
    }

    /// Removes all but the first of consecutive elements in the vector that resolve to the same
    /// key.
    ///
    /// If the vector is sorted, this removes all duplicates.
    pub fn dedup_by_key<F, K>(&mut self, mut key: F)
    where
        F: FnMut(&mut T) -> K,
        K: PartialEq,
    {
        self.dedup_by(|a, b| key(a) == key(b))
    }

    // ##########################################################################
    // Private methods ##########################################################
    // ##########################################################################

    /// Get pointer to ith element.
    /// # Safety
    ///
    /// ix must be <= alloc.
    #[inline]
    unsafe fn ixp(&self, i: usize) -> *mut T {
        unsafe { self.nn.as_ptr().add(i) }
    }

    /// Get ith value.
    /// # Safety
    ///
    /// i must be < cap, and the element must have been set (Written).
    #[inline]
    unsafe fn get(&mut self, i: usize) -> T {
        unsafe { ptr::read(self.ixp(i)) }
    }

    /// Set ith value.
    /// # Safety
    ///
    /// i must be < cap, and the element must be unset.
    #[inline]
    unsafe fn set(&mut self, i: usize, elem: T) {
        unsafe {
            ptr::write(self.ixp(i), elem);
        }
    }

    /// Set the allocation. This must be at least the current length.
    fn set_capacity(&mut self, na: usize) -> Result<(), AllocError> {
        assert!(na >= self.len);
        if na == self.cap {
            return Ok(());
        }
        let result = unsafe { self.basic_set_capacity(self.cap, na) };
        self.cap = na;
        result
    }

    /// Set capacity ( allocate or reallocate memory ).
    /// # Safety
    ///
    /// `oa` must be the previous alloc set (0 if no alloc has yet been set).
    unsafe fn basic_set_capacity(&mut self, oa: usize, na: usize) -> Result<(), AllocError> {
        unsafe {
            if mem::size_of::<T>() == 0 {
                return Ok(());
            }
            if na == 0 {
                self.alloc.deallocate(
                    NonNull::new(self.nn.as_ptr().cast::<u8>()).unwrap(),
                    Layout::array::<T>(oa).unwrap(),
                );
                self.nn = NonNull::dangling();
                return Ok(());
            }
            let new_layout = Layout::array::<T>(na).unwrap(); // Need to handle error here.
            let new_ptr = if oa == 0 {
                self.alloc.allocate(new_layout)
            } else {
                let old_layout = Layout::array::<T>(oa).unwrap();
                let old_ptr = self.nn.as_ptr().cast::<u8>();
                let old_ptr = NonNull::new(old_ptr).unwrap();
                if new_layout.size() > old_layout.size() {
                    self.alloc.grow(old_ptr, old_layout, new_layout)
                } else {
                    self.alloc.shrink(old_ptr, old_layout, new_layout)
                }
            }?;
            self.nn = NonNull::new(new_ptr.as_ptr().cast::<T>()).unwrap();
        }
        Ok(())
    }

    fn get_range<R>(&self, range: R) -> (usize, usize)
    where
        R: RangeBounds<usize>,
    {
        let start = match range.start_bound() {
            Bound::Included(x) => *x,
            Bound::Excluded(x) => {
                assert!(*x < usize::MAX);
                *x + 1
            }
            Bound::Unbounded => 0,
        };
        let end = match range.end_bound() {
            Bound::Included(x) => {
                assert!(*x < usize::MAX);
                *x + 1
            }
            Bound::Excluded(x) => *x,
            Bound::Unbounded => self.len,
        };
        assert!(end <= self.len);
        assert!(start <= end);
        (start, end)
    }
}

/// # Allocation methods.
/// These are used to adjust the vector capacity and allocator.
impl<T, A: Allocator> Vec<T, A> {
    /// Create a new Vec in specified allocator.
    #[must_use]
    pub const fn new_in(alloc: A) -> Vec<T, A> {
        Self {
            len: 0,
            cap: 0,
            alloc,
            nn: NonNull::dangling(),
        }
    }

    /// Returns a reference to the underlying allocator.
    pub fn allocator(&self) -> &A {
        &self.alloc
    }

    /// Returns the current capacity.
    pub const fn capacity(&self) -> usize {
        if size_of::<T>() == 0 {
            usize::MAX
        } else {
            self.cap
        }
    }

    /// Constructs a new, empty `Vec<T, A>` with at least the specified capacity
    /// with the provided allocator.
    pub fn with_capacity_in(capacity: usize, alloc: A) -> Vec<T, A> {
        let mut v = Self::new_in(alloc);
        v.set_capacity(capacity).unwrap();
        v
    }

    /// Reserves capacity for at least `additional` more elements to be inserted
    /// in the given `Vec<T>`.
    pub fn reserve(&mut self, additional: usize) {
        let capacity = self.len + additional;
        // Could round up to power of 2 here.
        if capacity > self.cap {
            self.set_capacity(capacity).unwrap();
        }
    }

    /// Reserves minimum capacity for at least `additional` more elements to be inserted
    /// in the given `Vec<T>`.
    pub fn reserve_exact(&mut self, additional: usize) {
        let capacity = self.len + additional;
        if capacity > self.cap {
            self.set_capacity(capacity).unwrap();
        }
    }

    /// Trim excess storage allocation.
    pub fn shrink_to_fit(&mut self) {
        let _ = self.set_capacity(self.len);
    }

    /// Trim excess capacity to specified value.
    pub fn shrink_to(&mut self, capacity: usize) {
        if self.cap > capacity {
            let _ = self.set_capacity(cmp::max(self.len, capacity));
        }
    }
}

impl<T> Vec<T> {
    /// Constructs a new, empty `Vec<T>` with at least the specified capacity..
    ///
    /// # Example
    ///
    /// ```
    /// use pstd::vec::Vec;
    /// let mut v = Vec::with_capacity(10);
    /// v.push("England");
    /// v.push("France");
    /// assert!( v.len() == 2 );
    /// for s in &v { println!("s={}",s); }
    /// ```
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Vec<T> {
        let mut v = Vec::<T>::new();
        v.set_capacity(capacity).unwrap();
        v
    }
}

/// # Conversion methods.
///
/// These convert a [`Vec`] to and from various types.
impl<T, A: Allocator> Vec<T, A> {
    /// Extracts a slice containing the entire vector.
    ///
    /// Equivalent to `&s[..]`.
    ///
    pub const fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.nn.as_ptr(), self.len) }
    }

    /// Extracts a mut slice containing the entire vector.
    ///
    /// Equivalent to `&mut s[..]`.
    ///
    pub const fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.nn.as_ptr(), self.len) }
    }

    /// Returns the `NonNull` pointer to the vector's buffer.
    pub const fn as_non_null(&mut self) -> NonNull<T> {
        self.nn
    }

    /// Returns a raw pointer to the vector's buffer, or a dangling raw pointer
    /// valid for zero sized reads if the vector didn't allocate.
    pub const fn as_ptr(&self) -> *const T {
        self.nn.as_ptr()
    }

    /// Returns a raw mutable pointer to the vector's buffer, or a dangling
    /// raw pointer valid for zero sized reads if the vector didn't allocate.
    pub const fn as_mut_ptr(&mut self) -> *mut T {
        self.nn.as_ptr()
    }

    /// Decomposes a `Vec<T>` into its components: `(NonNull pointer, length, capacity)`.
    pub fn into_parts(self) -> (NonNull<T>, usize, usize) {
        let (ptr, len, capacity) = self.into_raw_parts();
        // SAFETY: A `Vec` always has a non-null pointer.
        (unsafe { NonNull::new_unchecked(ptr) }, len, capacity)
    }

    /// Decomposes a `Vec<T>` into its components: `(NonNull pointer, length, capacity, allocator)`.
    pub fn into_parts_with_alloc(self) -> (NonNull<T>, usize, usize, A) {
        let (ptr, len, capacity, alloc) = self.into_raw_parts_with_alloc();
        // SAFETY: A `Vec` always has a non-null pointer.
        (unsafe { NonNull::new_unchecked(ptr) }, len, capacity, alloc)
    }

    /// Decomposes a `Vec<T>` into its raw components: `(pointer, length, capacity)`.
    pub fn into_raw_parts(self) -> (*mut T, usize, usize) {
        /* let len = self.len;
        let cap = self.cap;
        let ptr = unsafe{ self.nn.as_mut() };
        mem::forget(self);
        (ptr, len, cap)
        */
        let mut me = ManuallyDrop::new(self);
        (me.as_mut_ptr(), me.len, me.cap)
    }

    /// Decomposes a `Vec<T>` into its raw components: `(pointer, length, capacity, allocator)`.
    pub fn into_raw_parts_with_alloc(self) -> (*mut T, usize, usize, A) {
        let mut me = ManuallyDrop::new(self);
        let alloc = unsafe { ptr::read(&me.alloc) };
        (me.as_mut_ptr(), me.len, me.cap, alloc)
    }

    /// Creates a `Vec<T, A>` directly from a `NonNull` pointer, a length, a capacity,
    /// and an allocator.
    ///
    /// # Safety
    ///
    /// Parameters must all be correct, ptr must have been allocated from alloc.
    pub unsafe fn from_parts_in(ptr: NonNull<T>, length: usize, capacity: usize, alloc: A) -> Self {
        assert!(capacity >= length);
        Self {
            nn: ptr,
            cap: capacity,
            alloc,
            len: length,
        }
    }

    /// Creates a `Vec<T, A>` directly from a pointer, a length, a capacity,
    /// and an allocator.
    ///
    /// # Safety
    ///
    /// Parameters must all be correct, ptr must have been allocated from alloc.
    pub unsafe fn from_raw_parts_in(ptr: *mut T, length: usize, capacity: usize, alloc: A) -> Self {
        assert!(capacity >= length);
        let nn = unsafe { NonNull::new_unchecked(ptr) };
        Self {
            nn,
            cap: capacity,
            alloc,
            len: length,
        }
    }

    /* Would need to implement Box first to make this possible under Stable.
        /// Converts the vector into [`Box<[T]>`][owned slice].
        ///
        /// Before doing the conversion, this method discards excess capacity like [`shrink_to_fit`].
        ///
        /// [owned slice]: Box
        /// [`shrink_to_fit`]: Vec::shrink_to_fit
        pub fn into_boxed_slice(mut self) -> Box<[T], A> {
            unsafe {
                self.shrink_to_fit();
                let me = ManuallyDrop::new(self);
                let alloc = unsafe { ptr::read(&me.alloc) };
                let slice = me.as_mut_slice();
                Box::from_raw_in(slice, alloc)
            }
        }
    */

    /// Consumes and leaks the `Vec`, returning a mutable reference to the contents.
    pub fn leak<'a>(self) -> &'a mut [T]
    where
        A: 'a,
    {
        let mut me = ManuallyDrop::new(self);
        unsafe { slice::from_raw_parts_mut(me.as_mut_ptr(), me.len) }
    }

    /* Not clear how to implement this...
    /// Interns the `Vec<T>`, making the underlying memory read-only. This method should be
    /// called during compile time. (This is a no-op if called during runtime)
    ///
    /// This method must be called if the memory used by `Vec` needs to appear in the final
    /// values of constants.
    pub const fn const_make_global(mut self) -> &'static [T]
    where
        T: Freeze,
    {
        unsafe { core::intrinsics::const_make_global(self.as_mut_ptr().cast()) };
        let me = ManuallyDrop::new(self);
        unsafe { slice::from_raw_parts(me.as_ptr(), me.len) }
    }
    */
}

impl<T> Vec<T> {
    /// Creates a `Vec<T>` where each element is produced by calling `f` with
    /// that element's index while walking forward through the `Vec<T>`.
    ///
    /// # Example
    ///
    /// ```
    /// use pstd::vec::Vec;
    /// let v = Vec::from_fn(10, |i| i * 2);
    /// ```
    pub fn from_fn<F>(length: usize, f: F) -> Vec<T>
    where
        F: FnMut(usize) -> T,
    {
        let mut f = f;
        let mut m = Vec::with_capacity(length);
        for i in 0..length {
            m.push(f(i));
        }
        m
    }

    /// Creates a `Vec<T>` directly from a `NonNull` pointer, a length, and a capacity.
    ///
    /// # Safety
    ///
    /// Parameters must all be correct, ptr must have been allocated from Global allocator.
    ///
    /// # Example
    ///
    /// ```
    /// use pstd::vec::Vec;
    /// let mut v = Vec::new();
    /// v.push("Hello");
    ///
    /// // Deconstruct the vector into parts.
    /// let (p, len, cap) = v.into_parts();
    ///
    /// unsafe {
    ///     // Put everything back together into a Vec
    ///     let rebuilt = Vec::from_parts(p, len, cap);
    /// }
    /// ```
    pub unsafe fn from_parts(ptr: NonNull<T>, length: usize, capacity: usize) -> Vec<T> {
        let mut v = Vec::new();
        v.len = length;
        v.cap = capacity;
        v.nn = ptr;
        v
    }

    /// Creates a `Vec<T>` directly from a pointer, a length, and a capacity.
    ///
    /// # Safety
    ///
    /// Parameters must all be correct, ptr must have been allocated from Global allocator.
    pub unsafe fn from_raw_parts(ptr: *mut T, length: usize, capacity: usize) -> Vec<T> {
        let mut v = Vec::new();
        v.len = length;
        v.cap = capacity;
        v.nn = unsafe { NonNull::new_unchecked(ptr) };
        v
    }

    /// Returns the remaining spare capacity of the vector as a slice of
    /// `MaybeUninit<T>`.
    ///
    /// The returned slice can be used to fill the vector with data (e.g. by
    /// reading from a file) before marking the data as initialized using the
    /// [`set_len`] method.
    ///
    /// [`set_len`]: Vec::set_len
    ///
    /// # Examples
    ///
    /// ```
    /// // Allocate vector big enough for 10 elements.
    /// let mut v = Vec::with_capacity(10);
    ///
    /// // Fill in the first 3 elements.
    /// let uninit = v.spare_capacity_mut();
    /// uninit[0].write(0);
    /// uninit[1].write(1);
    /// uninit[2].write(2);
    ///
    /// // Mark the first 3 elements of the vector as being initialized.
    /// unsafe {
    ///     v.set_len(3);
    /// }
    ///
    /// assert_eq!(&v, &[0, 1, 2]);
    ///
    pub fn spare_capacity_mut(&mut self) -> &mut [MaybeUninit<T>] {
        unsafe {
            slice::from_raw_parts_mut(
                self.as_mut_ptr().add(self.len) as *mut MaybeUninit<T>,
                self.cap - self.len,
            )
        }
    }

    /// Forces the length of the vector to `new_len`.
    ///
    /// This is a low-level operation that maintains none of the normal
    /// invariants of the type. Normally changing the length of a vector
    /// is done using one of the a safe operation.
    ///
    /// # Safety
    ///
    /// - `new_len` must be less than or equal to [`capacity()`].
    /// - The elements at `old_len..new_len` must be initialized.
    ///
    /// [`capacity()`]: Vec::capacity
    ///
    pub unsafe fn set_len(&mut self, new_len: usize) {
        assert!(new_len <= self.cap);
        self.len = new_len;
    }
}

/// # Non-panic methods.
/// These are panic-free alternatives for programs that must not panic.
impl<T, A: Allocator> Vec<T, A> {
    /// Constructs a new, empty Vec<T, A> with at least the specified capacity with the provided allocator.
    pub fn try_with_capacity_in(capacity: usize, alloc: A) -> Result<Self, AllocError> {
        let mut v = Self::new_in(alloc);
        v.set_capacity(capacity)?;
        Ok(v)
    }

    /// Tries to reserve capacity for at least additional more elements to be inserted.
    pub fn try_reserve(&mut self, additional: usize) -> Result<(), AllocError> {
        let capacity = self.len + additional;
        // Could round up to power of 2 here.
        if capacity > self.cap {
            return self.set_capacity(capacity);
        }
        Ok(())
    }

    /// Tries to reserve capacity for at least additional more elements to be inserted.
    pub fn try_reserve_exact(&mut self, additional: usize) -> Result<(), AllocError> {
        let capacity = self.len + additional;
        if capacity > self.cap {
            return self.set_capacity(capacity);
        }
        Ok(())
    }

    /// Appends an element and returns a reference to it if there is sufficient spare capacity,
    /// otherwise an error is returned with the element.
    pub fn push_within_capacity(&mut self, value: T) -> Result<&mut T, T> {
        if self.cap == self.len {
            return Err(value);
        }
        let index = self.len;
        self.len += 1;
        unsafe {
            self.set(index, value);
        }
        Ok(unsafe { &mut *self.ixp(index) })
    }

    /// Remove the value at index, elements are moved down to fill the space.
    /// Returns None if index >= len().
    pub fn try_remove(&mut self, index: usize) -> Option<T> {
        if index >= self.len {
            return None;
        }
        unsafe {
            let result = self.get(index);
            ptr::copy(self.ixp(index + 1), self.ixp(index), self.len() - index - 1);
            self.len -= 1;
            Some(result)
        }
    }
}

impl<T> Vec<T> {
    /// Constructs a new, empty `Vec<T>` with at least the specified capacity.
    pub fn try_with_capacity(capacity: usize) -> Result<Vec<T>, AllocError> {
        let mut v = Vec::<T>::new();
        v.set_capacity(capacity)?;
        Ok(v)
    }
}

// ##########################################################################
// Impls ####################################################################
// ##########################################################################

unsafe impl<T: Send> Send for Vec<T> {}
unsafe impl<T: Sync> Sync for Vec<T> {}

impl<T: Eq, A: Allocator> Eq for Vec<T, A> {}

impl<T: PartialEq, A: Allocator> PartialEq for Vec<T, A> {
    fn eq(&self, other: &Vec<T, A>) -> bool {
        self[..] == other[..]
    }
}

impl<T, U, A: Allocator, const N: usize> PartialEq<[U; N]> for Vec<T, A>
where
    T: PartialEq<U>,
{
    fn eq(&self, other: &[U; N]) -> bool {
        self[..] == other[..]
    }
}

impl<T, U, A: Allocator, const N: usize> PartialEq<&[U; N]> for Vec<T, A>
where
    T: PartialEq<U>,
{
    fn eq(&self, other: &&[U; N]) -> bool {
        self[..] == other[..]
    }
}

impl<T, A: Allocator> Deref for Vec<T, A> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.nn.as_ptr(), self.len) }
    }
}

impl<T, A: Allocator> DerefMut for Vec<T, A> {
    fn deref_mut(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.nn.as_ptr(), self.len) }
    }
}

impl<'a, T: 'a> IntoIterator for &'a Vec<T> {
    type Item = &'a T;
    type IntoIter = slice::Iter<'a, T>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, T: 'a> IntoIterator for &'a mut Vec<T> {
    type Item = &'a mut T;
    type IntoIter = slice::IterMut<'a, T>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

impl<T, A: Allocator> IntoIterator for Vec<T, A> {
    type Item = T;
    type IntoIter = IntoIter<T, A>;
    fn into_iter(self) -> Self::IntoIter {
        IntoIter::new(self)
    }
}

impl<T, A: Allocator> Extend<T> for Vec<T, A> {
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        for e in iter {
            self.push(e);
        }
    }
}

impl<'a, T: Copy + 'a, A: Allocator> Extend<&'a T> for Vec<T, A> {
    fn extend<I: IntoIterator<Item = &'a T>>(&mut self, iter: I) {
        for e in iter {
            self.push(*e);
        }
    }
}

impl<T: Clone, A: Allocator + Clone> Clone for Vec<T, A> {
    fn clone(&self) -> Self {
        let mut v = Vec::with_capacity_in(self.len, self.alloc.clone());
        for e in self.iter() {
            v.push(e.clone());
        }
        v
    }
}

impl<T, A: Allocator> Drop for Vec<T, A> {
    fn drop(&mut self) {
        while self.len != 0 {
            self.pop();
        }
        let _ = self.set_capacity(0);
    }
}

impl<T> Default for Vec<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Debug, A: Allocator> Debug for Vec<T, A> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // panic!();
        // f.debug_list().entries((*self).iter().finish()
        fmt::Debug::fmt(&**self, f)
    }
}

// From implementations
impl<T: Clone> From<&[T]> for Vec<T> {
    /// Allocates a `Vec<T>` and fills it by cloning `s`'s items.
    fn from(s: &[T]) -> Vec<T> {
        let mut v = Vec::new();
        for e in s {
            v.push(e.clone());
        }
        v
    }
}

impl<T, const N: usize> From<[T; N]> for Vec<T> {
    /// Allocates a `Vec<T>` and moves `a`'s items into it.
    ///
    /// # Examples
    ///
    /// ```
    /// assert_eq!(Vec::from([1, 2, 3]), vec![1, 2, 3]);
    /// ```
    fn from(a: [T; N]) -> Vec<T> {
        Vec::from_iter(a)
    }
}

impl<T> FromIterator<T> for Vec<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Vec<T> {
        let mut v = Vec::new();
        for e in iter {
            v.push(e);
        }
        v
    }
}

// ##########################################################################
// Iterators ################################################################
// ##########################################################################

/// Consuming iterator for [`Vec`].
#[derive(Debug)]
pub struct IntoIter<T, A: Allocator = Global> {
    start: usize,
    end: usize,
    v: Vec<T, A>,
}

impl<T, A: Allocator> IntoIter<T, A> {
    fn new(mut v: Vec<T, A>) -> Self {
        let end = v.len;
        v.len = 0;
        Self { start: 0, end, v }
    }

    /// Returns the remaining items of this iterator as a slice.
    pub fn as_slice(&self) -> &[T] {
        unsafe { slice::from_raw_parts(self.v.ixp(self.start), self.len()) }
    }

    /// Returns the remaining items of this iterator as a mutable slice.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { slice::from_raw_parts_mut(self.v.ixp(self.start), self.len()) }
    }
}

impl<T, A: Allocator> Iterator for IntoIter<T, A> {
    type Item = T;
    fn next(&mut self) -> Option<T> {
        if self.start == self.end {
            None
        } else {
            let ix = self.start;
            self.start += 1;
            Some(unsafe { self.v.get(ix) })
        }
    }
}

impl<T, A: Allocator> ExactSizeIterator for IntoIter<T, A> {
    fn len(&self) -> usize {
        self.end - self.start
    }
}

impl<T, A: Allocator> DoubleEndedIterator for IntoIter<T, A> {
    fn next_back(&mut self) -> Option<T> {
        if self.start == self.end {
            None
        } else {
            self.end -= 1;
            Some(unsafe { self.v.get(self.end) })
        }
    }
}

impl<T, A: Allocator> FusedIterator for IntoIter<T, A> {}

impl<T, A: Allocator> Drop for IntoIter<T, A> {
    fn drop(&mut self) {
        while self.next().is_some() {}
    }
}

/// For removing multiple elements from a Vec.
/// When dropped, it closes up any gap. The gap size is r-w.
struct Gap<'a, T, A: Allocator> {
    r: usize,   // Read index
    w: usize,   // Write index
    len: usize, // Original len of v
    v: &'a mut Vec<T, A>,
}

impl<'a, T, A: Allocator> Gap<'a, T, A> {
    fn new(v: &'a mut Vec<T, A>, b: usize) -> Self {
        let len = v.len;
        v.len = 0; // Correct length will be calculated when Gap drops.
        Self { w: b, r: b, v, len }
    }

    fn close_gap(&mut self) // Called from drop
    {
        let n = self.len - self.r;
        let g = self.r - self.w;
        if n > 0 && g != 0 {
            unsafe {
                let dst = self.v.ixp(self.w);
                let src = self.v.ixp(self.r);
                ptr::copy(src, dst, n);
            }
        }
        self.v.len = self.len - g;
    }

    fn keep(&mut self, upto: usize) {
        unsafe {
            while self.r < upto {
                // Could use ptr::copy
                let nxt = self.v.ixp(self.r);
                // Retain element
                if self.r != self.w {
                    let to = self.v.ixp(self.w);
                    ptr::write(to, ptr::read(nxt));
                }
                self.r += 1;
                self.w += 1;
            }
        }
    }

    /// For splice ( size must be > 0 )
    unsafe fn fill(&mut self, e: T) {
        unsafe {
            let to = self.v.ixp(self.w);
            ptr::write(to, e);
            self.w += 1;
        }
    }

    // For splice
    fn append<I: Iterator<Item = T>>(&mut self, src: &mut I)
    where
        A: Clone,
    {
        // Fill the gap
        while self.w < self.r {
            if let Some(e) = src.next() {
                unsafe {
                    self.fill(e);
                }
            } else {
                return;
            }
        }
        self.v.len = self.len; // Restore len ( no gap ).
        let mut tail = self.v.split_off(self.w);
        for e in src {
            self.v.push(e);
        }
        self.v.append(&mut tail);
        self.len = self.v.len; // So drop doesn't truncate back to original len
    }

    fn retain<F>(&mut self, mut f: F)
    where
        F: FnMut(&T) -> bool,
    {
        unsafe {
            while self.r < self.len {
                let nxt = self.v.ixp(self.r);
                if f(&mut *nxt) {
                    // Retain element
                    if self.r != self.w {
                        let to = self.v.ixp(self.w);
                        ptr::write(to, ptr::read(nxt));
                    }
                    self.r += 1;
                    self.w += 1;
                } else {
                    // Discard
                    self.r += 1;
                    let _v = ptr::read(nxt); // Could panic on drop.
                }
            }
        }
    }

    fn retain_mut<F>(&mut self, mut f: F)
    where
        F: FnMut(&mut T) -> bool,
    {
        unsafe {
            while self.r < self.len {
                let nxt = self.v.ixp(self.r);
                if f(&mut *nxt) {
                    // Retain element
                    if self.r != self.w {
                        let to = self.v.ixp(self.w);
                        ptr::write(to, ptr::read(nxt));
                    }
                    self.r += 1;
                    self.w += 1;
                } else {
                    // Discard
                    self.r += 1;
                    let _v = ptr::read(nxt); // Could panic on drop.
                }
            }
        }
    }

    fn dedup_by<F>(&mut self, mut same_bucket: F)
    where
        F: FnMut(&mut T, &mut T) -> bool,
    {
        if self.len > 0 {
            unsafe {
                let mut cur = 0;
                self.r = 1;
                self.w = 1;
                while self.r < self.len {
                    let cp = self.v.ixp(cur);
                    let nxt = self.v.ixp(self.r);
                    if same_bucket(&mut *nxt, &mut *cp) {
                        // Discard duplicate
                        self.r += 1;
                        let _v = ptr::read(nxt); // Could panic on drop.
                    } else {
                        cur = self.w;
                        if self.r != self.w {
                            let to = self.v.ixp(self.w);
                            ptr::write(to, ptr::read(nxt));
                        }
                        self.r += 1;
                        self.w += 1;
                    }
                }
            }
        }
    }

    fn extract_if<F>(&mut self, f: &mut F, end: usize) -> Option<T>
    where
        F: FnMut(&mut T) -> bool,
    {
        unsafe {
            while self.r < end {
                let nxt = self.v.ixp(self.r);
                if f(&mut *nxt) {
                    self.r += 1;
                    return Some(ptr::read(nxt));
                }
                if self.r != self.w {
                    let to = self.v.ixp(self.w);
                    ptr::write(to, ptr::read(nxt));
                }
                self.r += 1;
                self.w += 1;
            }
            None
        }
    }
}

impl<'a, T, A: Allocator> Drop for Gap<'a, T, A> {
    // Close up any gap.
    fn drop(&mut self) {
        self.close_gap();
    }
}

/// An iterator to remove a range.
///
/// This struct is created by [`Vec::drain`].
/// See its documentation for more.
pub struct Drain<'a, T, A: Allocator = Global> {
    gap: Gap<'a, T, A>,
    end: usize,
    br: usize,
}

impl<'a, T, A: Allocator> Drain<'a, T, A> {
    fn new<R: RangeBounds<usize>>(vec: &'a mut Vec<T, A>, range: R) -> Self {
        let (b, end) = vec.get_range(range);
        let gap = Gap::new(vec, b);
        Self { gap, end, br: end }
    }

    /// Keep unyielded elements in the source `Vec`.
    pub fn keep_rest(mut self) {
        self.gap.keep(self.br);
    }
}

impl<T, A: Allocator> Iterator for Drain<'_, T, A> {
    type Item = T;
    fn next(&mut self) -> Option<T> {
        if self.gap.r == self.br {
            return None;
        }
        let x = self.gap.r;
        self.gap.r += 1;
        Some(unsafe { self.gap.v.get(x) })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.br - self.gap.r;
        (n, Some(n))
    }
}

impl<T, A: Allocator> DoubleEndedIterator for Drain<'_, T, A> {
    fn next_back(&mut self) -> Option<T> {
        if self.gap.r == self.br {
            return None;
        }
        self.br -= 1;
        let x = self.br;
        Some(unsafe { self.gap.v.get(x) })
    }
}

impl<T, A: Allocator> Drop for Drain<'_, T, A> {
    fn drop(&mut self) {
        while self.next().is_some() {}
        self.gap.r = self.end;
    }
}

impl<T, A: Allocator> ExactSizeIterator for Drain<'_, T, A> {}

impl<T, A: Allocator> FusedIterator for Drain<'_, T, A> {}

/// An iterator which uses a closure to determine if an element should be removed.
///
/// This struct is created by [`Vec::extract_if`].
/// See its documentation for more.
pub struct ExtractIf<'a, T, F, A: Allocator = Global> {
    gap: Gap<'a, T, A>,
    end: usize,
    pred: F,
}

impl<'a, T, F, A: Allocator> ExtractIf<'a, T, F, A> {
    pub(super) fn new<R: RangeBounds<usize>>(vec: &'a mut Vec<T, A>, pred: F, range: R) -> Self {
        let (b, end) = vec.get_range(range);
        let gap = Gap::new(vec, b);
        Self { gap, pred, end }
    }
}

impl<T, F, A: Allocator> Iterator for ExtractIf<'_, T, F, A>
where
    F: FnMut(&mut T) -> bool,
{
    type Item = T;
    fn next(&mut self) -> Option<T> {
        self.gap.extract_if(&mut self.pred, self.end)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(self.end - self.gap.r))
    }
}

/// A splicing iterator for `Vec`.
///
/// This struct is created by [`Vec::splice()`].
/// See its documentation for more.
///
pub struct Splice<'a, I: Iterator + 'a, A: Allocator + Clone + 'a = Global> {
    drain: Drain<'a, I::Item, A>,
    replace_with: I,
}

impl<I: Iterator, A: Allocator + Clone> Drop for Splice<'_, I, A> {
    fn drop(&mut self) {
        self.drain.by_ref().for_each(drop);
        self.drain.gap.append(&mut self.replace_with);
    }
}

impl<I: Iterator, A: Allocator + Clone> Iterator for Splice<'_, I, A> {
    type Item = I::Item;

    fn next(&mut self) -> Option<Self::Item> {
        self.drain.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.drain.size_hint()
    }
}

impl<I: Iterator, A: Allocator + Clone> DoubleEndedIterator for Splice<'_, I, A> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.drain.next_back()
    }
}

/// Creates a [`Vec`] containing the arguments.
#[macro_export]
macro_rules! vec {
    () => (
        $crate::vec::Vec::new()
    );
    ($elem:expr; $n:expr) => (
        $crate::vec::from_elem($elem, $n)
    );
    ($($x:expr),+ $(,)?) => (
        Vec::from_iter([$($x),+].into_iter())
    );
}

#[doc(hidden)]
pub fn from_elem<T: Clone>(elem: T, n: usize) -> Vec<T> {
    let mut v = Vec::with_capacity(n);
    for _i in 0..n {
        v.push(elem.clone());
    }
    v
}

#[test]
fn test() {
    let mut v = Vec::new();
    v.push(99);
    v.push(314);
    println!("v={:?}", &v);
    assert!(v[0] == 99);
    assert!(v[1] == 314);
    assert!(v.len() == 2);
    v[1] = 316;
    assert!(v[1] == 316);
    for x in &v {
        println!("x={}", x);
    }
    for x in &mut v {
        *x += 1;
        println!("x={}", x);
    }
    for x in v {
        println!("x={}", x);
    }
    //assert!(v.pop() == Some(316));
    //assert!(v.pop() == Some(99));
    //assert!(v.pop() == None);

    let mut v = vec![199, 200, 200, 201, 201];
    println!("v={:?}", &v);
    v.dedup();
    println!("v={:?}", &v);

    let mut numbers = vec![1, 2, 3, 4, 5, 6, 8, 9, 11, 13, 14, 15];
    let extr: Vec<_> = numbers.extract_if(3..9, |x| *x % 2 == 0).collect();

    println!("numbers={:?} extr={:?}", &numbers, &extr);

    let mut a = vec![1, 2, 0, 5]; // Vec::from(&[1, 2, 0, 5][..]);
    let b = vec![3, 4];
    a.splice(2..3, b);
    println!("a={:?}", &a);
    assert_eq!(a, [1, 2, 3, 4, 5]);

    let v = vec![99; 5];
    println!("v={:?}", &v);

    let v: Vec<_> = Vec::from_iter([1, 2, 3, 4].into_iter());
    println!("v={:?}", &v);
}

#[cfg(test)]
mod test;
