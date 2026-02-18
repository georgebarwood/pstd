use crate::alloc::{Allocator, Global};

use std::{
    alloc::Layout,
    //borrow::Borrow,
    cmp,
    //cmp::Ordering,
    fmt,
    fmt::Debug,
    iter::FusedIterator,
    //marker::PhantomData,
    mem,
    // ops::{Bound, Deref, DerefMut, RangeBounds},
    ops::{Deref, DerefMut},
    ptr,
    ptr::NonNull,
    slice,
};

/// A vector that grows as elements are pushed onto it similar to similar to [`std::vec::Vec`].
pub struct Vec<T, A: Allocator = Global> {
    len: usize,
    resvd: usize,
    p: NonNull<T>,
    alloc: A,
}

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
        Self::new_in(Global)
    }
}

/// # Basic methods.
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
        if self.resvd == self.len {
            let nc = if self.resvd == 0 { 4 } else { self.resvd * 2 };
            self.set_capacity(nc);
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

    /// Insert value at index, after moving elements up to make a space.
    /// # Panics
    ///
    /// Panics if `index` > len().
    ///
    pub fn insert(&mut self, index: usize, value: T) {
        assert!(index <= self.len);
        if self.resvd == self.len {
            let na = if self.resvd == 0 { 4 } else { self.resvd * 2 };
            self.set_capacity(na);
        }
        unsafe {
            if index < self.len {
                ptr::copy(self.ixp(index), self.ixp(index + 1), self.len - index);
            }
            self.set(index, value);
        }
        self.len += 1;
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
            if index != self.len
            {
                let last = self.get(self.len);
                self.set(index, last);
            }
            result
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

    /// Removes consecutive repeated elements in the vector according to the
    /// [`PartialEq`] trait implementation.
    ///
    /// If the vector is sorted, this removes all duplicates.
    ///
    pub fn dedup(&mut self)
    where
        T: Ord,
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
        Gap::new(self).dedup_by( same_bucket );
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
        unsafe { self.p.as_ptr().add(i) }
    }
    
    /// Get ith value.
    /// # Safety
    ///
    /// i must be < alloc, and the element must have been set (Written).
    #[inline]
    unsafe fn get(&mut self, i: usize) -> T {
        unsafe { ptr::read(self.ixp(i)) }
    }

    /// Set ith value.
    /// # Safety
    ///
    /// i must be < alloc, and the element must be unset.
    #[inline]
    unsafe fn set(&mut self, i: usize, elem: T) {
        unsafe {
            ptr::write(self.ixp(i), elem);
        }
    }

    /// Set the allocation. This must be at least the current length.
    fn set_capacity(&mut self, na: usize) {
        assert!(na >= self.len);
        if na == self.resvd {
            return;
        }
        unsafe {
            self.basic_set_capacity(self.resvd, na);
        }
        self.resvd = na;
    }

    /// Set capacity ( allocate or reallocate memory ).
    /// # Safety
    ///
    /// `oa` must be the previous alloc set (0 if no alloc has yet been set).
    unsafe fn basic_set_capacity(&mut self, oa: usize, na: usize) {
        unsafe {
            if mem::size_of::<T>() == 0 {
                return;
            }
            if na == 0 {
                self.alloc.deallocate(
                    NonNull::new(self.p.as_ptr().cast::<u8>()).unwrap(),
                    Layout::array::<T>(oa).unwrap(),
                );
                self.p = NonNull::dangling();
                return;
            }
            let new_layout = Layout::array::<T>(na).unwrap();
            let new_ptr = if oa == 0 {
                self.alloc.allocate(new_layout)
            } else {
                let old_layout = Layout::array::<T>(oa).unwrap();
                let old_ptr = self.p.as_ptr().cast::<u8>();
                let old_ptr = NonNull::new(old_ptr).unwrap();
                if new_layout.size() > old_layout.size() {
                    self.alloc.grow(old_ptr, old_layout, new_layout)
                } else {
                    self.alloc.shrink(old_ptr, old_layout, new_layout)
                }
            }
            .unwrap();
            self.p = NonNull::new(new_ptr.as_ptr().cast::<T>()).unwrap();
        }
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
            resvd: 0,
            alloc,
            p: NonNull::dangling(),
        }
    }

    /// Returns a reference to the underlying allocator.
    pub fn allocator(&self) -> &A {
        &self.alloc
    }

    /// Constructs a new, empty `Vec<T>` with at least the specified capacity.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Vec<T> {
        let mut v = Vec::<T>::new();
        v.set_capacity(capacity);
        v
    }

    /// Returns the current capacity.
    pub const fn capacity(&self) -> usize {
        self.resvd
    }

    /// Constructs a new, empty `Vec<T, A>` with at least the specified capacity
    /// with the provided allocator.
    pub fn with_capacity_in(capacity: usize, alloc: A) -> Self {
        let mut v = Self::new_in(alloc);
        v.set_capacity(capacity);
        v
    }

    /// Reserves capacity for at least `additional` more elements to be inserted
    /// in the given `Vec<T>`.
    pub fn reserve(&mut self, additional: usize) {
        let capacity = self.len + additional;
        // Could round up to power of 2 here.
        if capacity > self.resvd {
            self.set_capacity(capacity);
        }
    }

    /// Reserves minimum capacity for at least `additional` more elements to be inserted
    /// in the given `Vec<T>`.
    pub fn reserve_exact(&mut self, additional: usize) {
        let capacity = self.len + additional;
        if capacity > self.resvd {
            self.set_capacity(capacity);
        }
    }

    /// Trim excess storage allocation.
    pub fn shrink_to_fit(&mut self) {
        self.set_capacity(self.len);
    }

    /// Trim excess capacity to specified value.
    pub fn shrink_to(&mut self, capacity: usize) {
        if self.resvd > capacity {
            self.set_capacity(cmp::max(self.len, capacity));
        }
    }
}

unsafe impl<T: Send> Send for Vec<T> {}
unsafe impl<T: Sync> Sync for Vec<T> {}

impl<T, A: Allocator> Deref for Vec<T, A> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.p.as_ptr(), self.len) }
    }
}

impl<T, A: Allocator> DerefMut for Vec<T, A> {
    fn deref_mut(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.p.as_ptr(), self.len) }
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
        IntoIter { start: 0, v: self }
    }
}

impl<T, A: Allocator> Drop for Vec<T, A> {
    fn drop(&mut self) {
        while self.len != 0 {
            self.pop();
        }
        self.set_capacity(0);
    }
}

impl<T> Default for Vec<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T, A: Allocator> fmt::Debug for Vec<T, A>
where
    T: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // panic!();
        // f.debug_list().entries((*self).iter().finish()
        fmt::Debug::fmt(&**self, f)
    }
}

/// Consuming iterator for [`Vec`].
#[derive(Debug)]
pub struct IntoIter<T, A: Allocator = Global> {
    start: usize,
    v: Vec<T, A>,
}

impl<T, A: Allocator> Iterator for IntoIter<T, A> {
    type Item = T;
    fn next(&mut self) -> Option<T> {
        if self.start == self.v.len() {
            self.start = 0;
            self.v.len = 0;
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
        self.v.len() - self.start
    }
}

impl<T, A: Allocator> DoubleEndedIterator for IntoIter<T, A> {
    fn next_back(&mut self) -> Option<T> {
        if self.start == self.v.len() {
            self.start = 0;
            self.v.len = 0;
            None
        } else {
            self.v.len -= 1;
            Some(unsafe { self.v.get(self.v.len()) })
        }
    }
}

impl<T, A: Allocator> FusedIterator for IntoIter<T, A> {}

impl<T, A: Allocator> Drop for IntoIter<T, A> {
    fn drop(&mut self) {
        while self.next().is_some() {}
    }
}


impl<T: Clone> From<&[T]> for Vec<T> {
    /// Allocates a `Vec<T>` and fills it by cloning `s`'s items.
    fn from(s: &[T]) -> Vec<T> {
        let mut v = Vec::new();
        for e in s { v.push(e.clone()); }
        v
    }
}

/// For removing multiple elements from a Vec.
/// When dropped, it closes up any gap.
struct Gap<'a, T, A: Allocator> {
    r: usize,  // Read index
    w: usize,  // Write index
    v: &'a mut Vec<T, A>,
}

impl<'a, T, A: Allocator> Gap<'a, T, A> {
    fn new(v: &'a mut Vec<T, A>) -> Self {
        Self { w: 0, r: 0, v }
    }

    fn dedup_by<F>(&mut self, mut same_bucket: F)
    where
        F: FnMut(&mut T, &mut T) -> bool,
    {
        if self.v.len > 0 {
            unsafe {
                let mut cur = self.v.ixp(0);
                self.r = 1;
                self.w = 1;
                while self.r < self.v.len {
                    let nxt = self.v.ixp(self.r);
                    if same_bucket(&mut *nxt, &mut *cur) {
                        // Discard duplicate
                        self.r += 1;
                        let _v = ptr::read(nxt); // Could panic on drop.
                    } else {
                        cur = nxt;
                        if self.r != self.w {
                            let to = self.v.ixp(self.w);
                            *to = ptr::read(nxt);
                        }
                        self.r += 1;
                        self.w += 1;
                    }
                }
            }
        }
    }
}

impl<'a, T, A: Allocator> Drop for Gap<'a, T, A> {
    // Close up any gap.
    fn drop(&mut self) {
        let n = self.v.len - self.r;
        if n > 0 && self.w != self.r {
            unsafe {
                let dst = self.v.ixp(self.w);
                let src = self.v.ixp(self.r);
                ptr::copy(src, dst, n);
                self.v.len = self.w + n;
            }
        } else {
            self.v.len = self.w;
        }
    }
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

    let mut v = Vec::from( &[199,200,200,201,201][..] );
    println!("v={:?}", &v);
    v.dedup();
    println!("v={:?}", &v);
}
