use std::{
    alloc::Layout,
    borrow::Borrow,
    cmp::Ordering,
    fmt,
    fmt::Debug,
    marker::PhantomData,
    mem,
    ops::{Bound, Deref, DerefMut, RangeBounds},
    ptr,
    ptr::NonNull,
};

use crate::collections::btree_map::Tuning;

/// In debug mode or feature unsafe-optim not enabled, same as assert! otherwise does nothing.
#[cfg(any(debug_assertions, not(feature = "unsafe-optim")))]
macro_rules! safe_assert {
    ( $cond: expr ) => {
        assert!($cond)
    };
}

/// In debug mode or feature unsafe-optim not enabled, same as assert! otherwise does nothing.
#[cfg(all(not(debug_assertions), feature = "unsafe-optim"))]
macro_rules! safe_assert {
    ( $cond: expr ) => {
        if !$cond {
            unsafe { std::hint::unreachable_unchecked() }
        }
    };
}

/// Vec with manual allocation.
pub struct ShortVec<T> {
    /// Pointer to elements.
    p: NonNull<T>,
    /// Current length.
    len: u16,
    /// Currently allocated.
    pub alloc: u16,
}

impl<T> Default for ShortVec<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> ShortVec<T> {
    pub fn new() -> Self {
        Self {
            len: 0,
            alloc: 0,
            p: NonNull::dangling(),
        }
    }

    pub fn len(&self) -> usize {
        self.len as usize
    }

    pub fn set_alloc<A: Tuning>(&mut self, na: usize, alloc: &A) {
        safe_assert!(na >= self.len());
        if na == self.alloc as usize {
            return;
        }
        unsafe {
            self.basic_set_alloc(self.alloc as usize, na, alloc);
        }
        self.alloc = na as u16;
    }

    #[inline]
    pub fn push(&mut self, value: T) {
        safe_assert!(self.len < self.alloc);
        unsafe {
            self.set(self.len(), value);
        }
        self.len += 1;
    }

    #[inline]
    pub fn pop(&mut self) -> Option<T> {
        if self.len == 0 {
            None
        } else {
            self.len -= 1;
            unsafe { Some(self.get(self.len())) }
        }
    }

    pub fn insert(&mut self, at: usize, value: T) {
        safe_assert!(self.len < self.alloc);
        unsafe {
            if at < self.len() {
                ptr::copy(self.ixp(at), self.ixp(at + 1), self.len() - at);
            }
            self.set(at, value);
            self.len += 1;
        }
    }

    pub fn remove(&mut self, at: usize) -> T {
        safe_assert!(at < self.len());
        unsafe {
            let result = self.get(at);
            ptr::copy(self.ixp(at + 1), self.ixp(at), self.len() - at - 1);
            self.len -= 1;
            result
        }
    }

    pub fn split<A: Tuning>(&mut self, at: usize, a1: usize, a2: usize, alloc: &A) -> Self {
        safe_assert!(at < self.len());
        let len = self.len() - at;
        safe_assert!(a1 >= at);
        safe_assert!(a2 >= len);
        let mut result = Self::new();
        result.set_alloc(a2, alloc);
        unsafe {
            ptr::copy_nonoverlapping(self.ixp(at), result.ixp(0), len);
        }
        result.len = len as u16;
        self.len = at as u16;
        self.set_alloc(a1, alloc);
        result
    }

    /// Get reference to ith element.
    #[inline]
    pub fn ix(&self, i: usize) -> &T {
        safe_assert!(i < self.len());
        unsafe { &*self.ixp(i) }
    }

    /// Get mutable reference to ith element.
    #[inline]
    pub fn ixm(&mut self, i: usize) -> &mut T {
        safe_assert!(i < self.len());
        unsafe { &mut *self.ixp(i) }
    }

    pub fn sv_into_iter(self) -> IntoIterShortVec<T> {
        IntoIterShortVec { start: 0, v: self }
    }

    /// Set capacity ( allocate or reallocate memory ).
    /// # Safety
    ///
    /// `oa` must be the previous alloc set (0 if no alloc has yet been set).
    unsafe fn basic_set_alloc<A: Tuning>(&mut self, oa: usize, na: usize, alloc: &A) {
        if mem::size_of::<T>() == 0 {
            return;
        }
        if na == 0 {
            alloc.deallocate(
                NonNull::new(self.p.as_ptr().cast::<u8>()).unwrap(),
                Layout::array::<T>(oa).unwrap(),
            );
            self.p = NonNull::dangling();
            return;
        }
        let new_layout = Layout::array::<T>(na).unwrap();
        let new_ptr = if oa == 0 {
            alloc.allocate(new_layout)
        } else {
            let old_layout = Layout::array::<T>(oa).unwrap();
            let old_ptr = self.p.as_ptr().cast::<u8>();
            let old_ptr = NonNull::new(old_ptr).unwrap();
            if new_layout.size() > old_layout.size() {
                alloc.grow(old_ptr, old_layout, new_layout)
            } else {
                alloc.shrink(old_ptr, old_layout, new_layout)
            }
        }
        .unwrap();
        self.p = NonNull::new(new_ptr.as_ptr().cast::<T>()).unwrap();
    }

    /// Get pointer to ith element.
    /// # Safety
    ///
    /// ix must be <= alloc.
    #[inline]
    unsafe fn ixp(&self, i: usize) -> *mut T {
        self.p.as_ptr().add(i)
    }

    /// Set value.
    /// # Safety
    ///
    /// ix must be < alloc, and the element must be unset.
    #[inline]
    unsafe fn set(&mut self, i: usize, elem: T) {
        ptr::write(self.ixp(i), elem);
    }

    /// Get value.
    /// # Safety
    ///
    /// ix must be less < alloc, and the element must have been set.
    #[inline]
    unsafe fn get(&mut self, i: usize) -> T {
        ptr::read(self.ixp(i))
    }
}

unsafe impl<T: Send> Send for ShortVec<T> {}
unsafe impl<T: Sync> Sync for ShortVec<T> {}

impl<T> Deref for ShortVec<T> {
    type Target = [T];
    #[inline]
    fn deref(&self) -> &[T] {
        let len: usize = ShortVec::len(self);
        unsafe { std::slice::from_raw_parts(self.p.as_ptr(), len) }
    }
}

impl<T> DerefMut for ShortVec<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut [T] {
        let len: usize = ShortVec::len(self);
        unsafe { std::slice::from_raw_parts_mut(self.p.as_ptr(), len) }
    }
}

impl<T> fmt::Debug for ShortVec<T>
where
    T: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Debug::fmt(&**self, f)
    }
}

pub struct IntoIterShortVec<T> {
    start: usize,
    v: ShortVec<T>,
}

impl<T> IntoIterShortVec<T> {
    #[inline]
    pub fn next<A: Tuning>(&mut self, alloc: &A) -> Option<T> {
        if self.start == self.v.len() {
            self.start = 0;
            self.v.len = 0;
            self.v.set_alloc(0, alloc);
            None
        } else {
            let ix = self.start;
            self.start += 1;
            Some(unsafe { self.v.get(ix) })
        }
    }

    #[inline]
    pub fn next_back<A: Tuning>(&mut self, alloc: &A) -> Option<T> {
        if self.start == self.v.len() {
            self.start = 0;
            self.v.len = 0;
            self.v.set_alloc(0, alloc);
            None
        } else {
            self.v.len -= 1;
            Some(unsafe { self.v.get(self.v.len()) })
        }
    }

    pub fn len(&self) -> usize {
        self.v.len() - self.start
    }
}

/// Vector of (key,value) pairs, keys stored separately from values for cache efficient search.
pub struct PairVec<K, V> {
    p: NonNull<u8>,
    len: u16, // Current length
    // Currently allocated.
    pub alloc: u16,
    _pd: PhantomData<(K, V)>,
}

impl<K, V> Default for PairVec<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

unsafe impl<K: Send, V: Send> Send for PairVec<K, V> {}
unsafe impl<K: Sync, V: Sync> Sync for PairVec<K, V> {}

impl<K, V> PairVec<K, V> {
    pub fn new() -> Self {
        Self {
            p: NonNull::dangling(),
            len: 0,
            alloc: 0,
            _pd: PhantomData,
        }
    }

    pub fn dealloc<A: Tuning>(&mut self, alloc: &A) {
        while self.len != 0 {
            self.pop();
        }
        self.set_alloc(0, alloc);
    }

    pub fn len(&self) -> usize {
        self.len as usize
    }

    pub fn state(&self) -> (usize, usize) {
        (self.len as usize, self.alloc as usize)
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn full(&self) -> bool {
        self.len == self.alloc
    }

    #[inline]
    unsafe fn layout(amount: usize) -> (Layout, usize) {
        let layout = Layout::array::<K>(amount).unwrap_unchecked();
        let (layout, off) = layout
            .extend(Layout::array::<V>(amount).unwrap_unchecked())
            .unwrap_unchecked();
        (layout, off)
    }

    #[inline]
    unsafe fn off(amount: usize) -> usize {
        Self::layout(amount).1
    }

    pub fn set_alloc<A: Tuning>(&mut self, na: usize, alloc: &A) {
        safe_assert!(na >= self.len());
        if mem::size_of::<K>() == 0 && mem::size_of::<V>() == 0 {
            self.alloc = na as u16;
            return;
        }
        if na == self.alloc as usize {
            return;
        }
        unsafe {
            let (old_layout, old_off) = Self::layout(self.alloc as usize);
            let np = if na == 0 {
                NonNull::dangling()
            } else {
                let (layout, off) = Self::layout(na);
                // println!("allocating old={} new={} size={}", self.alloc, na, layout.size());
                let np = alloc.allocate(layout).unwrap();
                let np = np.cast::<u8>();

                // Copy keys and values from old allocation to new allocation.
                if self.len > 0 {
                    let from = self.p.as_ptr().cast::<K>();
                    let to = np.as_ptr().cast::<K>();
                    ptr::copy_nonoverlapping(from, to, self.len as usize);

                    let from = self.p.as_ptr().add(old_off).cast::<V>();
                    let to = np.as_ptr().add(off).cast::<V>();
                    ptr::copy_nonoverlapping(from, to, self.len as usize);
                }
                np
            };

            // Free the old allocation.
            if self.alloc > 0 {
                alloc.deallocate(self.p, old_layout);
            }
            self.alloc = na as u16;
            self.p = np;
        }
    }

    pub fn split<A: Tuning>(
        &mut self,
        at: usize,
        a1: usize,
        a2: usize,
        alloc: &A,
    ) -> ((K, V), Self) {
        safe_assert!(at <= self.len());
        let x = at + 1;
        let len = self.len() - x;
        safe_assert!(a1 >= at);
        safe_assert!(a2 >= len);
        let mut result = Self::new();
        result.set_alloc(a2, alloc);
        unsafe {
            let (kf, vf) = self.ixmp(x);
            let (kt, vt) = result.ixmp(0);
            ptr::copy_nonoverlapping(kf, kt, len);
            ptr::copy_nonoverlapping(vf, vt, len);
        }
        result.len = len as u16;
        self.len -= len as u16;
        let med = self.pop().unwrap();
        self.set_alloc(a1, alloc);
        (med, result)
    }

    pub fn insert(&mut self, at: usize, (key, value): (K, V)) {
        safe_assert!(at <= self.len());
        safe_assert!(self.len < self.alloc);
        unsafe {
            let n = self.len() - at;
            let (kp, vp) = self.ixmp(at);
            if n > 0 {
                ptr::copy(kp, kp.add(1), n);
                ptr::copy(vp, vp.add(1), n);
            }
            kp.write(key);
            vp.write(value);
            self.len += 1;
        }
    }

    pub fn remove(&mut self, at: usize) -> (K, V) {
        safe_assert!(at < self.len());
        unsafe {
            let n = self.len() - at - 1;
            let (kp, vp) = self.ixmp(at);
            let result = (kp.read(), vp.read());
            if n > 0 {
                ptr::copy(kp.add(1), kp, n);
                ptr::copy(vp.add(1), vp, n);
            }
            self.len -= 1;
            result
        }
    }

    pub fn push(&mut self, (key, value): (K, V)) {
        safe_assert!(self.len < self.alloc);
        unsafe {
            let (kp, vp) = self.ixmp(self.len());
            kp.write(key);
            vp.write(value);
            self.len += 1;
        }
    }

    pub fn pop(&mut self) -> Option<(K, V)> {
        unsafe {
            if self.len == 0 {
                return None;
            }
            self.len -= 1;
            let (kp, vp) = self.ixmp(self.len());
            Some((kp.read(), vp.read()))
        }
    }

    #[inline]
    pub fn search<Q>(&self, key: &Q) -> Result<usize, usize>
    where
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        self.search_to(self.len as usize, key)
    }

    #[inline]
    pub fn search_to<Q>(&self, mut j: usize, key: &Q) -> Result<usize, usize>
    where
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        unsafe {
            let mut i = 0;
            let p = self.p.as_ptr().cast::<K>();
            while i != j {
                let m = (i + j) >> 1;
                match (*p.add(m)).borrow().cmp(key) {
                    Ordering::Equal => return Ok(m),
                    Ordering::Less => i = m + 1,
                    Ordering::Greater => j = m,
                }
            }
            Err(i)
        }
    }

    #[inline]
    unsafe fn ixmp(&mut self, i: usize) -> (*mut K, *mut V) {
        let off = Self::off(self.alloc as usize);
        let kp = self.p.as_ptr().cast::<K>().add(i);
        let vp = self.p.as_ptr().add(off).cast::<V>().add(i);
        (kp, vp)
    }

    #[inline]
    unsafe fn ixp(&self, i: usize) -> (*const K, *const V) {
        let off = Self::off(self.alloc as usize);
        let kp = self.p.as_ptr().cast::<K>().add(i);
        let vp = self.p.as_ptr().add(off).cast::<V>().add(i);
        (kp, vp)
    }

    #[inline]
    pub fn ixv(&self, i: usize) -> &V {
        safe_assert!(i < self.len());
        unsafe {
            let off = Self::off(self.alloc as usize);
            let vp = self.p.as_ptr().add(off).cast::<V>().add(i);
            &*vp
        }
    }

    #[inline]
    pub fn ixmv(&mut self, i: usize) -> &mut V {
        safe_assert!(i < self.len());
        unsafe {
            let (_kp, vp) = self.ixmp(i);
            &mut *vp
        }
    }

    #[inline]
    pub fn ix(&self, i: usize) -> (&K, &V) {
        safe_assert!(i < self.len());
        unsafe {
            let (kp, vp) = self.ixp(i);
            (&*kp, &*vp)
        }
    }

    #[inline]
    pub fn ixbm(&mut self, i: usize) -> (&mut K, &mut V) {
        safe_assert!(i < self.len());
        unsafe {
            let (kp, vp) = self.ixmp(i);
            (&mut *kp, &mut *vp)
        }
    }

    pub fn iter(&self) -> IterPairVec<K, V> {
        IterPairVec {
            v: Some(self),
            ix: 0,
            ixb: self.len(),
        }
    }

    pub fn range(&self, x: usize, y: usize) -> IterPairVec<K, V> {
        safe_assert!(x <= y && y <= self.len());
        IterPairVec {
            v: Some(self),
            ix: x,
            ixb: y,
        }
    }

    pub fn iter_mut(&mut self) -> IterMutPairVec<K, V> {
        let ixb = self.len();
        IterMutPairVec {
            v: Some(self),
            ix: 0,
            ixb,
        }
    }

    pub fn range_mut(&mut self, x: usize, y: usize) -> IterMutPairVec<K, V> {
        safe_assert!(x <= y && y <= self.len());
        IterMutPairVec {
            v: Some(self),
            ix: x,
            ixb: y,
        }
    }

    pub fn into_iter(self) -> IntoIterPairVec<K, V> {
        let ixb = self.len();
        IntoIterPairVec {
            v: self,
            ix: 0,
            ixb,
        }
    }

    pub fn clone<A: Tuning>(&self, alloc: &A) -> Self
    where
        K: Clone,
        V: Clone,
    {
        let mut c = PairVecDropper {
            v: Self::new(),
            alloc,
        };
        c.v.set_alloc(self.alloc as usize, alloc);
        let mut n = self.len;
        if n > 0 {
            unsafe {
                let (mut sk, mut sv) = self.ixp(0);
                let (mut dk, mut dv) = c.v.ixmp(0);
                loop {
                    let k = (*sk).clone();
                    let v = (*sv).clone();
                    dk.write(k);
                    dv.write(v);
                    c.v.len += 1;
                    n -= 1;
                    if n == 0 {
                        break;
                    }
                    sk = sk.add(1);
                    sv = sv.add(1);
                    dk = dk.add(1);
                    dv = dv.add(1);
                }
            }
        }
        mem::take(&mut c.v)
    }

    pub fn get_xy<T, R>(&self, range: &R) -> (usize, usize)
    where
        T: Ord + ?Sized,
        K: Borrow<T> + Ord,
        R: RangeBounds<T>,
    {
        let y = self.get_upper(range.end_bound());
        let x = match range.start_bound() {
            Bound::Unbounded => 0,
            Bound::Included(k) => match self.search_to(y, k) {
                Ok(i) | Err(i) => i,
            },
            Bound::Excluded(k) => match self.search_to(y, k) {
                Ok(i) => i + 1,
                Err(i) => i,
            },
        };
        (x, y)
    }

    fn skip<Q>(&self, key: &Q) -> usize
    where
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        match self.search(key) {
            Ok(i) | Err(i) => i,
        }
    }

    fn skip_over<Q>(&self, key: &Q) -> usize
    where
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        match self.search(key) {
            Ok(i) => i + 1,
            Err(i) => i,
        }
    }

    pub fn get_lower<Q>(&self, bound: Bound<&Q>) -> usize
    where
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        match bound {
            Bound::Unbounded => 0,
            Bound::Included(k) => self.skip(k),
            Bound::Excluded(k) => self.skip_over(k),
        }
    }

    pub fn get_upper<Q>(&self, bound: Bound<&Q>) -> usize
    where
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        match bound {
            Bound::Unbounded => self.len(),
            Bound::Included(k) => self.skip_over(k),
            Bound::Excluded(k) => self.skip(k),
        }
    }
}

struct PairVecDropper<'a, K, V, A: Tuning> {
    v: PairVec<K, V>,
    alloc: &'a A,
}

impl<'a, K, V, A: Tuning> Drop for PairVecDropper<'a, K, V, A> {
    fn drop(&mut self) {
        self.v.dealloc(self.alloc);
    }
}

impl<K, V> fmt::Debug for PairVec<K, V>
where
    K: Debug,
    V: Debug,
{
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt.debug_map().entries(self.iter()).finish()
    }
}

#[derive(Debug, Clone)]
pub struct IterPairVec<'a, K, V> {
    v: Option<&'a PairVec<K, V>>,
    ix: usize,
    ixb: usize,
}
impl<'a, K, V> Default for IterPairVec<'a, K, V> {
    fn default() -> Self {
        Self {
            v: None,
            ix: 0,
            ixb: 0,
        }
    }
}
impl<'a, K, V> IterPairVec<'a, K, V> {
    pub fn len(&self) -> usize {
        self.ixb - self.ix
    }
}
impl<'a, K, V> Iterator for IterPairVec<'a, K, V> {
    type Item = (&'a K, &'a V);
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.ix == self.ixb {
            return None;
        }
        let v = unsafe { self.v.unwrap_unchecked() };
        let kv = v.ix(self.ix);
        self.ix += 1;
        Some(kv)
    }
}
impl<'a, K, V> DoubleEndedIterator for IterPairVec<'a, K, V> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.ix == self.ixb {
            return None;
        }
        let v = unsafe { self.v.unwrap_unchecked() };
        self.ixb -= 1;
        let kv = v.ix(self.ixb);
        Some(kv)
    }
}
#[derive(Debug)]
pub struct IterMutPairVec<'a, K, V> {
    v: Option<&'a mut PairVec<K, V>>,
    ix: usize,
    ixb: usize,
}
impl<'a, K, V> Default for IterMutPairVec<'a, K, V> {
    fn default() -> Self {
        Self {
            v: None,
            ix: 0,
            ixb: 0,
        }
    }
}
impl<'a, K, V> IterMutPairVec<'a, K, V> {
    pub fn len(&self) -> usize {
        self.ixb - self.ix
    }
}
impl<'a, K, V> Iterator for IterMutPairVec<'a, K, V> {
    type Item = (&'a K, &'a mut V);
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        unsafe {
            if self.ix == self.ixb {
                return None;
            }
            let v = self.v.as_mut().unwrap_unchecked();
            let (kp, vp) = v.ixmp(self.ix);
            self.ix += 1;
            Some((&mut *kp, &mut *vp))
        }
    }
}
impl<'a, K, V> DoubleEndedIterator for IterMutPairVec<'a, K, V> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        unsafe {
            if self.ix == self.ixb {
                return None;
            }
            self.ixb -= 1;
            let v = self.v.as_mut().unwrap_unchecked();
            let (kp, vp) = v.ixmp(self.ixb);
            Some((&mut *kp, &mut *vp))
        }
    }
}

#[derive(Debug)]
pub struct IntoIterPairVec<K, V> {
    v: PairVec<K, V>,
    ix: usize,
    ixb: usize,
}
impl<K, V> IntoIterPairVec<K, V> {
    pub fn len(&self) -> usize {
        self.ixb - self.ix
    }
    pub fn empty() -> Self {
        Self {
            v: PairVec::default(),
            ix: 0,
            ixb: 0,
        }
    }

    #[inline]
    pub fn next<A: Tuning>(&mut self, alloc: &A) -> Option<(K, V)> {
        unsafe {
            if self.ix == self.ixb {
                self.v.len = 0;
                self.v.set_alloc(0, alloc);
                return None;
            }
            let (kp, vp) = self.v.ixmp(self.ix);
            self.ix += 1;
            Some((kp.read(), vp.read()))
        }
    }

    #[inline]
    pub fn next_back<A: Tuning>(&mut self, alloc: &A) -> Option<(K, V)> {
        unsafe {
            if self.ix == self.ixb {
                self.v.len = 0;
                self.v.set_alloc(0, alloc);
                return None;
            }
            self.ixb -= 1;
            let (kp, vp) = self.v.ixmp(self.ixb);
            Some((kp.read(), vp.read()))
        }
    }
}
