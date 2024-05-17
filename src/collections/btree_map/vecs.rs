use std::{
    alloc,
    alloc::Layout,
    borrow::Borrow,
    cmp::Ordering,
    fmt,
    fmt::Debug,
    marker::PhantomData,
    mem,
    ops::{Deref, DerefMut},
    ptr,
    ptr::NonNull,
};

/// Basic vec, does not have own capacity or length, just a pointer to memory.
/// Kind-of cribbed from <https://doc.rust-lang.org/nomicon/vec/vec-final.html>.
struct BasicVec<T> {
    p: NonNull<T>,
}

unsafe impl<T: Send> Send for BasicVec<T> {}
unsafe impl<T: Sync> Sync for BasicVec<T> {}

impl<T> Default for BasicVec<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> BasicVec<T> {
    /// Construct new `BasicVec`.
    pub fn new() -> Self {
        Self {
            p: NonNull::dangling(),
        }
    }

    /// Get mutable raw pointer to specified element.
    /// # Safety
    /// index must be < set allocation.
    #[inline]
    pub unsafe fn ix(&self, index: usize) -> *mut T {
        self.p.as_ptr().add(index)
    }

    /// Set capacity ( allocate or reallocate memory ).
    /// # Safety
    ///
    /// `oa` must be the previous alloc set (0 if no alloc has yet been set).
    pub unsafe fn set_alloc(&mut self, oa: usize, na: usize) {
        if mem::size_of::<T>() == 0 {
            return;
        }
        if na == 0 {
            self.free(oa);
            return;
        }
        let new_layout = Layout::array::<T>(na).unwrap();

        let new_ptr = if oa == 0 {
            alloc::alloc(new_layout)
        } else {
            let old_layout = Layout::array::<T>(oa).unwrap();
            let old_ptr = self.p.as_ptr().cast::<u8>();
            alloc::realloc(old_ptr, old_layout, new_layout.size())
        };

        // If allocation fails, `new_ptr` will be null, in which case we abort.
        self.p = match NonNull::new(new_ptr.cast::<T>()) {
            Some(p) => p,
            None => alloc::handle_alloc_error(new_layout),
        };
    }

    /// Free memory.
    /// # Safety
    ///
    /// oa must be the last allocation set.
    pub unsafe fn free(&mut self, oa: usize) {
        let elem_size = mem::size_of::<T>();
        if oa != 0 && elem_size != 0 {
            alloc::dealloc(
                self.p.as_ptr().cast::<u8>(),
                Layout::array::<T>(oa).unwrap(),
            );
        }
        self.p = NonNull::dangling();
    }

    /// Set value.
    /// # Safety
    ///
    /// ix must be < alloc, and the element must be unset.
    #[inline]
    pub unsafe fn set(&mut self, i: usize, elem: T) {
        ptr::write(self.ix(i), elem);
    }

    /// Get value.
    /// # Safety
    ///
    /// ix must be less < alloc, and the element must have been set.
    #[inline]
    pub unsafe fn get(&mut self, i: usize) -> T {
        ptr::read(self.ix(i))
    }

    /// Get whole as slice.
    /// # Safety
    ///
    /// len must be <= alloc and 0..len elements must have been set.
    #[inline]
    pub unsafe fn slice(&self, len: usize) -> &[T] {
        std::slice::from_raw_parts(self.p.as_ptr(), len)
    }

    /// Get whole as mut slice.
    /// # Safety
    ///
    /// len must be <= alloc and 0..len elements must have been set.
    #[inline]
    pub unsafe fn slice_mut(&mut self, len: usize) -> &mut [T] {
        std::slice::from_raw_parts_mut(self.p.as_ptr(), len)
    }

    /// Move elements.
    /// # Safety
    ///
    /// The set status of the elements changes in the obvious way. from, to and len must be in range.
    pub unsafe fn move_self(&mut self, from: usize, to: usize, len: usize) {
        ptr::copy(self.ix(from), self.ix(to), len);
    }

    /// Move elements from another `BasicVec`.
    /// # Safety
    ///
    /// The set status of the elements changes in the obvious way. from, to and len must be in range.
    pub unsafe fn move_from(&mut self, from: usize, src: &mut Self, to: usize, len: usize) {
        ptr::copy_nonoverlapping(src.ix(from), self.ix(to), len);
    }
}

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
    len: u16, // Current length.
    /// Currently allocated.
    pub alloc: u16,
    v: BasicVec<T>,
}

impl<T> Default for ShortVec<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Drop for ShortVec<T> {
    fn drop(&mut self) {
        let mut len = self.len as usize;
        while len > 0 {
            len -= 1;
            unsafe {
                self.v.get(len);
            }
        }
        self.len = 0;
        self.set_alloc(0);
    }
}

impl<T> ShortVec<T> {
    pub fn new() -> Self {
        let v = BasicVec::new();
        Self {
            len: 0,
            alloc: 0,
            v,
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len as usize
    }

    pub fn set_alloc(&mut self, na: usize) {
        safe_assert!(na >= self.len());
        if na == self.alloc as usize {
            return;
        }
        unsafe {
            self.v.set_alloc(self.alloc as usize, na);
        }
        self.alloc = na as u16;
    }

    #[inline]
    pub fn push(&mut self, value: T) {
        safe_assert!(self.len < self.alloc);
        unsafe {
            self.v.set(self.len(), value);
        }
        self.len += 1;
    }

    #[inline]
    pub fn pop(&mut self) -> Option<T> {
        if self.len == 0 {
            None
        } else {
            self.len -= 1;
            unsafe { Some(self.v.get(self.len())) }
        }
    }

    pub fn insert(&mut self, at: usize, value: T) {
        safe_assert!(self.len < self.alloc);
        unsafe {
            if at < self.len() {
                self.v.move_self(at, at + 1, self.len() - at);
            }
            self.v.set(at, value);
            self.len += 1;
        }
    }

    pub fn remove(&mut self, at: usize) -> T {
        safe_assert!(at < self.len());
        unsafe {
            let result = self.v.get(at);
            self.v.move_self(at + 1, at, self.len() - at - 1);
            self.len -= 1;
            result
        }
    }

    pub fn split(&mut self, at: usize, a1: usize, a2: usize) -> Self {
        safe_assert!(at < self.len());
        let len = self.len() - at;
        safe_assert!(a1 >= at);
        safe_assert!(a2 >= len);
        let mut result = Self::new();
        result.set_alloc(a2);
        unsafe {
            result.v.move_from(at, &mut self.v, 0, len);
        }
        result.len = len as u16;
        self.len = at as u16;
        self.set_alloc(a1);
        result
    }

    /// Get reference to ith element.
    #[inline]
    pub fn ix(&self, i: usize) -> &T {
        safe_assert!(i < self.len());
        unsafe { &*self.v.ix(i) }
    }

    /// Get mutable reference to ith element.
    #[inline]
    pub fn ixm(&mut self, i: usize) -> &mut T {
        safe_assert!(i < self.len());
        unsafe { &mut *self.v.ix(i) }
    }
}

impl<T> Clone for ShortVec<T>
where
    T: Clone,
{
    fn clone(&self) -> Self {
        let mut c = Self::new();
        c.set_alloc(self.alloc as usize);
        let mut n = self.len;
        if n > 0 {
            unsafe {
                let mut src = self.v.p.as_ptr();
                let mut dest = c.v.p.as_ptr();
                loop {
                    dest.write((*src).clone());
                    c.len += 1;
                    n -= 1;
                    if n == 0 {
                        break;
                    }
                    src = src.add(1);
                    dest = dest.add(1);
                }
            }
        }
        c
    }
}

impl<T> Deref for ShortVec<T> {
    type Target = [T];
    #[inline]
    fn deref(&self) -> &[T] {
        let len: usize = ShortVec::len(self);
        unsafe { self.v.slice(len) }
    }
}

impl<T> DerefMut for ShortVec<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut [T] {
        let len: usize = ShortVec::len(self);
        unsafe { self.v.slice_mut(len) }
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

impl<T> IntoIterator for ShortVec<T> {
    type Item = T;
    type IntoIter = IntoIterShortVec<T>;
    fn into_iter(self) -> Self::IntoIter {
        IntoIterShortVec { start: 0, v: self }
    }
}

pub struct IntoIterShortVec<T> {
    start: usize,
    v: ShortVec<T>,
}

impl<T> Iterator for IntoIterShortVec<T> {
    type Item = T;
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.start == self.v.len() {
            None
        } else {
            let ix = self.start;
            self.start += 1;
            Some(unsafe { self.v.v.get(ix) })
        }
    }
}
impl<T> DoubleEndedIterator for IntoIterShortVec<T> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.start == self.v.len() {
            None
        } else {
            self.v.len -= 1;
            Some(unsafe { self.v.v.get(self.v.len()) })
        }
    }
}
impl<T> Drop for IntoIterShortVec<T> {
    fn drop(&mut self) {
        while self.len() > 0 {
            self.next();
        }
        self.v.len = 0;
    }
}
impl<T> ExactSizeIterator for IntoIterShortVec<T> {
    fn len(&self) -> usize {
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

impl<K, V> Drop for PairVec<K, V> {
    fn drop(&mut self) {
        while self.len != 0 {
            self.pop();
        }
        self.set_alloc(0);
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

    pub fn set_alloc(&mut self, na: usize) {
        safe_assert!(na >= self.len());
        if na == self.alloc as usize {
            return;
        }
        if mem::size_of::<K>() == 0 && mem::size_of::<V>() == 0 {
            self.alloc = na as u16;
            return;
        }
        unsafe {
            let (old_layout, old_off) = Self::layout(self.alloc as usize);

            let np = if na == 0 {
                NonNull::dangling()
            } else {
                let (layout, off) = Self::layout(na);
                let np = alloc::alloc(layout);
                let np = match NonNull::new(np.cast::<u8>()) {
                    Some(np) => np,
                    None => alloc::handle_alloc_error(layout),
                };

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
                alloc::dealloc(self.p.as_ptr(), old_layout);
            }
            self.alloc = na as u16;
            self.p = np;
        }
    }

    pub fn split(&mut self, at: usize, a1: usize, a2: usize) -> ((K, V), Self) {
        safe_assert!(at <= self.len());
        let x = at + 1;
        let len = self.len() - x;
        safe_assert!(a1 >= at);
        safe_assert!(a2 >= len);
        let mut result = Self::new();
        result.set_alloc(a2);
        unsafe {
            let (kf, vf) = self.ixmp(x);
            let (kt, vt) = result.ixmp(0);
            ptr::copy_nonoverlapping(kf, kt, len);
            ptr::copy_nonoverlapping(vf, vt, len);
        }
        result.len = len as u16;
        self.len -= len as u16;
        let med = self.pop().unwrap();
        self.set_alloc(a1);
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
}

impl<K, V> Clone for PairVec<K, V>
where
    K: Clone,
    V: Clone,
{
    fn clone(&self) -> Self {
        let mut c = Self::new();
        c.set_alloc(self.alloc as usize);
        let mut n = self.len;
        if n > 0 {
            unsafe {
                let (mut sk, mut sv) = self.ixp(0);
                let (mut dk, mut dv) = c.ixmp(0);
                loop {
                    let k = (*sk).clone();
                    let v = (*sv).clone();
                    dk.write(k);
                    dv.write(v);
                    c.len += 1;
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
        c
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
impl<K, V> Default for IntoIterPairVec<K, V> {
    fn default() -> Self {
        Self {
            v: PairVec::default(),
            ix: 0,
            ixb: 0,
        }
    }
}
impl<K, V> IntoIterPairVec<K, V> {
    pub fn len(&self) -> usize {
        self.ixb - self.ix
    }
}
impl<K, V> Iterator for IntoIterPairVec<K, V> {
    type Item = (K, V);
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        unsafe {
            if self.ix == self.ixb {
                return None;
            }
            let (kp, vp) = self.v.ixmp(self.ix);
            self.ix += 1;
            Some((kp.read(), vp.read()))
        }
    }
}
impl<K, V> DoubleEndedIterator for IntoIterPairVec<K, V> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        unsafe {
            if self.ix == self.ixb {
                return None;
            }
            self.ixb -= 1;
            let (kp, vp) = self.v.ixmp(self.ixb);
            Some((kp.read(), vp.read()))
        }
    }
}
impl<K, V> Drop for IntoIterPairVec<K, V> {
    fn drop(&mut self) {
        while self.ix != self.ixb {
            self.next();
        }
        self.v.len = 0;
    }
}
