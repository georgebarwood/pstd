//! [`BTreeSet`] similar to [`std::collections::BTreeSet`].

use crate::collections::btree_map::*;
use std::borrow::Borrow;
use std::cmp::Ordering;
use std::fmt::{self, Debug};
use std::hash::Hash;
use std::hash::Hasher;
use std::iter::FusedIterator;
use std::ops::Bound;

/// `BTreeSet` similar to [`std::collections::BTreeSet`].
/// An ordered set based on a B-Tree.
///
/// See [`BTreeMap`]'s documentation for a detailed discussion of this collection's performance
/// benefits and drawbacks.
///
/// # Examples
///
/// ```
/// use pstd::collections::BTreeSet;
///
/// // Type inference lets us omit an explicit type signature (which
/// // would be `BTreeSet<&str>` in this example).
/// let mut books = BTreeSet::new();
///
/// // Add some books.
/// books.insert("A Dance With Dragons");
/// books.insert("To Kill a Mockingbird");
/// books.insert("The Odyssey");
/// books.insert("The Great Gatsby");
///
/// // Check for a specific one.
/// if !books.contains("The Winds of Winter") {
///     println!("We have {} books, but The Winds of Winter ain't one.",
///              books.len());
/// }
///
/// // Remove a book.
/// books.remove("The Odyssey");
///
/// // Iterate over everything.
/// for book in &books {
///     println!("{book}");
/// }
/// ```
///
/// A `BTreeSet` with a known list of items can be initialized from an array:
///
/// ```
/// use pstd::collections::BTreeSet;
///
/// let set = BTreeSet::from([1, 2, 3]);
/// ```
pub struct BTreeSet<T, A: Tuning = DefaultTuning> {
    map: BTreeMap<T, (), A>,
}

impl<T> BTreeSet<T> {
    /// Returns a new, empty `BTreeSet`.
    ///
    /// # Examples
    ///
    /// ```
    /// # #![allow(unused_mut)]
    /// use pstd::collections::BTreeSet;
    ///
    /// let mut set: BTreeSet<i32> = BTreeSet::new();
    /// ```
    #[must_use]
    pub const fn new() -> BTreeSet<T> {
        BTreeSet {
            map: BTreeMap::new(),
        }
    }
}

impl<T, A: Tuning> BTreeSet<T, A> {
    /// Returns number of elements in the set
    ///
    /// # Examples
    ///
    /// ```
    /// use pstd::collections::BTreeSet;
    ///
    /// let mut v = BTreeSet::new();
    /// assert_eq!(v.len(), 0);
    /// v.insert(1);
    /// assert_eq!(v.len(), 1);
    /// ```
    pub const fn len(&self) -> usize {
        self.map.len()
    }

    /// Does the set have any elements
    ///
    /// # Examples
    ///
    /// ```
    /// use pstd::collections::BTreeSet;
    ///
    /// let mut v = BTreeSet::new();
    /// assert!(v.is_empty());
    /// v.insert(1);
    /// assert!(!v.is_empty());
    /// ```
    pub const fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Returns new set with specified tuning
    #[must_use]
    pub const fn new_in(tuning: A) -> BTreeSet<T, A> {
        BTreeSet {
            map: BTreeMap::with_tuning(tuning),
        }
    }

    /// Clears the set, removing all elements.
    ///
    /// # Examples
    ///
    /// ```
    /// use pstd::collections::BTreeSet;
    ///
    /// let mut v = BTreeSet::new();
    /// v.insert(1);
    /// v.clear();
    /// assert!(v.is_empty());
    /// ```
    pub fn clear(&mut self) {
        self.map.clear();
    }

    /// Adds a value to the set.
    ///
    /// # Examples
    ///
    /// ```
    /// use pstd::collections::BTreeSet;
    ///
    /// let mut set = BTreeSet::new();
    /// set.insert(Vec::<i32>::new());
    ///
    /// assert_eq!(set.get(&[][..]).unwrap().capacity(), 0);
    /// set.replace(Vec::with_capacity(10));
    /// assert_eq!(set.get(&[][..]).unwrap().capacity(), 10);
    /// ```
    pub fn insert(&mut self, value: T) -> bool
    where
        T: Ord,
    {
        self.map.insert(value, ()).is_none()
    }

    /// Remove element from set
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::BTreeSet;
    ///
    /// let mut set = BTreeSet::new();
    ///
    /// set.insert(2);
    /// assert_eq!(set.remove(&2), true);
    /// assert_eq!(set.remove(&2), false);
    /// ```
    pub fn remove<Q>(&mut self, value: &Q) -> bool
    where
        T: Borrow<Q> + Ord,
        Q: ?Sized + Ord,
    {
        self.map.remove(value).is_some()
    }

    /// Remove and return element from set
    ///
    /// The value may be any borrowed form of the set's element type,
    /// but the ordering on the borrowed form *must* match the
    /// ordering on the element type.
    ///
    /// # Examples
    ///
    /// ```
    /// use pstd::collections::BTreeSet;
    ///
    /// let mut set = BTreeSet::from([1, 2, 3]);
    /// assert_eq!(set.take(&2), Some(2));
    /// assert_eq!(set.take(&2), None);
    /// ```
    pub fn take<Q>(&mut self, value: &Q) -> Option<T>
    where
        T: Borrow<Q> + Ord,
        Q: ?Sized + Ord,
    {
        self.map.remove_entry(value).map(|(k, _)| k)
    }

    /// Adds a value to the set, replacing the existing element, if any, that is
    /// equal to the value. Returns the replaced element.
    ///
    /// # Examples
    ///
    /// ```
    /// use pstd::collections::BTreeSet;
    ///
    /// let mut set = BTreeSet::new();
    /// set.insert(Vec::<i32>::new());
    ///
    /// assert_eq!(set.get(&[][..]).unwrap().capacity(), 0);
    /// set.replace(Vec::with_capacity(10));
    /// assert_eq!(set.get(&[][..]).unwrap().capacity(), 10);
    /// ```
    pub fn replace(&mut self, value: T) -> Option<T>
    where
        T: Ord,
    {
        // self.map.replace(value); // optimised version is todo
        let result = self.map.remove_entry(&value).map(|(k, _v)| k);
        self.insert(value);
        result
    }

    /// Returns `true` if the set contains an element equal to the value.
    ///
    /// The value may be any borrowed form of the set's element type,
    /// but the ordering on the borrowed form *must* match the
    /// ordering on the element type.
    ///
    /// # Examples
    ///
    /// ```
    /// use pstd::collections::BTreeSet;
    ///
    /// let set = BTreeSet::from([1, 2, 3]);
    /// assert_eq!(set.contains(&1), true);
    /// assert_eq!(set.contains(&4), false);
    /// ```
    pub fn contains<Q>(&self, value: &Q) -> bool
    where
        T: Borrow<Q> + Ord,
        Q: ?Sized + Ord,
    {
        self.map.contains_key(value)
    }

    /// Returns a reference to the element in the set, if any, that is equal to
    /// the value.
    ///
    /// The value may be any borrowed form of the set's element type,
    /// but the ordering on the borrowed form *must* match the
    /// ordering on the element type.
    ///
    /// # Examples
    ///
    /// ```
    /// use pstd::collections::BTreeSet;
    ///
    /// let set = BTreeSet::from([1, 2, 3]);
    /// assert_eq!(set.get(&2), Some(&2));
    /// assert_eq!(set.get(&4), None);
    /// ```
    pub fn get<Q>(&self, value: &Q) -> Option<&T>
    where
        T: Borrow<Q> + Ord,
        Q: ?Sized + Ord,
    {
        self.map.get_key_value(value).map(|(k, _)| k)
    }

    /// Retains only the elements specified by the predicate.
    ///
    /// In other words, remove all elements `e` for which `f(&e)` returns `false`.
    /// The elements are visited in ascending order.
    ///
    /// # Examples
    ///
    /// ```
    /// use pstd::collections::BTreeSet;
    ///
    /// let mut set = BTreeSet::from([1, 2, 3, 4, 5, 6]);
    /// // Keep only the even numbers.
    /// set.retain(|&k| k % 2 == 0);
    /// assert!(set.iter().eq([2, 4, 6].iter()));
    /// ```
    pub fn retain<F>(&mut self, mut f: F)
    where
        T: Ord,
        F: FnMut(&T) -> bool,
    {
        self.map.retain(|k, _| f(k));
    }

    /// Moves all elements from `other` into `self`, leaving `other` empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use pstd::collections::BTreeSet;
    ///
    /// let mut a = BTreeSet::new();
    /// a.insert(1);
    /// a.insert(2);
    /// a.insert(3);
    ///
    /// let mut b = BTreeSet::new();
    /// b.insert(3);
    /// b.insert(4);
    /// b.insert(5);
    ///
    /// a.append(&mut b);
    ///
    /// assert_eq!(a.len(), 5);
    /// assert_eq!(b.len(), 0);
    ///
    /// assert!(a.contains(&1));
    /// assert!(a.contains(&2));
    /// assert!(a.contains(&3));
    /// assert!(a.contains(&4));
    /// assert!(a.contains(&5));
    /// ```
    pub fn append(&mut self, other: &mut Self)
    where
        T: Ord,
        A: Clone,
    {
        self.map.append(&mut other.map);
    }

    /// Gets an iterator that visits the elements in the `BTreeSet` in ascending
    /// order.
    ///
    /// # Examples
    ///
    /// ```
    /// use pstd::collections::BTreeSet;
    ///
    /// let set = BTreeSet::from([3, 1, 2]);
    /// let mut set_iter = set.iter();
    /// assert_eq!(set_iter.next(), Some(&1));
    /// assert_eq!(set_iter.next(), Some(&2));
    /// assert_eq!(set_iter.next(), Some(&3));
    /// assert_eq!(set_iter.next(), None);
    /// ```
    pub fn iter(&self) -> Iter<'_, T> {
        Iter {
            iter: self.map.keys(),
        }
    }

    /// Returns a reference to the first element in the set, if any.
    /// This element is always the minimum of all elements in the set.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// use pstd::collections::BTreeSet;
    ///
    /// let mut set = BTreeSet::new();
    /// assert_eq!(set.first(), None);
    /// set.insert(1);
    /// assert_eq!(set.first(), Some(&1));
    /// set.insert(2);
    /// assert_eq!(set.first(), Some(&1));
    /// ```
    #[must_use]
    pub fn first(&self) -> Option<&T>
    where
        T: Ord,
    {
        self.map.first_key_value().map(|(k, _)| k)
    }

    /// Returns a reference to the last element in the set, if any.
    /// This element is always the maximum of all elements in the set.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// use pstd::collections::BTreeSet;
    ///
    /// let mut set = BTreeSet::new();
    /// assert_eq!(set.last(), None);
    /// set.insert(1);
    /// assert_eq!(set.last(), Some(&1));
    /// set.insert(2);
    /// assert_eq!(set.last(), Some(&2));
    /// ```
    #[must_use]
    pub fn last(&self) -> Option<&T>
    where
        T: Ord,
    {
        self.map.last_key_value().map(|(k, _)| k)
    }

    /// Removes the first element from the set and returns it, if any.
    /// The first element is always the minimum element in the set.
    ///
    /// # Examples
    ///
    /// ```
    /// use pstd::collections::BTreeSet;
    ///
    /// let mut set = BTreeSet::new();
    ///
    /// set.insert(1);
    /// while let Some(n) = set.pop_first() {
    ///     assert_eq!(n, 1);
    /// }
    /// assert!(set.is_empty());
    /// ```
    pub fn pop_first(&mut self) -> Option<T>
    where
        T: Ord,
    {
        self.map.pop_first().map(|kv| kv.0)
    }

    /// Removes the last element from the set and returns it, if any.
    /// The last element is always the maximum element in the set.
    ///
    /// # Examples
    ///
    /// ```
    /// use pstd::collections::BTreeSet;
    ///
    /// let mut set = BTreeSet::new();
    ///
    /// set.insert(1);
    /// while let Some(n) = set.pop_last() {
    ///     assert_eq!(n, 1);
    /// }
    /// assert!(set.is_empty());
    /// ```
    pub fn pop_last(&mut self) -> Option<T>
    where
        T: Ord,
    {
        self.map.pop_last().map(|kv| kv.0)
    }

    /// Returns a [`Cursor`] pointing at the gap before the smallest element
    /// greater than the given bound.
    ///
    /// Passing `Bound::Included(x)` will return a cursor pointing to the
    /// gap before the smallest element greater than or equal to `x`.
    ///
    /// Passing `Bound::Excluded(x)` will return a cursor pointing to the
    /// gap before the smallest element greater than `x`.
    ///
    /// Passing `Bound::Unbounded` will return a cursor pointing to the
    /// gap before the smallest element in the set.
    ///
    /// # Examples
    ///
    /// ```
    ///
    /// use pstd::collections::BTreeSet;
    /// use std::ops::Bound;
    ///
    /// let set = BTreeSet::from([1, 2, 3, 4]);
    ///
    /// let cursor = set.lower_bound(Bound::Included(&2));
    /// assert_eq!(cursor.peek_prev(), Some(&1));
    /// assert_eq!(cursor.peek_next(), Some(&2));
    ///
    /// let cursor = set.lower_bound(Bound::Excluded(&2));
    /// assert_eq!(cursor.peek_prev(), Some(&2));
    /// assert_eq!(cursor.peek_next(), Some(&3));
    ///
    /// let cursor = set.lower_bound(Bound::Unbounded);
    /// assert_eq!(cursor.peek_prev(), None);
    /// assert_eq!(cursor.peek_next(), Some(&1));
    /// ```
    pub fn lower_bound<Q>(&self, bound: Bound<&Q>) -> Cursor<'_, T, A>
    where
        T: Borrow<Q> + Ord,
        Q: ?Sized + Ord,
    {
        Cursor { inner: self.map.lower_bound(bound) }
    }

    /// Returns a [`Cursor`] pointing at the gap after the greatest element
    /// smaller than the given bound.
    ///
    /// Passing `Bound::Included(x)` will return a cursor pointing to the
    /// gap after the greatest element smaller than or equal to `x`.
    ///
    /// Passing `Bound::Excluded(x)` will return a cursor pointing to the
    /// gap after the greatest element smaller than `x`.
    ///
    /// Passing `Bound::Unbounded` will return a cursor pointing to the
    /// gap after the greatest element in the set.
    ///
    /// # Examples
    ///
    /// ```
    /// use pstd::collections::BTreeSet;
    /// use std::ops::Bound;
    ///
    /// let set = BTreeSet::from([1, 2, 3, 4]);
    ///
    /// let cursor = set.upper_bound(Bound::Included(&3));
    /// assert_eq!(cursor.peek_prev(), Some(&3));
    /// assert_eq!(cursor.peek_next(), Some(&4));
    ///
    /// let cursor = set.upper_bound(Bound::Excluded(&3));
    /// assert_eq!(cursor.peek_prev(), Some(&2));
    /// assert_eq!(cursor.peek_next(), Some(&3));
    ///
    /// let cursor = set.upper_bound(Bound::Unbounded);
    /// assert_eq!(cursor.peek_prev(), Some(&4));
    /// assert_eq!(cursor.peek_next(), None);
    /// ```
    pub fn upper_bound<Q>(&self, bound: Bound<&Q>) -> Cursor<'_, T, A>
    where
        T: Borrow<Q> + Ord,
        Q: ?Sized + Ord,
    {
        Cursor { inner: self.map.upper_bound(bound) }
    }

    /// Returns a [`CursorMut`] pointing at the gap before the smallest element
    /// greater than the given bound.
    ///
    /// Passing `Bound::Included(x)` will return a cursor pointing to the
    /// gap before the smallest element greater than or equal to `x`.
    ///
    /// Passing `Bound::Excluded(x)` will return a cursor pointing to the
    /// gap before the smallest element greater than `x`.
    ///
    /// Passing `Bound::Unbounded` will return a cursor pointing to the
    /// gap before the smallest element in the set.
    ///
    /// # Examples
    ///
    /// ```
    /// use pstd::collections::BTreeSet;
    /// use std::ops::Bound;
    ///
    /// let mut set = BTreeSet::from([1, 2, 3, 4]);
    ///
    /// let mut cursor = set.lower_bound_mut(Bound::Included(&2));
    /// assert_eq!(cursor.peek_prev(), Some(&1));
    /// assert_eq!(cursor.peek_next(), Some(&2));
    ///
    /// let mut cursor = set.lower_bound_mut(Bound::Excluded(&2));
    /// assert_eq!(cursor.peek_prev(), Some(&2));
    /// assert_eq!(cursor.peek_next(), Some(&3));
    ///
    /// let mut cursor = set.lower_bound_mut(Bound::Unbounded);
    /// assert_eq!(cursor.peek_prev(), None);
    /// assert_eq!(cursor.peek_next(), Some(&1));
    /// ```
    pub fn lower_bound_mut<Q>(&mut self, bound: Bound<&Q>) -> CursorMut<'_, T, A>
    where
        T: Borrow<Q> + Ord,
        Q: ?Sized + Ord,
    {
        CursorMut { inner: self.map.lower_bound_mut(bound) }
    }

    /// Returns a [`CursorMut`] pointing at the gap after the greatest element
    /// smaller than the given bound.
    ///
    /// Passing `Bound::Included(x)` will return a cursor pointing to the
    /// gap after the greatest element smaller than or equal to `x`.
    ///
    /// Passing `Bound::Excluded(x)` will return a cursor pointing to the
    /// gap after the greatest element smaller than `x`.
    ///
    /// Passing `Bound::Unbounded` will return a cursor pointing to the
    /// gap after the greatest element in the set.
    ///
    /// # Examples
    ///
    /// ```
    ///
    /// use pstd::collections::BTreeSet;
    /// use std::ops::Bound;
    ///
    /// let mut set = BTreeSet::from([1, 2, 3, 4]);
    ///
    /// let mut cursor = set.upper_bound_mut(Bound::Included(&3));
    /// assert_eq!(cursor.peek_prev(), Some(&3));
    /// assert_eq!(cursor.peek_next(), Some(&4));
    ///
    /// let mut cursor = set.upper_bound_mut(Bound::Excluded(&3));
    /// assert_eq!(cursor.peek_prev(), Some(&2));
    /// assert_eq!(cursor.peek_next(), Some(&3));
    ///
    /// let mut cursor = set.upper_bound_mut(Bound::Unbounded);
    /// assert_eq!(cursor.peek_prev(), Some(&4));
    /// assert_eq!(cursor.peek_next(), None);
    /// ```
    pub fn upper_bound_mut<Q>(&mut self, bound: Bound<&Q>) -> CursorMut<'_, T, A>
    where
        T: Borrow<Q> + Ord,
        Q: ?Sized + Ord,
    {
        CursorMut { inner: self.map.upper_bound_mut(bound) }
    }


    /// Creates an iterator that visits elements in the specified range in ascending order and
    /// uses a closure to determine if an element should be removed.
    ///
    /// If the closure returns `true`, the element is removed from the set and
    /// yielded. If the closure returns `false`, or panics, the element remains
    /// in the set and will not be yielded.
    ///
    /// If the returned `ExtractIf` is not exhausted, e.g. because it is dropped without iterating
    /// or the iteration short-circuits, then the remaining elements will be retained.
    /// Use `extract_if().for_each(drop)` if you do not need the returned iterator,
    /// or [`retain`] with a negated predicate if you also do not need to restrict the range.
    ///
    /// [`retain`]: BTreeSet::retain
        /// # Examples
    ///
    /// ```
    /// use std::collections::BTreeSet;
    ///
    /// // Splitting a set into even and odd values, reusing the original set:
    /// let mut set: BTreeSet<i32> = (0..8).collect();
    /// let evens: BTreeSet<_> = set.extract_if(.., |v| v % 2 == 0).collect();
    /// let odds = set;
    /// assert_eq!(evens.into_iter().collect::<Vec<_>>(), vec![0, 2, 4, 6]);
    /// assert_eq!(odds.into_iter().collect::<Vec<_>>(), vec![1, 3, 5, 7]);
    ///
    /// // Splitting a set into low and high halves, reusing the original set:
    /// let mut set: BTreeSet<i32> = (0..8).collect();
    /// let low: BTreeSet<_> = set.extract_if(0..4, |_v| true).collect();
    /// let high = set;
    /// assert_eq!(low.into_iter().collect::<Vec<_>>(), [0, 1, 2, 3]);
    /// assert_eq!(high.into_iter().collect::<Vec<_>>(), [4, 5, 6, 7]);
    /// ```
    pub fn extract_if<F>(&mut self, pred: F) -> ExtractIf<'_, T, F, A>
    where
        T: Ord,
        F: FnMut(&T) -> bool,
    {
        let source = self.lower_bound_mut(Bound::Unbounded);
        ExtractIf { source, pred }
    }
    
} // end impl BTreeSet

// start impl for BTreeSet

impl<T: Ord, const N: usize> From<[T; N]> for BTreeSet<T> {
    /// Converts a `[T; N]` into a `BTreeSet<T>`.
    ///
    /// If the array contains any equal values,
    /// all but one will be dropped.
    ///
    /// # Examples
    ///
    /// ```
    /// use pstd::collections::BTreeSet;
    ///
    /// let set1 = BTreeSet::from([1, 2, 3, 4]);
    /// let set2: BTreeSet<_> = [1, 2, 3, 4].into();
    /// assert_eq!(set1, set2);
    /// ```
    fn from(arr: [T; N]) -> Self {
        let mut result = BTreeSet::new();
        for e in arr {
            result.insert(e);
        }
        result
    }
}

impl<T> Default for BTreeSet<T> {
    /// Creates an empty `BTreeSet`.
    fn default() -> BTreeSet<T> {
        BTreeSet::new()
    }
}

impl<T: Hash, A: Tuning> Hash for BTreeSet<T, A> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.map.hash(state)
    }
}

impl<T: PartialEq, A: Tuning> PartialEq for BTreeSet<T, A> {
    fn eq(&self, other: &BTreeSet<T, A>) -> bool {
        self.map.eq(&other.map)
    }
}

impl<T: Eq, A: Tuning> Eq for BTreeSet<T, A> {}

impl<T: PartialOrd, A: Tuning> PartialOrd for BTreeSet<T, A> {
    fn partial_cmp(&self, other: &BTreeSet<T, A>) -> Option<Ordering> {
        self.map.partial_cmp(&other.map)
    }
}

impl<T: Ord, A: Tuning> Ord for BTreeSet<T, A> {
    fn cmp(&self, other: &BTreeSet<T, A>) -> Ordering {
        self.map.cmp(&other.map)
    }
}

impl<T: Clone, A: Tuning> Clone for BTreeSet<T, A> {
    fn clone(&self) -> Self {
        BTreeSet {
            map: self.map.clone(),
        }
    }

    fn clone_from(&mut self, source: &Self) {
        self.map.clone_from(&source.map);
    }
}

impl<T: Debug, A: Tuning> Debug for BTreeSet<T, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_set().entries(self.map.iter()).finish() // ToDo - remove map when iter is done.
    }
}

impl<K: Ord> FromIterator<K> for BTreeSet<K> {
    fn from_iter<T: IntoIterator<Item = K>>(iter: T) -> BTreeSet<K> {
        let mut result = BTreeSet::new();
        for k in iter {
            result.insert(k);
        }
        result
    }
}

impl<T, A: Tuning> IntoIterator for BTreeSet<T, A> {
    type Item = T;
    type IntoIter = IntoIter<T, A>;

    /// Gets an iterator for moving out the `BTreeSet`'s contents in ascending order.
    ///
    /// # Examples
    ///
    /// ```
    /// use pstd::collections::BTreeSet;
    ///
    /// let set = BTreeSet::from([1, 2, 3, 4]);
    ///
    /// let v: Vec<_> = set.into_iter().collect();
    /// assert_eq!(v, [1, 2, 3, 4]);
    /// ```
    fn into_iter(self) -> IntoIter<T, A> {
        IntoIter {
            iter: self.map.into_iter(),
        }
    }
}

impl<'a, T, A: Tuning> IntoIterator for &'a BTreeSet<T, A> {
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;

    fn into_iter(self) -> Iter<'a, T> {
        self.iter()
    }
}

// end impl for BTreeSet

/// An iterator over the items of a `BTreeSet`.
///
/// This `struct` is created by the [`iter`] method on [`BTreeSet`].
/// See its documentation for more.
///
/// [`iter`]: BTreeSet::iter
#[must_use = "iterators are lazy and do nothing unless consumed"]
pub struct Iter<'a, T: 'a> {
    iter: Keys<'a, T, ()>,
}

impl<'a, T> Iterator for Iter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<&'a T> {
        self.iter.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }

    fn last(mut self) -> Option<&'a T> {
        self.next_back()
    }

    fn min(mut self) -> Option<&'a T>
    where
        &'a T: Ord,
    {
        self.next()
    }

    fn max(mut self) -> Option<&'a T>
    where
        &'a T: Ord,
    {
        self.next_back()
    }
}

impl<'a, T> DoubleEndedIterator for Iter<'a, T> {
    fn next_back(&mut self) -> Option<&'a T> {
        self.iter.next_back()
    }
}

impl<T> ExactSizeIterator for Iter<'_, T> {
    fn len(&self) -> usize {
        self.iter.len()
    }
}

impl<T> FusedIterator for Iter<'_, T> {}

impl<T, A: Tuning> Iterator for IntoIter<T, A> {
    type Item = T;

    fn next(&mut self) -> Option<T> {
        self.iter.next().map(|(k, _)| k)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

// end impl for Iter

/// An owning iterator over the items of a `BTreeSet` in ascending order.
///
/// This `struct` is created by the [`into_iter`] method on [`BTreeSet`]
/// (provided by the [`IntoIterator`] trait). See its documentation for more.
///
/// [`into_iter`]: BTreeSet#method.into_iter
#[derive(Debug)]
pub struct IntoIter<T, A: Tuning = DefaultTuning> {
    iter: super::btree_map::IntoIter<T, (), A>,
}

impl<T, A: Tuning> DoubleEndedIterator for IntoIter<T, A> {
    fn next_back(&mut self) -> Option<T> {
        self.iter.next_back().map(|(k, _)| k)
    }
}

impl<T, A: Tuning> ExactSizeIterator for IntoIter<T, A> {
    fn len(&self) -> usize {
        self.iter.len()
    }
}

impl<T, A: Tuning> FusedIterator for IntoIter<T, A> {}

/* Cannot get this to compile.
impl<T, A> Default for IntoIter<T, A>
where
    A: Tuning + Clone,
{
    /// Creates an empty `btree_set::IntoIter`.
    ///
    /// ```
    /// use pstd::collections::btree_set;
    /// let iter: btree_set::IntoIter<u8> = Default::default();
    /// assert_eq!(iter.len(), 0);
    /// ```
    fn default() -> Self {
        IntoIter { iter: Default::default() }
    }
}
*/

// end impl for IntoIter

/// ToDo
pub struct Intersection<'a, T>
{
    _a: Iter<'a, T>,
    _b: Iter<'a, T>,
}

/// A cursor over a `BTreeSet`.
///
/// A `Cursor` is like an iterator, except that it can freely seek back-and-forth.
///
/// Cursors always point to a gap between two elements in the set, and can
/// operate on the two immediately adjacent elements.
///
/// A `Cursor` is created with the [`BTreeSet::lower_bound`] and [`BTreeSet::upper_bound`] methods.
#[derive(Clone)]
pub struct Cursor<'a, K: 'a, A: Tuning = DefaultTuning> {
    inner: super::btree_map::Cursor<'a, K, (), A>,
}

impl<K: Debug> Debug for Cursor<'_, K> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Cursor")
    }
}

impl<'a, K, A: Tuning> Cursor<'a, K, A> {
    /// Advances the cursor to the next gap, returning the element that it
    /// moved over.
    ///
    /// If the cursor is already at the end of the set then `None` is returned
    /// and the cursor is not moved.
    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> Option<&K> {
         self.inner.next().map(|(k, _)| k)
    }

    /// Advances the cursor to the previous gap, returning the element that it
    /// moved over.
    ///
    /// If the cursor is already at the start of the set then `None` is returned
    /// and the cursor is not moved.
    pub fn prev(&mut self) -> Option<&K> {
        self.inner.prev().map(|(k, _)| k)
    }

    /// Returns a reference to next element without moving the cursor.
    ///
    /// If the cursor is at the end of the set then `None` is returned
    pub fn peek_next(&self) -> Option<&K> {
        self.inner.peek_next().map(|(k, _)| k)
    }

    /// Returns a reference to the previous element without moving the cursor.
    ///
    /// If the cursor is at the start of the set then `None` is returned.
    pub fn peek_prev(&self) -> Option<&K> {
        self.inner.peek_prev().map(|(k, _)| k)
    }
}

/// A cursor over a `BTreeSet` with editing operations, and which allows
/// mutating elements.
///
/// A `Cursor` is like an iterator, except that it can freely seek back-and-forth, and can
/// safely mutate the set during iteration. This is because the lifetime of its yielded
/// references is tied to its own lifetime, instead of just the underlying set. This means
/// cursors cannot yield multiple elements at once.
///
/// Cursors always point to a gap between two elements in the set, and can
/// operate on the two immediately adjacent elements.
///
/// A `CursorMutKey` is created from a [`CursorMut`] with the
/// [`CursorMut::with_mutable_key`] method.
///
/// Since this cursor allows mutating elements, you should ensure that the
/// `BTreeSet` invariants are maintained. Specifically:
///
/// * The newly inserted element should be unique in the tree.
/// * All elements in the tree should remain in sorted order.
pub struct CursorMutKey<
    'a,
    T: 'a,
    A: Tuning = DefaultTuning,
> 
{
    inner: super::btree_map::CursorMutKey<'a, T, (), A>,
}

impl<K: Debug, A: Tuning> Debug for CursorMutKey<'_, K, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("CursorMutKey")
    }
}

impl<'a, T: Ord, A: Tuning> CursorMutKey<'a, T, A> {
    /// Advances the cursor to the next gap, returning the  element that it
    /// moved over.
    ///
    /// If the cursor is already at the end of the set then `None` is returned
    /// and the cursor is not moved.
    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> Option<&mut T> {
        self.inner.next().map(|(k, _)| k)
    }

    /// Advances the cursor to the previous gap, returning the element that it
    /// moved over.
    ///
    /// If the cursor is already at the start of the set then `None` is returned
    /// and the cursor is not moved.
    pub fn prev(&mut self) -> Option<&mut T> {
        self.inner.prev().map(|(k, _)| k)
    }

    /// Returns a reference to the next element without moving the cursor.
    ///
    /// If the cursor is at the end of the set then `None` is returned
    pub fn peek_next(&mut self) -> Option<&mut T> {
        self.inner.peek_next().map(|(k, _)| k)
    }

    /// Returns a reference to the previous element without moving the cursor.
    ///
    /// If the cursor is at the start of the set then `None` is returned.
    pub fn peek_prev(&mut self) -> Option<&mut T> {
        self.inner.peek_prev().map(|(k, _)| k)
    }

    /// Inserts a new element into the set in the gap that the
    /// cursor is currently pointing to.
    ///
    /// After the insertion the cursor will be pointing at the gap before the
    /// newly inserted element.
    ///
    /// You should ensure that the `BTreeSet` invariants are maintained.
    /// Specifically:
    ///
    /// * The key of the newly inserted element must be unique in the tree.
    /// * All elements in the tree must remain in sorted order.
    pub fn insert_after_unchecked(&mut self, value: T) {
        self.inner.insert_after_unchecked(value, ())
    }

    /// Inserts a new element into the set in the gap that the
    /// cursor is currently pointing to.
    ///
    /// After the insertion the cursor will be pointing at the gap after the
    /// newly inserted element.
    ///
    /// You should ensure that the `BTreeSet` invariants are maintained.
    /// Specifically:
    ///
    /// * The newly inserted element must be unique in the tree.
    /// * All elements in the tree must remain in sorted order.
    pub fn insert_before_unchecked(&mut self, value: T) {
        self.inner.insert_before_unchecked(value, ())
    }
    
    /// Inserts a new element into the set in the gap that the
    /// cursor is currently pointing to.
    ///
    /// After the insertion the cursor will be pointing at the gap before the
    /// newly inserted element.
    ///
    /// If the inserted element is not greater than the element before the
    /// cursor (if any), or if it not less than the element after the cursor (if
    /// any), then an [`UnorderedKeyError`] is returned since this would
    /// invalidate the [`Ord`] invariant between the elements of the set.
    pub fn insert_after(&mut self, value: T) -> Result<(), UnorderedKeyError> {
        self.inner.insert_after(value, ())
    }

    /// Inserts a new element into the set in the gap that the
    /// cursor is currently pointing to.
    ///
    /// After the insertion the cursor will be pointing at the gap after the
    /// newly inserted element.
    ///
    /// If the inserted element is not greater than the element before the
    /// cursor (if any), or if it not less than the element after the cursor (if
    /// any), then an [`UnorderedKeyError`] is returned since this would
    /// invalidate the [`Ord`] invariant between the elements of the set.
    pub fn insert_before(&mut self, value: T) -> Result<(), UnorderedKeyError> {
        self.inner.insert_before(value, ())
    }

    /// Removes the next element from the `BTreeSet`.
    ///
    /// The element that was removed is returned. The cursor position is
    /// unchanged (before the removed element).
    pub fn remove_next(&mut self) -> Option<T> {
        self.inner.remove_next().map(|(k, _)| k)
    }

    /// Removes the preceding element from the `BTreeSet`.
    ///
    /// The element that was removed is returned. The cursor position is
    /// unchanged (after the removed element).
    pub fn remove_prev(&mut self) -> Option<T> {
        self.inner.remove_prev().map(|(k, _)| k)
    }
}

/// A cursor over a `BTreeSet` with editing operations.
///
/// A `Cursor` is like an iterator, except that it can freely seek back-and-forth, and can
/// safely mutate the set during iteration. This is because the lifetime of its yielded
/// references is tied to its own lifetime, instead of just the underlying map. This means
/// cursors cannot yield multiple elements at once.
///
/// Cursors always point to a gap between two elements in the set, and can
/// operate on the two immediately adjacent elements.
///
/// A `CursorMut` is created with the [`BTreeSet::lower_bound_mut`] and [`BTreeSet::upper_bound_mut`]
/// methods.
pub struct CursorMut<
    'a,
    T: 'a,
    A: Tuning = DefaultTuning,
> 
{
    inner: super::btree_map::CursorMut<'a, T, (), A>,
}

impl<K: Debug, A: Tuning> Debug for CursorMut<'_, K, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("CursorMutKey")
    }
}

impl<'a, T: Ord, A: Tuning> CursorMut<'a, T, A> {
    /// Advances the cursor to the next gap, returning the  element that it
    /// moved over.
    ///
    /// If the cursor is already at the end of the set then `None` is returned
    /// and the cursor is not moved.
    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> Option<&T> {
        self.inner.next().map(|(k, _)| k)
    }

    /// Advances the cursor to the previous gap, returning the element that it
    /// moved over.
    ///
    /// If the cursor is already at the start of the set then `None` is returned
    /// and the cursor is not moved.
    pub fn prev(&mut self) -> Option<&T> {
        self.inner.prev().map(|(k, _)| k)
    }

    /// Returns a reference to the next element without moving the cursor.
    ///
    /// If the cursor is at the end of the set then `None` is returned
    pub fn peek_next(&mut self) -> Option<&T> {
        self.inner.peek_next().map(|(k, _)| k)
    }

    /// Returns a reference to the previous element without moving the cursor.
    ///
    /// If the cursor is at the start of the set then `None` is returned.
    pub fn peek_prev(&mut self) -> Option<&T> {
        self.inner.peek_prev().map(|(k, _)| k)
    }

    /// Inserts a new element into the set in the gap that the
    /// cursor is currently pointing to.
    ///
    /// After the insertion the cursor will be pointing at the gap before the
    /// newly inserted element.
    ///
    /// You should ensure that the `BTreeSet` invariants are maintained.
    /// Specifically:
    ///
    /// * The key of the newly inserted element must be unique in the tree.
    /// * All elements in the tree must remain in sorted order.
    pub fn insert_after_unchecked(&mut self, value: T) {
        self.inner.insert_after_unchecked(value, ())
    }

    /// Inserts a new element into the set in the gap that the
    /// cursor is currently pointing to.
    ///
    /// After the insertion the cursor will be pointing at the gap after the
    /// newly inserted element.
    ///
    /// You should ensure that the `BTreeSet` invariants are maintained.
    /// Specifically:
    ///
    /// * The newly inserted element must be unique in the tree.
    /// * All elements in the tree must remain in sorted order.
    pub fn insert_before_unchecked(&mut self, value: T) {
        self.inner.insert_before_unchecked(value, ())
    }
    
    /// Inserts a new element into the set in the gap that the
    /// cursor is currently pointing to.
    ///
    /// After the insertion the cursor will be pointing at the gap before the
    /// newly inserted element.
    ///
    /// If the inserted element is not greater than the element before the
    /// cursor (if any), or if it not less than the element after the cursor (if
    /// any), then an [`UnorderedKeyError`] is returned since this would
    /// invalidate the [`Ord`] invariant between the elements of the set.
    pub fn insert_after(&mut self, value: T) -> Result<(), UnorderedKeyError> {
        self.inner.insert_after(value, ())
    }

    /// Inserts a new element into the set in the gap that the
    /// cursor is currently pointing to.
    ///
    /// After the insertion the cursor will be pointing at the gap after the
    /// newly inserted element.
    ///
    /// If the inserted element is not greater than the element before the
    /// cursor (if any), or if it not less than the element after the cursor (if
    /// any), then an [`UnorderedKeyError`] is returned since this would
    /// invalidate the [`Ord`] invariant between the elements of the set.
    pub fn insert_before(&mut self, value: T) -> Result<(), UnorderedKeyError> {
        self.inner.insert_before(value, ())
    }

    /// Removes the next element from the `BTreeSet`.
    ///
    /// The element that was removed is returned. The cursor position is
    /// unchanged (before the removed element).
    pub fn remove_next(&mut self) -> Option<T> {
        self.inner.remove_next().map(|(k, _)| k)
    }

    /// Removes the preceding element from the `BTreeSet`.
    ///
    /// The element that was removed is returned. The cursor position is
    /// unchanged (after the removed element).
    pub fn remove_prev(&mut self) -> Option<T> {
        self.inner.remove_prev().map(|(k, _)| k)
    }

    /// Converts the cursor into a [`CursorMutKey`], which allows mutating
    /// elements in the tree.
    ///
    /// Since this cursor allows mutating elements, you should ensure that the
    /// `BTreeSet` invariants are maintained. Specifically:
    ///
    /// * The newly inserted element must be unique in the tree.
    /// * All elements in the tree must remain in sorted order.
    pub fn with_mutable_key(self) -> CursorMutKey<'a, T, A> {
        CursorMutKey { inner: self.inner.with_mutable_key() }
    }
}

/// Iterator returned by [`BTreeSet::extract_if`].
// #[derive(Debug)]
pub struct ExtractIf<'a, K, F, A: Tuning = DefaultTuning>
where
    F: FnMut(&K) -> bool,
{
    source: CursorMut<'a, K, A>,
    pred: F,
}

impl<K, A: Tuning, F> fmt::Debug for ExtractIf<'_, K, F, A>
where
    K: Ord + fmt::Debug,
    F: FnMut(&K) -> bool,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("ExtractIf")
            // .field(&self.source.peek_next())
            .finish_non_exhaustive()
    }
}

impl<'a, K, A: Tuning, F> Iterator for ExtractIf<'a, K, F, A>
where
    K: Ord,
    F: FnMut(&K) -> bool,
{
    type Item = K;
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let k = self.source.peek_next()?;
            if (self.pred)(k) {
                return self.source.remove_next();
            }
            self.source.next();
        }
    }
}
impl<'a, K, A: Tuning, F> FusedIterator for ExtractIf<'a, K, F, A> where
    K: Ord,
    F: FnMut(&K) -> bool
{
}

