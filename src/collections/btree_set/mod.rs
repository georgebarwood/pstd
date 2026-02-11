//! [`BTreeSet`] similar to [`std::collections::BTreeSet`].

use crate::alloc::Allocator;
use std::borrow::Borrow;
use std::cmp::{Ordering, max, min};
use std::fmt::{self, Debug};
use std::hash::{Hash, Hasher};
use std::iter::{FusedIterator, Peekable};
use std::ops::{BitAnd, BitOr, BitXor, Bound, RangeBounds, Sub};

use super::merge_iter::MergeIterInner;
use crate::collections::btree_map as map;

pub use map::{CustomTuning, DefaultTuning, Tuning, UnorderedKeyError};

mod entry;
pub use entry::{Entry, OccupiedEntry, VacantEntry};

const ITER_PERFORMANCE_TIPPING_SIZE_DIFF: usize = 16;

/// An ordered set based on a B-Tree similar to [`std::collections::BTreeSet`].
///
/// # Guide to methods
///
/// Set Creation: [`new`], [`new_in`], [`with_tuning`]
///
/// Properties: [`len`], [`is_empty`], [`contains`],
/// [`is_subset`], [`is_superset`], [`is_disjoint`]
///
/// Insertion: [`insert`], [`get_or_insert`], [`get_or_insert_with`], [`entry`]
///
/// Retrieve: [`get`], [`first`], [`last`]
///
/// Removal: [`remove`], [`take`], [`pop_first`], [`pop_last`]
///
/// Bulk: [`append`], [`split_off`], [`retain`], [`clear`]
///
/// Iterators: [`iter`], [`range`], [`extract_if`], [`union`], [`intersection`]
/// , [`difference`], [`symmetric_difference`]
///
/// Cursors: [`lower_bound`], [`upper_bound`], [`lower_bound_mut`], [`upper_bound_mut`]
///
/// [`new`]: BTreeSet::new
/// [`new_in`]: BTreeSet::new_in
/// [`with_tuning`]: BTreeSet::with_tuning
/// [`len`]: BTreeSet::len
/// [`is_empty`]: BTreeSet::is_empty
/// [`contains`]: BTreeSet::contains
/// [`is_subset`]: BTreeSet::is_subset
/// [`is_superset`]: BTreeSet::is_superset
/// [`is_disjoint`]: BTreeSet::is_disjoint
/// [`insert`]: BTreeSet::insert
/// [`get_or_insert`]: BTreeSet::get_or_insert
/// [`get_or_insert_with`]: BTreeSet::get_or_insert_with
/// [`entry`]: BTreeSet::entry
/// [`get`]: BTreeSet::get
/// [`first`]: BTreeSet::first
/// [`last`]: BTreeSet::last
/// [`remove`]: BTreeSet::remove
/// [`take`]: BTreeSet::take
/// [`pop_first`]: BTreeSet::pop_first
/// [`pop_last`]: BTreeSet::pop_last
/// [`append`]: BTreeSet::append
/// [`split_off`]: BTreeSet::split_off
/// [`retain`]: BTreeSet::retain
/// [`clear`]: BTreeSet::clear
/// [`iter`]: BTreeSet::iter
/// [`range`]: BTreeSet::range
/// [`extract_if`]: BTreeSet::extract_if
/// [`union`]: BTreeSet::union
/// [`intersection`]: BTreeSet::intersection
/// [`difference`]: BTreeSet::difference
/// [`symmetric_difference`]: BTreeSet::symmetric_difference
/// [`lower_bound`]: BTreeSet::lower_bound
/// [`upper_bound`]: BTreeSet::upper_bound
/// [`lower_bound_mut`]: BTreeSet::lower_bound_mut
/// [`upper_bound_mut`]: BTreeSet::upper_bound_mut
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
    map: map::BTreeMap<T, (), A>,
}

impl<T> BTreeSet<T> {
    /// Returns a new, empty `BTreeSet`.
    ///
    /// # Example
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
            map: map::BTreeMap::new(),
        }
    }

    /// Returns new set with specified allocator
    ///
    /// # Example
    ///
    /// ```
    /// # #![allow(unused_mut)]
    /// use pstd::collections::BTreeSet;
    /// use pstd::alloc::Global;
    ///
    /// let mut set: BTreeSet<i32> = BTreeSet::new_in(Global);
    /// ```
    #[must_use]
    pub const fn new_in<AL>(a: AL) -> BTreeSet<T, CustomTuning<AL>>
    where
        AL: Allocator + Clone,
    {
        BTreeSet {
            map: map::BTreeMap::with_tuning(CustomTuning::new_in_def(a)),
        }
    }
}

impl<T, A: Tuning> BTreeSet<T, A> {
    /// Returns a new, empty set with specified tuning.
    ///
    /// # Example
    ///
    /// ```
    ///     use pstd::collections::BTreeSet;
    ///     use pstd::collections::btree_set::DefaultTuning;
    ///     let mut set = BTreeSet::with_tuning(DefaultTuning::new(8,2));
    ///     set.insert("England");
    ///     set.insert("France");
    ///     assert!(set.contains("England"));
    /// ```
    #[must_use]
    pub const fn with_tuning(atune: A) -> Self {
        Self {
            map: map::BTreeMap::with_tuning(atune),
        }
    }

    /// Returns number of elements in the set
    ///
    /// # Example
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
    /// # Example
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

    /// Clears the set, removing all elements.
    ///
    /// # Example
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
    /// # Example
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
    /// # Example
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
    /// # Example
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
    /// # Example
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
    /// # Example
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
    /// # Example
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

    /// Inserts the given `value` into the set if it is not present, then
    /// returns a reference to the value in the set.
    ///
    /// # Example
    ///
    /// ```
    ///
    /// use pstd::collections::BTreeSet;
    ///
    /// let mut set = BTreeSet::from([1, 2, 3]);
    /// assert_eq!(set.len(), 3);
    /// assert_eq!(set.get_or_insert(2), &2);
    /// assert_eq!(set.get_or_insert(100), &100);
    /// assert_eq!(set.len(), 4); // 100 was inserted
    /// ```
    #[inline]
    pub fn get_or_insert(&mut self, value: T) -> &T
    where
        T: Ord,
    {
        self.map.entry(value).insert_entry(()).into_key()
    }

    /// Inserts a value computed from `f` into the set if the given `value` is
    /// not present, then returns a reference to the value in the set.
    ///
    /// # Example
    ///
    /// ```
    ///
    /// use pstd::collections::BTreeSet;
    ///
    /// let mut set: BTreeSet<String> = ["cat", "dog", "horse"]
    ///     .iter().map(|&pet| pet.to_owned()).collect();
    ///
    /// assert_eq!(set.len(), 3);
    /// for &pet in &["cat", "dog", "fish"] {
    ///     let value = set.get_or_insert_with(pet, str::to_owned);
    ///     assert_eq!(value, pet);
    /// }
    /// assert_eq!(set.len(), 4); // a new "fish" was inserted
    /// ```
    #[inline]
    pub fn get_or_insert_with<Q, F>(&mut self, value: &Q, f: F) -> &T
    where
        T: Borrow<Q> + Ord,
        Q: ?Sized + Ord,
        F: FnOnce(&Q) -> T,
    {
        // Simple method
        // let nv = f(value);
        // self.get_or_insert(nv)
        let mut cursor = self.lower_bound_mut(Bound::Included(value));
        if let Some(vr) = cursor.peek_next()
            && vr.borrow() == value
        {
        } else {
            let v = f(value);
            cursor.insert_after_unchecked(v);
        };
        unsafe { cursor.into() }
    }

    /// Retains only the elements specified by the predicate.
    ///
    /// In other words, remove all elements `e` for which `f(&e)` returns `false`.
    /// The elements are visited in ascending order.
    ///
    /// # Example
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

    /// Returns a reference to the first element in the set, if any.
    /// This element is always the minimum of all elements in the set.
    ///
    /// # Example
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
    /// # Example
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
    /// # Example
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
    /// # Example
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
    /// # Example
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

    /// Splits the collection into two at the value. Returns a new collection
    /// with all elements greater than or equal to the value.
    ///
    /// # Example
    ///
    /// Basic usage:
    ///
    /// ```
    /// use pstd::collections::BTreeSet;
    ///
    /// let mut a = BTreeSet::new();
    /// a.insert(1);
    /// a.insert(2);
    /// a.insert(3);
    /// a.insert(17);
    /// a.insert(41);
    ///
    /// let b = a.split_off(&3);
    ///
    /// assert_eq!(a.len(), 2);
    /// assert_eq!(b.len(), 3);
    ///
    /// assert!(a.contains(&1));
    /// assert!(a.contains(&2));
    ///
    /// assert!(b.contains(&3));
    /// assert!(b.contains(&17));
    /// assert!(b.contains(&41));
    /// ```
    pub fn split_off<Q: ?Sized + Ord>(&mut self, value: &Q) -> Self
    where
        T: Borrow<Q> + Ord,
        A: Clone,
    {
        BTreeSet {
            map: self.map.split_off(value),
        }
    }

    /// Moves all elements from `other` into `self`, leaving `other` empty.
    ///
    /// # Example
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
    /// # Example
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
    #[inline]
    pub fn iter(&self) -> Iter<'_, T> {
        Iter {
            iter: self.map.keys(),
        }
    }

    /// Constructs a double-ended iterator over a sub-range of elements in the set.
    /// The simplest way is to use the range syntax `min..max`, thus `range(min..max)` will
    /// yield elements from min (inclusive) to max (exclusive).
    /// The range may also be entered as `(Bound<T>, Bound<T>)`, so for example
    /// `range((Excluded(4), Included(10)))` will yield a left-exclusive, right-inclusive
    /// range from 4 to 10.
    ///
    /// # Panics
    ///
    /// Panics if range `start > end`.
    /// Panics if range `start == end` and both bounds are `Excluded`.
    ///
    /// # Example
    ///
    /// ```
    /// use pstd::collections::BTreeSet;
    /// use std::ops::Bound::Included;
    ///
    /// let mut set = BTreeSet::new();
    /// set.insert(3);
    /// set.insert(5);
    /// set.insert(8);
    /// for &elem in set.range((Included(&4), Included(&8))) {
    ///     println!("{elem}");
    /// }
    /// assert_eq!(Some(&5), set.range(4..).next());
    /// ```
    pub fn range<K, R>(&self, range: R) -> Range<'_, T>
    where
        K: ?Sized + Ord,
        T: Borrow<K> + Ord,
        R: RangeBounds<K>,
    {
        Range {
            iter: self.map.range(range),
        }
    }

    /// Visits the elements representing the intersection,
    /// i.e., the elements that are both in `self` and `other`,
    /// in ascending order.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::BTreeSet;
    ///
    /// let mut a = BTreeSet::new();
    /// a.insert(1);
    /// a.insert(2);
    ///
    /// let mut b = BTreeSet::new();
    /// b.insert(2);
    /// b.insert(3);
    ///
    /// let intersection: Vec<_> = a.intersection(&b).cloned().collect();
    /// assert_eq!(intersection, [2]);
    /// ```
    pub fn intersection<'a>(&'a self, other: &'a BTreeSet<T, A>) -> Intersection<'a, T, A>
    where
        T: Ord,
    {
        if let Some(self_min) = self.first()
            && let Some(self_max) = self.last()
            && let Some(other_min) = other.first()
            && let Some(other_max) = other.last()
        {
            Intersection {
                inner: match (self_min.cmp(other_max), self_max.cmp(other_min)) {
                    (Ordering::Greater, _) | (_, Ordering::Less) => IntersectionInner::Answer(None),
                    (Ordering::Equal, _) => IntersectionInner::Answer(Some(self_min)),
                    (_, Ordering::Equal) => IntersectionInner::Answer(Some(self_max)),
                    _ if self.len() <= other.len() / ITER_PERFORMANCE_TIPPING_SIZE_DIFF => {
                        IntersectionInner::Search {
                            small_iter: self.iter(),
                            large_set: other,
                        }
                    }
                    _ if other.len() <= self.len() / ITER_PERFORMANCE_TIPPING_SIZE_DIFF => {
                        IntersectionInner::Search {
                            small_iter: other.iter(),
                            large_set: self,
                        }
                    }
                    _ => IntersectionInner::Stitch {
                        a: self.iter(),
                        b: other.iter(),
                    },
                },
            }
        } else {
            Intersection {
                inner: IntersectionInner::Answer(None),
            }
        }
    }

    /// Visits the elements representing the union,
    /// i.e., all the elements in `self` or `other`, without duplicates,
    /// in ascending order.
    ///
    /// # Example
    ///
    /// ```
    /// use pstd::collections::BTreeSet;
    ///
    /// let mut a = BTreeSet::new();
    /// a.insert(1);
    ///
    /// let mut b = BTreeSet::new();
    /// b.insert(2);
    ///
    /// let union: Vec<_> = a.union(&b).cloned().collect();
    /// assert_eq!(union, [1, 2]);
    /// ```
    pub fn union<'a>(&'a self, other: &'a BTreeSet<T, A>) -> Union<'a, T>
    where
        T: Ord,
    {
        Union(MergeIterInner::new(self.iter(), other.iter()))
    }

    /// Visits the elements representing the difference,
    /// i.e., the elements that are in `self` but not in `other`,
    /// in ascending order.
    ///
    /// # Example
    ///
    /// ```
    /// use pstd::collections::BTreeSet;
    ///
    /// let mut a = BTreeSet::new();
    /// a.insert(1);
    /// a.insert(2);
    ///
    /// let mut b = BTreeSet::new();
    /// b.insert(2);
    /// b.insert(3);
    ///
    /// let diff: Vec<_> = a.difference(&b).cloned().collect();
    /// assert_eq!(diff, [1]);
    /// ```
    pub fn difference<'a>(&'a self, other: &'a BTreeSet<T, A>) -> Difference<'a, T, A>
    where
        T: Ord,
    {
        if let Some(self_min) = self.first()
            && let Some(self_max) = self.last()
            && let Some(other_min) = other.first()
            && let Some(other_max) = other.last()
        {
            Difference {
                inner: match (self_min.cmp(other_max), self_max.cmp(other_min)) {
                    (Ordering::Greater, _) | (_, Ordering::Less) => {
                        DifferenceInner::Iterate(self.iter())
                    }
                    (Ordering::Equal, _) => {
                        let mut self_iter = self.iter();
                        self_iter.next();
                        DifferenceInner::Iterate(self_iter)
                    }
                    (_, Ordering::Equal) => {
                        let mut self_iter = self.iter();
                        self_iter.next_back();
                        DifferenceInner::Iterate(self_iter)
                    }
                    _ if self.len() <= other.len() / ITER_PERFORMANCE_TIPPING_SIZE_DIFF => {
                        DifferenceInner::Search {
                            self_iter: self.iter(),
                            other_set: other,
                        }
                    }
                    _ => DifferenceInner::Stitch {
                        self_iter: self.iter(),
                        other_iter: other.iter().peekable(),
                    },
                },
            }
        } else {
            Difference {
                inner: DifferenceInner::Iterate(self.iter()),
            }
        }
    }

    /// Visits the elements representing the symmetric difference,
    /// i.e., the elements that are in `self` or in `other` but not in both,
    /// in ascending order.
    ///
    /// # Example
    ///
    /// ```
    /// use pstd::collections::BTreeSet;
    ///
    /// let mut a = BTreeSet::new();
    /// a.insert(1);
    /// a.insert(2);
    ///
    /// let mut b = BTreeSet::new();
    /// b.insert(2);
    /// b.insert(3);
    ///
    /// let sym_diff: Vec<_> = a.symmetric_difference(&b).cloned().collect();
    /// assert_eq!(sym_diff, [1, 3]);
    /// ```
    pub fn symmetric_difference<'a>(
        &'a self,
        other: &'a BTreeSet<T, A>,
    ) -> SymmetricDifference<'a, T>
    where
        T: Ord,
    {
        SymmetricDifference(MergeIterInner::new(self.iter(), other.iter()))
    }

    /// Returns `true` if `self` has no elements in common with `other`.
    /// This is equivalent to checking for an empty intersection.
    ///
    /// # Example
    ///
    /// ```
    /// use pstd::collections::BTreeSet;
    ///
    /// let a = BTreeSet::from([1, 2, 3]);
    /// let mut b = BTreeSet::new();
    ///
    /// assert_eq!(a.is_disjoint(&b), true);
    /// b.insert(4);
    /// assert_eq!(a.is_disjoint(&b), true);
    /// b.insert(1);
    /// assert_eq!(a.is_disjoint(&b), false);
    /// ```
    #[must_use]
    pub fn is_disjoint(&self, other: &BTreeSet<T, A>) -> bool
    where
        T: Ord,
    {
        self.intersection(other).next().is_none()
    }

    /// Returns `true` if the set is a superset of another,
    /// i.e., `self` contains at least all the elements in `other`.
    ///
    /// # Example
    ///
    /// ```
    /// use pstd::collections::BTreeSet;
    ///
    /// let sub = BTreeSet::from([1, 2]);
    /// let mut set = BTreeSet::new();
    ///
    /// assert_eq!(set.is_superset(&sub), false);
    ///
    /// set.insert(0);
    /// set.insert(1);
    /// assert_eq!(set.is_superset(&sub), false);
    ///
    /// set.insert(2);
    /// assert_eq!(set.is_superset(&sub), true);
    /// ```
    #[must_use]
    pub fn is_superset(&self, other: &BTreeSet<T, A>) -> bool
    where
        T: Ord,
    {
        other.is_subset(self)
    }

    /// Returns `true` if the set is a subset of another,
    /// i.e., `other` contains at least all the elements in `self`.
    ///
    /// # Example
    ///
    /// ```
    /// use pstd::collections::BTreeSet;
    ///
    /// let sup = BTreeSet::from([1, 2, 3]);
    /// let mut set = BTreeSet::new();
    ///
    /// assert_eq!(set.is_subset(&sup), true);
    /// set.insert(2);
    /// assert_eq!(set.is_subset(&sup), true);
    /// set.insert(4);
    /// assert_eq!(set.is_subset(&sup), false);
    /// ```
    #[must_use]
    pub fn is_subset(&self, other: &BTreeSet<T, A>) -> bool
    where
        T: Ord,
    {
        // Same result as self.difference(other).next().is_none()
        // but the code below is faster (hugely in some cases).
        if self.len() > other.len() {
            return false; // self has more elements than other
        }
        let (Some(self_min), Some(self_max)) = (self.first(), self.last()) else {
            return true; // self is empty
        };
        let (Some(other_min), Some(other_max)) = (other.first(), other.last()) else {
            return false; // other is empty
        };
        let mut self_iter = self.iter();
        match self_min.cmp(other_min) {
            Ordering::Less => return false, // other does not contain self_min
            Ordering::Equal => {
                self_iter.next(); // self_min is contained in other, so remove it from consideration
                // other_min is now not in self_iter (used below)
            }
            Ordering::Greater => {} // other_min is not in self_iter (used below)
        };

        match self_max.cmp(other_max) {
            Ordering::Greater => return false, // other does not contain self_max
            Ordering::Equal => {
                self_iter.next_back(); // self_max is contained in other, so remove it from consideration
                // other_max is now not in self_iter (used below)
            }
            Ordering::Less => {} // other_max is not in self_iter (used below)
        };
        if self_iter.len() <= other.len() / ITER_PERFORMANCE_TIPPING_SIZE_DIFF {
            self_iter.all(|e| other.contains(e))
        } else {
            let mut other_iter = other.iter();
            {
                // remove other_min and other_max as they are not in self_iter (see above)
                other_iter.next();
                other_iter.next_back();
            }
            // custom `self_iter.all(|e| other.contains(e))`
            self_iter.all(|self1| {
                for other1 in other_iter.by_ref() {
                    match other1.cmp(self1) {
                        // happens up to `ITER_PERFORMANCE_TIPPING_SIZE_DIFF * self.len() - 1` times
                        Ordering::Less => continue, // skip over elements that are smaller
                        // happens `self.len()` times
                        Ordering::Equal => return true, // self1 is in other
                        // happens only once
                        Ordering::Greater => return false, // self1 is not in other
                    }
                }
                false
            })
        }
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
    /// # Example
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
        Cursor {
            inner: self.map.lower_bound(bound),
        }
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
    /// # Example
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
        Cursor {
            inner: self.map.upper_bound(bound),
        }
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
    /// # Example
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
        CursorMut {
            inner: self.map.lower_bound_mut(bound),
        }
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
    /// # Example
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
        CursorMut {
            inner: self.map.upper_bound_mut(bound),
        }
    }

    /// Create set from specified iterator and tuning.
    fn from_sorted_iter<I: Iterator<Item = T>>(iter: I, tuning: A) -> BTreeSet<T, A>
    where
        T: Ord,
    {
        let mut set = BTreeSet::with_tuning(tuning);
        let save = set.map.set_seq(true);
        {
            let mut cursor = set.lower_bound_mut(Bound::Unbounded);
            for e in iter {
                cursor.insert_before_unchecked(e)
            }
        }
        set.map.set_seq(save);
        set
    }

    /// Gets the given value's corresponding entry in the set for in-place manipulation.
    ///
    /// # Examples
    ///
    /// ```
    ///
    /// use pstd::collections::BTreeSet;
    /// use pstd::collections::btree_set::Entry::*;
    ///
    /// let mut singles = BTreeSet::new();
    /// let mut dupes = BTreeSet::new();
    ///
    /// for ch in "a short treatise on fungi".chars() {
    ///     if let Vacant(dupe_entry) = dupes.entry(ch) {
    ///         // We haven't already seen a duplicate, so
    ///         // check if we've at least seen it once.
    ///         match singles.entry(ch) {
    ///             Vacant(single_entry) => {
    ///                 // We found a new character for the first time.
    ///                 single_entry.insert()
    ///             }
    ///             Occupied(single_entry) => {
    ///                 // We've already seen this once, "move" it to dupes.
    ///                 single_entry.remove();
    ///                 dupe_entry.insert();
    ///             }
    ///         }
    ///     }
    /// }
    ///
    /// assert!(!singles.contains(&'t') && dupes.contains(&'t'));
    /// assert!(singles.contains(&'u') && !dupes.contains(&'u'));
    /// assert!(!singles.contains(&'v') && !dupes.contains(&'v'));
    /// ```
    #[inline]
    pub fn entry(&mut self, value: T) -> Entry<'_, T, A>
    where
        T: Ord,
    {
        match self.map.entry(value) {
            map::Entry::Occupied(entry) => Entry::Occupied(OccupiedEntry { inner: entry }),
            map::Entry::Vacant(entry) => Entry::Vacant(VacantEntry { inner: entry }),
        }
    }
} // end impl BTreeSet

// start impl for BTreeSet

impl<T: Ord, const N: usize> From<[T; N]> for BTreeSet<T> {
    /// Converts a `[T; N]` into a `BTreeSet<T>`.
    ///
    /// If the array contains any equal values,
    /// all but one will be dropped.
    ///
    /// # Example
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
        f.debug_set().entries(self.iter()).finish()
    }
}

impl<T: Ord> FromIterator<T> for BTreeSet<T> {
    fn from_iter<X: IntoIterator<Item = T>>(iter: X) -> BTreeSet<T> {
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
    /// # Example
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

    #[inline]
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
    iter: map::Keys<'a, T, ()>,
}

impl<T: fmt::Debug> fmt::Debug for Iter<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Iter").field(&self.iter).finish()
    }
}

impl<T> Clone for Iter<'_, T> {
    fn clone(&self) -> Self {
        Iter {
            iter: self.iter.clone(),
        }
    }
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
    iter: map::IntoIter<T, (), A>,
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
    inner: map::Cursor<'a, K, (), A>,
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
pub struct CursorMutKey<'a, T: 'a, A: Tuning = DefaultTuning> {
    inner: map::CursorMutKey<'a, T, (), A>,
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
pub struct CursorMut<'a, T: 'a, A: Tuning = DefaultTuning> {
    inner: map::CursorMut<'a, T, (), A>,
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
        CursorMutKey {
            inner: self.inner.with_mutable_key(),
        }
    }

    /// Convert into reference to current (next) key.
    unsafe fn into(self) -> &'a T {
        unsafe { self.inner.into_mut_key() }
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
impl<'a, K, A: Tuning, F> FusedIterator for ExtractIf<'a, K, F, A>
where
    K: Ord,
    F: FnMut(&K) -> bool,
{
}

/// A lazy iterator producing elements in the intersection of `BTreeSet`s.
///
/// This `struct` is created by the [`intersection`] method on [`BTreeSet`].
/// See its documentation for more.
///
/// [`intersection`]: BTreeSet::intersection
#[must_use = "this returns the intersection as an iterator, \
              without modifying either input set"]
pub struct Intersection<'a, T: 'a, A: Tuning = DefaultTuning> {
    inner: IntersectionInner<'a, T, A>,
}

enum IntersectionInner<'a, T: 'a, A: Tuning = DefaultTuning> {
    Stitch {
        // iterate similarly sized sets jointly, spotting matches along the way
        a: Iter<'a, T>,
        b: Iter<'a, T>,
    },
    Search {
        // iterate a small set, look up in the large set
        small_iter: Iter<'a, T>,
        large_set: &'a BTreeSet<T, A>,
    },
    Answer(Option<&'a T>), // return a specific element or emptiness
}

impl<'a, T: Ord, A: Tuning> Iterator for Intersection<'a, T, A> {
    type Item = &'a T;

    fn next(&mut self) -> Option<&'a T> {
        match &mut self.inner {
            IntersectionInner::Stitch { a, b } => {
                let mut a_next = a.next()?;
                let mut b_next = b.next()?;
                loop {
                    match a_next.cmp(b_next) {
                        Ordering::Less => a_next = a.next()?,
                        Ordering::Greater => b_next = b.next()?,
                        Ordering::Equal => return Some(a_next),
                    }
                }
            }
            IntersectionInner::Search {
                small_iter,
                large_set,
            } => loop {
                let small_next = small_iter.next()?;
                if large_set.contains(small_next) {
                    return Some(small_next);
                }
            },
            IntersectionInner::Answer(answer) => answer.take(),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        match &self.inner {
            IntersectionInner::Stitch { a, b } => (0, Some(min(a.len(), b.len()))),
            IntersectionInner::Search { small_iter, .. } => (0, Some(small_iter.len())),
            IntersectionInner::Answer(None) => (0, Some(0)),
            IntersectionInner::Answer(Some(_)) => (1, Some(1)),
        }
    }

    fn min(mut self) -> Option<&'a T> {
        self.next()
    }
}

/// An iterator over a sub-range of items in a `BTreeSet`.
///
/// This `struct` is created by the [`range`] method on [`BTreeSet`].
/// See its documentation for more.
///
/// [`range`]: BTreeSet::range
#[must_use = "iterators are lazy and do nothing unless consumed"]
#[derive(Debug)]
pub struct Range<'a, T: 'a> {
    iter: map::Range<'a, T, ()>,
}

impl<T> Clone for Range<'_, T> {
    fn clone(&self) -> Self {
        Range {
            iter: self.iter.clone(),
        }
    }
}

impl<'a, T> Iterator for Range<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<&'a T> {
        self.iter.next().map(|(k, _)| k)
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

impl<'a, T> DoubleEndedIterator for Range<'a, T> {
    fn next_back(&mut self) -> Option<&'a T> {
        self.iter.next_back().map(|(k, _)| k)
    }
}

impl<T> FusedIterator for Range<'_, T> {}

impl<T> Default for Range<'_, T> {
    /// Creates an empty `btree_set::Range`.
    ///
    /// ```
    /// # use std::collections::btree_set;
    /// let iter: btree_set::Range<'_, u8> = Default::default();
    /// assert_eq!(iter.count(), 0);
    /// ```
    fn default() -> Self {
        Range {
            iter: Default::default(),
        }
    }
}

/// A lazy iterator producing elements in the symmetric difference of `BTreeSet`s.
///
/// This `struct` is created by the [`symmetric_difference`] method on
/// [`BTreeSet`]. See its documentation for more.
///
/// [`symmetric_difference`]: BTreeSet::symmetric_difference
#[must_use = "this returns the difference as an iterator, \
              without modifying either input set"]
pub struct SymmetricDifference<'a, T: 'a>(MergeIterInner<Iter<'a, T>>);

impl<T: fmt::Debug> fmt::Debug for SymmetricDifference<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("SymmetricDifference").field(&self.0).finish()
    }
}

impl<T> Clone for SymmetricDifference<'_, T> {
    fn clone(&self) -> Self {
        SymmetricDifference(self.0.clone())
    }
}

impl<'a, T: Ord> Iterator for SymmetricDifference<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<&'a T> {
        loop {
            let (a_next, b_next) = self.0.nexts(Self::Item::cmp);
            if a_next.and(b_next).is_none() {
                return a_next.or(b_next);
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let (a_len, b_len) = self.0.lens();
        // No checked_add, because even if a and b refer to the same set,
        // and T is a zero-sized type, the storage overhead of sets limits
        // the number of elements to less than half the range of usize.
        (0, Some(a_len + b_len))
    }

    fn min(mut self) -> Option<&'a T> {
        self.next()
    }
}

impl<T: Ord> FusedIterator for SymmetricDifference<'_, T> {}

/// A lazy iterator producing elements in the union of `BTreeSet`s.
///
/// This `struct` is created by the [`union`] method on [`BTreeSet`].
/// See its documentation for more.
///
/// [`union`]: BTreeSet::union
#[must_use = "this returns the union as an iterator, \
              without modifying either input set"]
pub struct Union<'a, T: 'a>(MergeIterInner<Iter<'a, T>>);

impl<T: fmt::Debug> fmt::Debug for Union<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Union").field(&self.0).finish()
    }
}

impl<T> Clone for Union<'_, T> {
    fn clone(&self) -> Self {
        Union(self.0.clone())
    }
}

impl<'a, T: Ord> Iterator for Union<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<&'a T> {
        let (a_next, b_next) = self.0.nexts(Self::Item::cmp);
        a_next.or(b_next)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let (a_len, b_len) = self.0.lens();
        // No checked_add - see SymmetricDifference::size_hint.
        (max(a_len, b_len), Some(a_len + b_len))
    }

    fn min(mut self) -> Option<&'a T> {
        self.next()
    }
}

impl<T: Ord> FusedIterator for Union<'_, T> {}

/// A lazy iterator producing elements in the difference of `BTreeSet`s.
///
/// This `struct` is created by the [`difference`] method on [`BTreeSet`].
/// See its documentation for more.
///
/// [`difference`]: BTreeSet::difference
#[must_use = "this returns the difference as an iterator, \
              without modifying either input set"]
pub struct Difference<'a, T: 'a, A: Tuning> {
    inner: DifferenceInner<'a, T, A>,
}

#[derive(Debug)]
enum DifferenceInner<'a, T: 'a, A: Tuning> {
    Stitch {
        // iterate all of `self` and some of `other`, spotting matches along the way
        self_iter: Iter<'a, T>,
        other_iter: Peekable<Iter<'a, T>>,
    },
    Search {
        // iterate `self`, look up in `other`
        self_iter: Iter<'a, T>,
        other_set: &'a BTreeSet<T, A>,
    },
    Iterate(Iter<'a, T>), // simply produce all elements in `self`
}

impl<T, A: Tuning> Clone for Difference<'_, T, A> {
    fn clone(&self) -> Self {
        Difference {
            inner: match &self.inner {
                DifferenceInner::Stitch {
                    self_iter,
                    other_iter,
                } => DifferenceInner::Stitch {
                    self_iter: self_iter.clone(),
                    other_iter: other_iter.clone(),
                },
                DifferenceInner::Search {
                    self_iter,
                    other_set,
                } => DifferenceInner::Search {
                    self_iter: self_iter.clone(),
                    other_set,
                },
                DifferenceInner::Iterate(iter) => DifferenceInner::Iterate(iter.clone()),
            },
        }
    }
}

impl<'a, T: Ord, A: Tuning> Iterator for Difference<'a, T, A> {
    type Item = &'a T;

    fn next(&mut self) -> Option<&'a T> {
        match &mut self.inner {
            DifferenceInner::Stitch {
                self_iter,
                other_iter,
            } => {
                let mut self_next = self_iter.next()?;
                loop {
                    match other_iter
                        .peek()
                        .map_or(Ordering::Less, |other_next| self_next.cmp(other_next))
                    {
                        Ordering::Less => return Some(self_next),
                        Ordering::Equal => {
                            self_next = self_iter.next()?;
                            other_iter.next();
                        }
                        Ordering::Greater => {
                            other_iter.next();
                        }
                    }
                }
            }
            DifferenceInner::Search {
                self_iter,
                other_set,
            } => loop {
                let self_next = self_iter.next()?;
                if !other_set.contains(self_next) {
                    return Some(self_next);
                }
            },
            DifferenceInner::Iterate(iter) => iter.next(),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let (self_len, other_len) = match &self.inner {
            DifferenceInner::Stitch {
                self_iter,
                other_iter,
            } => (self_iter.len(), other_iter.len()),
            DifferenceInner::Search {
                self_iter,
                other_set,
            } => (self_iter.len(), other_set.len()),
            DifferenceInner::Iterate(iter) => (iter.len(), 0),
        };
        (self_len.saturating_sub(other_len), Some(self_len))
    }

    fn min(mut self) -> Option<&'a T> {
        self.next()
    }
}

impl<T: Ord, A: Tuning> FusedIterator for Difference<'_, T, A> {}

impl<T: Ord + Clone, A: Tuning> BitAnd<&BTreeSet<T, A>> for &BTreeSet<T, A> {
    type Output = BTreeSet<T, A>;

    /// Returns the intersection of `self` and `rhs` as a new `BTreeSet<T>`.
    ///
    /// # Examples
    ///
    /// ```
    /// use pstd::collections::BTreeSet;
    ///
    /// let a = BTreeSet::from([1, 2, 3]);
    /// let b = BTreeSet::from([2, 3, 4]);
    ///
    /// let result = &a & &b;
    /// assert_eq!(result, BTreeSet::from([2, 3]));
    /// ```
    fn bitand(self, rhs: &BTreeSet<T, A>) -> BTreeSet<T, A> {
        BTreeSet::from_sorted_iter(self.intersection(rhs).cloned(), self.map.get_tuning())
    }
}

impl<T: Ord + Clone, A: Tuning> BitOr<&BTreeSet<T, A>> for &BTreeSet<T, A> {
    type Output = BTreeSet<T, A>;

    /// Returns the union of `self` and `rhs` as a new `BTreeSet<T>`.
    ///
    /// # Examples
    ///
    /// ```
    /// use pstd::collections::BTreeSet;
    ///
    /// let a = BTreeSet::from([1, 2, 3]);
    /// let b = BTreeSet::from([3, 4, 5]);
    ///
    /// let result = &a | &b;
    /// assert_eq!(result, BTreeSet::from([1, 2, 3, 4, 5]));
    /// ```
    fn bitor(self, rhs: &BTreeSet<T, A>) -> BTreeSet<T, A> {
        BTreeSet::from_sorted_iter(self.union(rhs).cloned(), self.map.get_tuning())
    }
}

impl<T: Ord + Clone, A: Tuning> BitXor<&BTreeSet<T, A>> for &BTreeSet<T, A> {
    type Output = BTreeSet<T, A>;

    /// Returns the symmetric difference of `self` and `rhs` as a new `BTreeSet<T>`.
    ///
    /// # Examples
    ///
    /// ```
    /// use pstd::collections::BTreeSet;
    ///
    /// let a = BTreeSet::from([1, 2, 3]);
    /// let b = BTreeSet::from([2, 3, 4]);
    ///
    /// let result = &a ^ &b;
    /// assert_eq!(result, BTreeSet::from([1, 4]));
    /// ```
    fn bitxor(self, rhs: &BTreeSet<T, A>) -> BTreeSet<T, A> {
        BTreeSet::from_sorted_iter(
            self.symmetric_difference(rhs).cloned(),
            self.map.get_tuning(),
        )
    }
}

impl<T: Ord + Clone, A: Tuning> Sub<&BTreeSet<T, A>> for &BTreeSet<T, A> {
    type Output = BTreeSet<T, A>;

    /// Returns the difference of `self` and `rhs` as a new `BTreeSet<T>`.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::BTreeSet;
    ///
    /// let a = BTreeSet::from([1, 2, 3]);
    /// let b = BTreeSet::from([3, 4, 5]);
    ///
    /// let result = &a - &b;
    /// assert_eq!(result, BTreeSet::from([1, 2]));
    /// ```
    fn sub(self, rhs: &BTreeSet<T, A>) -> BTreeSet<T, A> {
        BTreeSet::from_sorted_iter(self.difference(rhs).cloned(), self.map.get_tuning())
    }
}

impl<T: Ord, A: Tuning> Extend<T> for BTreeSet<T, A> {
    #[inline]
    fn extend<Iter: IntoIterator<Item = T>>(&mut self, iter: Iter) {
        iter.into_iter().for_each(move |elem| {
            self.insert(elem);
        });
    }
    /*
        #[inline]
        fn extend_one(&mut self, elem: T) {
            self.insert(elem);
        }
    */
}

impl<'a, T: 'a + Ord + Copy, A: Tuning> Extend<&'a T> for BTreeSet<T, A> {
    fn extend<I: IntoIterator<Item = &'a T>>(&mut self, iter: I) {
        self.extend(iter.into_iter().cloned());
    }
    /*
        #[inline]
        fn extend_one(&mut self, &elem: &'a T) {
            self.insert(elem);
        }
    */
}

#[cfg(test)]
mod tests;
