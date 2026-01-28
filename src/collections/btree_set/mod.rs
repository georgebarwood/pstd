//! [`BTreeSet`] similar to [`std::collections::BTreeSet`].

use crate::collections::btree_map::*;
use std::borrow::Borrow;

/// `BTreeSet` similar to [`std::collections::BTreeSet`].
pub struct BTreeSet<T, A: Tuning = DefaultTuning> {
    map: BTreeMap<T, (), A>,
}

impl<T> BTreeSet<T> {
    /// Returns a new, empty `BTreeSet`.
    #[must_use]
    pub const fn new() -> BTreeSet<T> {
        BTreeSet {
            map: BTreeMap::new(),
        }
    }
}

impl<T, A: Tuning> BTreeSet<T, A> {
    /// Returns number of elements in the set
    pub const fn len(&self) -> usize {
        self.map.len()
    }

    /// Does the set have any elements
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
    pub fn clear(&mut self) {
        self.map.clear();
    }

    /// Adds a value to the set.
    pub fn insert(&mut self, value: T) -> bool
    where
        T: Ord,
    {
        self.map.insert(value, ()).is_none()
    }

    /// Remove element from set
    pub fn remove<Q: ?Sized>(&mut self, value: &Q) -> bool
    where
        T: Borrow<Q> + Ord,
        Q: Ord,
    {
        self.map.remove(value).is_some()
    }

    /// Remove and return element from set
    pub fn take<Q: ?Sized>(&mut self, value: &Q) -> Option<T>
    where
        T: Borrow<Q> + Ord,
        Q: Ord,
    {
        self.map.remove_entry(value).map(|(k, _)| k)
    }

    /// Adds a value to the set, replacing the existing element, if any, that is
    /// equal to the value. Returns the replaced element.
    pub fn replace(&mut self, value: T) -> Option<T>
    where
        T: Ord,
    {
        // self.map.replace(value); // optimised version is todo
        let result = match self.map.remove_entry(&value) {
            None => None,
            Some((k, _v)) => Some(k),
        };
        self.insert(value);
        result
    }
}
