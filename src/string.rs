use crate::VecA;
use crate::alloc::{Allocator, Global};
use std::ops;

/// A UTF-8–encoded, growable string allocated from Global.
pub type String = StringA<Global>;

/// A UTF-8–encoded, growable string.
#[derive(Clone, Debug)]
pub struct StringA<A: Allocator = Global>(VecA<u8, A>);

impl<A: Allocator> StringA<A> {
    /// Create an empty string.
    pub fn new() -> Self
    where
        A: Default,
    {
        Self(VecA::new_in(A::default()))
    }

    /// Creates a String from a str.
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Self
    where
        A: Default,
    {
        let mut v = VecA::with_capacity_in(s.len(), A::default());
        v.extend_from_slice(s.as_bytes());
        Self(v)
    }

    /// Creates a String from a str in the specified allocator.
    pub fn from_str_in(s: &str, alloc: A) -> Self {
        let mut v = VecA::with_capacity_in(s.len(), alloc);
        v.extend_from_slice(s.as_bytes());
        Self(v)
    }

    /// Appends a given string slice onto the end of this `String`.
    /// # Example
    ///
    /// ```
    /// use pstd::{String,alloc::Global};
    /// let mut s = String::from_str_in("foo",Global);
    ///
    /// s.push_str("bar");
    ///
    /// assert_eq!("foobar", &*s);
    /// ```
    pub fn push_str(&mut self, string: &str) {
        self.0.extend_from_slice(string.as_bytes())
    }

    /// Extracts a string slice containing the entire String.
    pub fn as_str(&self) -> &str {
        unsafe { str::from_utf8_unchecked(self.0.as_slice()) }
    }

    /// Converts a String into a mutable string slice.
    pub fn as_mut_str(&mut self) -> &mut str {
        unsafe { str::from_utf8_unchecked_mut(self.0.as_mut_slice()) }
    }
}

impl<A: Allocator> ops::Deref for StringA<A> {
    type Target = str;

    fn deref(&self) -> &str {
        self.as_str()
    }
}

impl<A: Allocator> ops::DerefMut for StringA<A> {
    fn deref_mut(&mut self) -> &mut str {
        self.as_mut_str()
    }
}

impl<A: Allocator> std::fmt::Display for StringA<A> {
    #[inline]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&**self, f)
    }
}

impl<A: Allocator> std::hash::Hash for StringA<A> {
    fn hash<H: std::hash::Hasher>(&self, hasher: &mut H) {
        (**self).hash(hasher)
    }
}

impl<A: Allocator> Eq for StringA<A> {}

impl<A1: Allocator, A2: Allocator> PartialEq<StringA<A2>> for StringA<A1> {
    fn eq(&self, other: &StringA<A2>) -> bool {
        self[..] == other[..]
    }
}

impl<A1, A2> PartialOrd<StringA<A2>> for StringA<A1>
where
    A1: Allocator,
    A2: Allocator,
{
    fn partial_cmp(&self, other: &StringA<A2>) -> Option<std::cmp::Ordering> {
        PartialOrd::partial_cmp(&**self, &**other)
    }
}

impl<A: Allocator> Ord for StringA<A> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        std::cmp::Ord::cmp(&**self, &**other)
    }
}

impl<A: Allocator> Default for StringA<A>
where
    A: Default,
{
    fn default() -> Self {
        Self::new()
    }
}

#[test]
fn test_string() {
    use crate::localalloc::Local;
    use std::ops::Deref;

    let mut s: StringA<Local> = StringA::new();
    s.push_str("George");
    assert!(s.deref() == "George");
}
