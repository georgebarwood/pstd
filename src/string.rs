use crate::{
    VecA,
    alloc::{Allocator, Global},
};
use std::{borrow::Borrow, fmt::Debug, ops, ptr};

/// A UTF-8–encoded, growable string allocated from Global.
pub type String = StringA<Global>;

/// A UTF-8–encoded, growable string.
#[derive(Clone)]
pub struct StringA<A: Allocator>(VecA<u8, A>);

impl<A: Allocator> StringA<A> {
    /// Create an empty string.
    pub fn new() -> Self
    where
        A: Default,
    {
        Self(VecA::new())
    }

    /// Create an empty string in specified allocator.
    pub const fn new_in(a: A) -> Self {
        Self(VecA::new_in(a))
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

    /// Appends the given [`char`] to the end of this `String`.
    ///
    /// # Panics
    ///
    /// Panics if the new capacity exceeds `isize::MAX` _bytes_.
    ///
    /// # Examples
    ///
    /// ```
    /// use pstd::String;
    /// let mut s = String::from("abc");
    ///
    /// s.push('1');
    /// s.push('2');
    /// s.push('3');
    ///
    /// assert_eq!("abc123", s);
    /// ```
    pub fn push(&mut self, ch: char) {
        let mut buf: [u8; 4] = [0; 4];
        self.push_str(ch.encode_utf8(&mut buf));
    }

    /// Removes the last character from the string buffer and returns it.
    ///
    /// Returns [`None`] if this `String` is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use pstd::String;
    /// let mut s = String::from("abč");
    ///
    /// assert_eq!(s.pop(), Some('č'));
    /// assert_eq!(s.pop(), Some('b'));
    /// assert_eq!(s.pop(), Some('a'));
    ///
    /// assert_eq!(s.pop(), None);
    /// ```
    pub fn pop(&mut self) -> Option<char> {
        let ch = self.chars().next_back()?;
        let newlen = self.len() - ch.len_utf8();
        self.0.truncate(newlen);
        Some(ch)
    }

    /// Inserts a string slice into this `String` at byte position `idx`.
    ///
    /// Reallocates if `self.capacity()` is insufficient, which may involve copying all
    /// `self.capacity()` bytes. Makes space for the insertion by copying all bytes of
    /// `&self[idx..]` to new positions.
    ///
    /// Note that calling this in a loop can result in quadratic behavior.
    ///
    /// # Panics
    ///
    /// Panics if `idx` is larger than the `String`'s length, or if it does not
    /// lie on a [`char`] boundary.
    ///
    /// # Examples
    ///
    /// ```
    /// use pstd::String;
    /// let mut s = String::from("bar");
    ///
    /// s.insert_str(0, "foo");
    ///
    /// assert_eq!("foobar", s);
    /// ```
    pub fn insert_str(&mut self, idx: usize, string: &str) {
        assert!(self.is_char_boundary(idx));

        let len = self.len();
        let amt = string.len();
        self.reserve(amt);

        // SAFETY: Move the bytes starting from `idx` to their new location `amt` bytes
        // ahead. This is safe because sufficient capacity was just reserved, and `idx`
        // is a char boundary.
        unsafe {
            ptr::copy(
                self.0.as_ptr().add(idx),
                self.0.as_mut_ptr().add(idx + amt),
                len - idx,
            );
        }

        // SAFETY: Copy the new string slice into the vacated region if `idx != len`,
        // or into the uninitialized spare capacity otherwise. The borrow checker
        // ensures that the source and destination do not overlap.
        unsafe {
            ptr::copy_nonoverlapping(string.as_ptr(), self.0.as_mut_ptr().add(idx), amt);
        }

        // SAFETY: Update the length to include the newly added bytes.
        unsafe {
            self.0.set_len(len + amt);
        }
    }

    /// Inserts a character into this `String` at byte position `idx`.
    ///
    /// Reallocates if `self.capacity()` is insufficient, which may involve copying all
    /// `self.capacity()` bytes. Makes space for the insertion by copying all bytes of
    /// `&self[idx..]` to new positions.
    ///
    /// Note that calling this in a loop can result in quadratic behavior.
    ///
    /// # Panics
    ///
    /// Panics if `idx` is larger than the `String`'s length, or if it does not
    /// lie on a [`char`] boundary.
    ///
    /// # Examples
    ///
    /// ```
    /// use pstd::String;
    /// let mut s = String::with_capacity(3);
    ///
    /// s.insert(0, 'f');
    /// s.insert(1, 'o');
    /// s.insert(2, 'o');
    ///
    /// assert_eq!("foo", s);
    /// ```
    pub fn insert(&mut self, idx: usize, ch: char) {
        assert!(self.is_char_boundary(idx));
        let mut buf: [u8; 4] = [0; 4];
        self.insert_str(idx, ch.encode_utf8(&mut buf));
    }

    /// Returns the length of this `String`, in bytes, not [`char`]s or
    /// graphemes. In other words, it might not be what a human considers the
    /// length of the string.
    ///
    /// # Examples
    ///
    /// ```
    /// use pstd::String;
    /// let a = String::from("foo");
    /// assert_eq!(a.len(), 3);
    ///
    /// let fancy_f = String::from("ƒoo");
    /// assert_eq!(fancy_f.len(), 4);
    /// assert_eq!(fancy_f.chars().count(), 3);
    /// ```
    #[must_use]
    pub const fn len(&self) -> usize {
        self.0.len()
    }

    /// Returns `true` if this `String` has a length of zero, and `false` otherwise.
    ///
    /// # Examples
    ///
    /// ```
    /// use pstd::String;
    /// let mut v = String::new();
    /// assert!(v.is_empty());
    ///
    /// v.push('a');
    /// assert!(!v.is_empty());
    /// ```
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Replaces all matches of a pattern with another string.
    pub fn replace(&self, pat: &str, with: &str) -> Self
    where
        A: Clone,
    {
        let mut result = Self::new_in(self.0.allocator().clone());
        let mut last_end = 0;
        for (start, part) in self.match_indices(pat) {
            result.push_str(unsafe { self.get_unchecked(last_end..start) });
            result.push_str(with);
            last_end = start + part.len();
        }
        result.push_str(unsafe { self.get_unchecked(last_end..self.len()) });
        result
    }

    /// Splits the string into two at the given byte index.
    ///
    /// Returns a newly allocated `String`. `self` contains bytes `[0, at)`, and
    /// the returned `String` contains bytes `[at, len)`. `at` must be on the
    /// boundary of a UTF-8 code point.
    ///
    /// Note that the capacity of `self` does not change.
    ///
    /// # Panics
    ///
    /// Panics if `at` is not on a `UTF-8` code point boundary, or if it is beyond the last
    /// code point of the string.
    ///
    /// # Examples
    ///
    /// ```
    /// use pstd::String;
    /// let mut hello = String::from("Hello, World!");
    /// let world = hello.split_off(7);
    /// assert_eq!(hello, "Hello, ");
    /// assert_eq!(world, "World!");
    /// ```
    #[must_use = "use `.truncate()` if you don't need the other half"]
    pub fn split_off(&mut self, at: usize) -> Self
    where
        A: Clone,
    {
        assert!(self.is_char_boundary(at));
        Self(self.0.split_off(at))
    }

    /// Truncates this `String`, removing all contents.
    ///
    /// While this means the `String` will have a length of zero, it does not
    /// touch its capacity.
    ///
    /// # Examples
    ///
    /// ```
    /// use pstd::String;
    /// let mut s = String::from("foo");
    ///
    /// s.clear();
    ///
    /// assert!(s.is_empty());
    /// assert_eq!(0, s.len());
    /// assert_eq!(3, s.capacity());
    /// ```
    pub fn clear(&mut self) {
        self.0.clear();
    }

    /// Creates a String from a str in the specified allocator.
    pub fn from_str_in(s: &str, alloc: A) -> Self {
        let mut v = VecA::with_capacity_in(s.len(), alloc);
        v.extend_from_slice(s.as_bytes());
        Self(v)
    }

    /// Create an empty string with specified capacity.
    pub fn with_capacity(cap: usize) -> Self
    where
        A: Default,
    {
        Self(VecA::with_capacity(cap))
    }

    /// Returns this `String`'s capacity, in bytes.
    ///
    /// # Examples
    ///
    /// ```
    /// use pstd::String;
    /// let s = String::with_capacity(10);
    ///
    /// assert!(s.capacity() >= 10);
    /// ```
    #[must_use]
    pub const fn capacity(&self) -> usize {
        self.0.capacity()
    }

    /// Reserves capacity for at least `additional` bytes more than the
    /// current length. The allocator may reserve more space to speculatively
    /// avoid frequent allocations. After calling `reserve`,
    /// capacity will be greater than or equal to `self.len() + additional`.
    /// Does nothing if capacity is already sufficient.
    ///
    /// # Panics
    ///
    /// Panics if the new capacity exceeds `isize::MAX` _bytes_.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// use pstd::String;
    /// let mut s = String::new();
    ///
    /// s.reserve(10);
    ///
    /// assert!(s.capacity() >= 10);
    /// ```
    ///
    /// This might not actually increase the capacity:
    ///
    /// ```
    /// use pstd::String;
    /// let mut s = String::with_capacity(10);
    /// s.push('a');
    /// s.push('b');
    ///
    /// // s now has a length of 2 and a capacity of at least 10
    /// let capacity = s.capacity();
    /// assert_eq!(2, s.len());
    /// assert!(capacity >= 10);
    ///
    /// // Since we already have at least an extra 8 capacity, calling this...
    /// s.reserve(8);
    ///
    /// // ... doesn't actually increase.
    /// assert_eq!(capacity, s.capacity());
    /// ```
    pub fn reserve(&mut self, additional: usize) {
        self.0.reserve(additional)
    }

    /// Extracts a string slice containing the entire String.
    pub fn as_str(&self) -> &str {
        unsafe { str::from_utf8_unchecked(self.0.as_slice()) }
    }

    /// Converts a String into a mutable string slice.
    pub fn as_mut_str(&mut self) -> &mut str {
        unsafe { str::from_utf8_unchecked_mut(self.0.as_mut_slice()) }
    }

    /// Returns a byte slice of this `String`'s contents.
    ///
    /// # Examples
    ///
    /// ```
    /// use pstd::String;
    /// let s = String::from("hello");
    ///
    /// assert_eq!(&[104, 101, 108, 108, 111], s.as_bytes());
    /// ```
    #[must_use]
    pub const fn as_bytes(&self) -> &[u8] {
        self.0.as_slice()
    }

    /// Converts a `String` into a byte vector.
    ///
    /// This consumes the `String`, so we do not need to copy its contents.
    ///
    /// # Examples
    ///
    /// ```
    /// use pstd::String;
    /// let s = String::from("hello");
    /// let bytes = s.into_bytes();
    ///
    /// assert_eq!(&[104, 101, 108, 108, 111][..], &bytes[..]);
    /// ```
    #[must_use = "`self` will be dropped if the result is not used"]
    pub fn into_bytes(self) -> VecA<u8, A> {
        self.0
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

impl<A: Allocator> Borrow<str> for StringA<A> {
    fn borrow(&self) -> &str {
        self.as_str()
    }
}

impl<A: Allocator> std::fmt::Display for StringA<A> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&**self, f)
    }
}

impl<A: Allocator> Debug for StringA<A> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        std::fmt::Debug::fmt(&**self, f)
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
        **self == **other
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

impl<A: Allocator> PartialEq<StringA<A>> for str {
    fn eq(&self, s: &StringA<A>) -> bool {
        self == &**s
    }
}

impl<A: Allocator> PartialEq<StringA<A>> for &str {
    fn eq(&self, s: &StringA<A>) -> bool {
        &**s == *self
    }
}

impl<A: Allocator> PartialEq<str> for StringA<A> {
    fn eq(&self, s: &str) -> bool {
        s == &**self
    }
}

impl<A: Allocator> PartialEq<&str> for StringA<A> {
    fn eq(&self, s: &&str) -> bool {
        &**self == *s
    }
}

impl<A: Allocator> std::fmt::Write for StringA<A> {
    fn write_str(&mut self, s: &str) -> Result<(), std::fmt::Error> {
        self.push_str(s);
        Ok(())
    }
}

impl<A: Allocator + Default> From<&str> for StringA<A> {
    fn from(s: &str) -> Self {
        let mut v = VecA::with_capacity(s.len());
        v.extend_from_slice(s.as_bytes());
        Self(v)
    }
}

#[cfg(feature = "serde")]
use {
    serde::{
        Deserialize, Deserializer, Serialize, Serializer,
        de::{Error, Visitor},
    },
    std::fmt,
};

#[cfg(feature = "serde")]
impl<A: Allocator> Serialize for StringA<A> {
    #[inline]
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(self)
    }
}

#[cfg(feature = "serde")]
impl<'de, A: Allocator + Default> Deserialize<'de> for StringA<A> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let v: MyVisitor<A> = MyVisitor {
            pd: std::marker::PhantomData,
        };
        let s = deserializer.deserialize_string(v).unwrap();
        Ok(Self::from(&*s))
    }
}

#[cfg(feature = "serde")]
struct MyVisitor<A: Allocator> {
    pd: std::marker::PhantomData<A>,
}

#[cfg(feature = "serde")]
impl<'a, A: Allocator + Default> Visitor<'a> for MyVisitor<A> {
    type Value = StringA<A>;

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("a string")
    }

    fn visit_string<E>(self, s: std::string::String) -> Result<Self::Value, E>
    where
        E: Error,
    {
        Ok(Self::Value::from(&*s))
    }

    fn visit_str<E>(self, s: &str) -> Result<Self::Value, E>
    where
        E: Error,
    {
        Ok(Self::Value::from(s))
    }
}

#[test]
fn test_string() {
    use crate::localalloc::Local;

    let mut s: StringA<Local> = StringA::new();
    s.push_str("George");
    assert!(s == "George");
    assert!("George" == s);
}

#[test]
fn test_string_replace() {
    use crate::localalloc::Temp;

    let mut s = StringA::<Temp>::from("George");
    s = s.replace("eorge", "raham");
    assert!(s == "Graham");
    println!("s={}", s);
}

#[test]
fn test_string_write() {
    use crate::localalloc::Temp;

    let mut output = StringA::<Temp>::new();

    use std::fmt::Write;

    let x: i64 = -319;

    write!(&mut output, "Hello {}! {}", "world", x).unwrap();

    assert_eq!(output, "Hello world! -319");

    println!("output={}", output);
}
