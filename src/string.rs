use crate::alloc::{Allocator, Global};
use crate::Vec;
use std::ops;

/// A UTF-8–encoded, growable string.
#[derive(Clone, Debug)]
pub struct String<A: Allocator=Global>( Vec<u8, A> );

impl <A:Allocator> String<A>
{
    /// Creates a String from a str in the specified allocator.
    pub fn from_str_in(s: &str, alloc: A) -> Self
    {
        let mut v = Vec::with_capacity_in(s.len(), alloc);
        v.extend_from_slice(s.as_bytes());
        Self( v )
    }

    /// Extracts a string slice containing the entire String.
    pub fn as_str(&self) -> &str
    {
        unsafe { str::from_utf8_unchecked(self.0.as_slice()) }
    }

    /// Converts a String into a mutable string slice.
    pub fn as_mut_str(&mut self) -> &mut str
    {
        unsafe { str::from_utf8_unchecked_mut(self.0.as_mut_slice()) }
    }
}

impl<A:Allocator> ops::Deref for String<A> {
    type Target = str;

    fn deref(&self) -> &str {
        self.as_str()
    }
}

impl<A:Allocator> ops::DerefMut for String<A> {
    fn deref_mut(&mut self) -> &mut str {
        self.as_mut_str()
    }
}

impl<A:Allocator> std::fmt::Display for String<A> {
    #[inline]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&**self, f)
    }
}

impl<A:Allocator>  std::hash::Hash for String<A> {
    fn hash<H: std::hash::Hasher>(&self, hasher: &mut H) {
        (**self).hash(hasher)
    }
}

impl<A: Allocator> Eq for String<A> {}

impl<A1: Allocator, A2: Allocator> PartialEq<String<A2>> for String<A1>
{
    fn eq(&self, other: &String<A2>) -> bool {
        self[..] == other[..]
    }
}

impl<A1, A2> PartialOrd<String<A2>> for String<A1>
where
    A1: Allocator,
    A2: Allocator,
{
    fn partial_cmp(&self, other: &String<A2>) -> Option<std::cmp::Ordering> {
        PartialOrd::partial_cmp(&**self, &**other)
    }
}

impl<A: Allocator> Ord for String<A> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        std::cmp::Ord::cmp(&**self, &**other)
    }
}
