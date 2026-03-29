use crate::alloc::{Allocator, Global};
use std::alloc::Layout;
use std::ops::Deref;
use std::ptr::NonNull;
use std::hash::Hash;
use std::hash::Hasher;
use std::cmp::Ordering;
use std::fmt;

/// A single-threaded reference-counting pointer. ‘Rc’ stands for ‘Reference Counted’.
pub struct Rc<T, A: Allocator = Global> {
    nn: NonNull<Inner<T, A>>,
}

impl<T, A: Allocator> Rc<T, A> {
    /// Allocate a new Rc in specified allocator and move v into it.
    pub fn new_in(v: T, a: A) -> Rc<T, A> {
        let layout = Layout::new::<Inner<T, A>>();
        let nn = a.allocate(layout).unwrap();
        let p = nn.as_ptr().cast::<Inner<T, A>>();

        let inner = Inner { n: 0, v, a };
        unsafe {
            std::ptr::write(p, inner);
        }

        let nn = unsafe { NonNull::new_unchecked(p) };
        Self { nn }
    }
}

/*impl<T: ?Sized, A: Allocator> Rc<T, A> {
    /// Allocates memory in the given allocator then copies s into it.
    pub fn from_str_in(s: &str, a: A) -> Rc<str, A> {
        let n = s.len();
        let layout = Layout::array::<u8>(n).unwrap();
        let nn: NonNull<[u8]> = a.allocate(layout).unwrap();
        let p: *mut u8 = nn.as_ptr().cast::<u8>();

        unsafe {
            ptr::copy_nonoverlapping(s.as_ptr(), p, n);
        }

        let p: *mut str = nn.as_ptr() as *mut str;
        let nn: NonNull<str> = unsafe { NonNull::new_unchecked(p) };
        Box::<str, A> { nn, a }
    }
}*/

struct Inner<T, A: Allocator> {
    n: usize,
    a: A,
    v: T,
}

impl<T, A: Allocator> Deref for Rc<T, A> {
    type Target = T;
    fn deref(&self) -> &T {
        let p = self.nn.as_ptr();
        unsafe { &(*p).v }
    }
}

impl<T, A: Allocator> Clone for Rc<T, A> {
    fn clone(&self) -> Rc<T, A> {
        unsafe {
            let p = self.nn.as_ptr();
            (*p).n += 1;
            let nn = NonNull::new_unchecked(p);
            Self { nn }
        }
    }
}

impl<T, A: Allocator> Drop for Rc<T, A> {
    fn drop(&mut self) {
        unsafe {
            let p = self.nn.as_ptr();
            if (*p).n == 0 {
                self.nn.drop_in_place();
                let layout = Layout::new::<Inner<T, A>>();
                let dp = NonNull::new(p.cast::<u8>()).unwrap();
                (*p).a.deallocate(dp, layout)
            } else {
                (*p).n -= 1;
            }
        }
    }
}

impl<T: Hash, A: Allocator> Hash for Rc<T, A> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (**self).hash(state);
    }
}

impl<T: Eq, A: Allocator> Eq for Rc<T, A> {}

impl<T: PartialEq, A: Allocator> PartialEq for Rc<T, A> {
    fn eq(&self, other: &Self) -> bool {
        PartialEq::eq(&**self, &**other)
    }
}

impl<T: Ord, A: Allocator> Ord for Rc<T, A> {
    fn cmp(&self, other: &Self) -> Ordering {
        Ord::cmp(&**self, &**other)
    }
}

impl<T: PartialOrd, A: Allocator> PartialOrd for Rc<T, A> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        PartialOrd::partial_cmp(&**self, &**other)
    }
}

impl<T: fmt::Display, A: Allocator> fmt::Display for Rc<T, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self.deref(), f)
    }
}

impl<T: fmt::Debug, A: Allocator> fmt::Debug for Rc<T, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self.deref(), f)
    }
}

use std::borrow::Borrow;
impl<T, A: Allocator> Borrow<T> for Rc<T, A> {
    fn borrow(&self) -> &T {
        self.deref()
    }
}


#[test]
fn rc_test()
{
    let r = Rc::new_in(99,Global);
    assert!( *r == 99 );
    {
        let r1 = r.clone();
        assert!( *r1 == 99 );
    }

    // use crate::collections::*;
    use crate::localalloc::*;
   
    let mut m = lbtreemap();
    let x : usize = 99;
    let k = lrc(x);
    m.insert( k, 55 );
    println!( "m={:?}", m);
    assert!( m.get(&99).is_some() );

    let mut m = std::collections::BTreeMap::new();
    let x: usize = 99;
    let k = std::rc::Rc::new(x);
    m.insert( k, 55 );
    assert!( m.get(&99).is_some() );

    let mut m = std::collections::BTreeMap::new();
    let x: usize = 99;
    let k = std::rc::Rc::new(std::boxed::Box::new(x));
    m.insert( k, 55 );
    assert!( m.get(&std::boxed::Box::new(x)).is_some());
    // assert!( m.get(&x).is_some() );

    let mut m = std::collections::BTreeMap::new();
    let x: String = "hello".to_string();
    let k = std::rc::Rc::new(x.clone());
    m.insert( k, 55 );
    // assert!( m.get("hello").is_some() ); // Doesn't work
    let k2 = std::rc::Rc::new(x);
    assert!( m.get(&k2).is_some() ); // Ok
    

    let mut m = std::collections::BTreeMap::new();
    let k : std::rc::Rc<str> = std::rc::Rc::from("hello");

    m.insert( k, 55 );
    assert!( m.get("hello").is_some() );

    let mut m = lbtreemap();
    let x : usize = 99;
    let k = lrc(lbox(x));
    m.insert( k.clone(), 55 );
    println!( "m={:?}", m);
    assert!( m.get(&k).is_some() );

    let bk = lbox(x);
    assert!( m.get(&bk).is_some() );
    // assert!( m.get(&x).is_some() );
}
