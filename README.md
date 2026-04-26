Crate with parts of Rust std library ( different implementations, features not yet stabilised etc ), in particular Box, Vec, Rc, String and collections::{ BTreeMap, BTreeSet, HashMap, HashSet }.

Box, Rc and String do not have all std methods/traits implemented.

HashMap and HashSet are imported from the hashbrown crate.

RcStr is a reference-counted string based on RcSlice.

The localalloc module has fast thread-local bump allocators.

crates.io : https://crates.io/crates/pstd

documentation: https://docs.rs/pstd/latest/pstd/
