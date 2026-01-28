#![deny(missing_docs)]
#![cfg_attr(test, feature(btree_cursors, assert_matches))]

//! Crate with parts of Rust std library ( different implementations, features not yet stabilised etc ), in particular [`collections::btree_map`].

/// Memory allocation.
pub mod alloc;

/// Containers.
pub mod collections {
    pub mod btree_map;

    pub use btree_map::BTreeMap;

    pub mod btree_set;

    pub use btree_set::BTreeSet;
}
