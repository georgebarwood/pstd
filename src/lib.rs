#![deny(missing_docs)]
#![cfg_attr(test, feature(btree_cursors, assert_matches))]

//! Crate with parts of Rust std library ( different implementations, features not yet stabilised etc )

/// Memory allocation.
pub mod alloc;

/// Containers.
pub mod collections {
    /// BTreeMap.
    pub mod btree_map;

    pub use btree_map::BTreeMap;
}
