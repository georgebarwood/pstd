#![deny(missing_docs)]
#![cfg_attr(test, feature(btree_cursors, assert_matches))]

//! crate with preview of parts of std ( unstable features ).

/// Memory allocation.
pub mod alloc {}

/// Containers.
pub mod collections {
    /// BTreeMap.
    pub mod btree_map;
}
