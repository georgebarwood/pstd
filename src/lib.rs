#![deny(missing_docs)]
// To run cargo test -F tests use Nightly ( rustup default nightly ) and enable line below
// #![cfg_attr(test, feature(btree_cursors, assert_matches))]

//! Crate with parts of Rust std library ( different implementations, features not yet stabilised etc ), in particular [`collections::BTreeMap`] and [`collections::BTreeSet`].

//!# Features
//!
//! This crate supports the following cargo features:
//! - `serde` : enables serialisation of [`collections::BTreeMap`] and [`collections::BTreeSet`] via serde crate.
//! - `unsafe-optim` : Enable unsafe optimisations in release mode.

/// Memory allocation.
pub mod alloc;

/// Containers.
pub mod collections {
    pub mod btree_map;

    pub use btree_map::BTreeMap;

    pub mod btree_set;

    pub use btree_set::BTreeSet;

    mod merge_iter;
}

#[cfg(test)]
mod testing {
    pub mod crash_test;
}
