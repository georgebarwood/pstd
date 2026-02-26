#![deny(missing_docs)]

//! Crate with parts of Rust std library ( different implementations, features not yet stabilised etc ), in particular [`collections::BTreeMap`], [`collections::BTreeSet`] and [`vec::Vec`].

//!# Features
//!
//! This crate supports the following cargo features:
//! - `serde` : enables serialisation of [`collections::BTreeMap`] and [`collections::BTreeSet`] via serde crate.
//! - `unsafe-optim` : Enable unsafe optimisations in release mode.

/// Memory allocation.
pub mod alloc;

/// Containers.
pub mod collections;

/// [`vec::Vec`] similar to [`std::vec::Vec`], not yet well tested.
pub mod vec;

#[cfg(test)]
mod testing {
    pub mod crash_test;
    pub mod macros;
}
