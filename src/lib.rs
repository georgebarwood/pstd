#![deny(missing_docs)]

//! Crate with parts of Rust std library ( different implementations, features not yet stabilised etc ), in particular [boxed::Box], [`collections::BTreeMap`], [`collections::BTreeSet`] and [`vec::Vec`].
//!
//! Box only has minimal methods/traits implemented so far.

//!# Features
//!
//! This crate supports the following cargo features:
//! - `serde` : enables serialisation of [`collections::BTreeMap`] and [`collections::BTreeSet`] via serde crate.
//! - `unsafe-optim` : Enable unsafe optimisations in release mode.

/// Memory allocation.
pub mod alloc;

/// Containers.
pub mod collections;

/// [`Vec`] similar to [`std::vec::Vec`].
pub mod vec;
pub use vec::Vec;

/// [`Box`] similar to [`std::boxed::Box`].
pub mod boxed;
pub use boxed::Box;

#[cfg(test)]
mod testing {
    pub mod crash_test;
    pub mod macros;
}
