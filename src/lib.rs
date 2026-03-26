#![deny(missing_docs)]
#![cfg_attr(feature = "dynbox", feature(unsize))]
#![cfg_attr(feature = "dynbox", feature(coerce_unsized))]

//! Crate with parts of Rust std library ( different implementations, features not yet stabilised etc ), in particular [`Box`], [`Vec`] and [collections]::{ [`BTreeMap`](collections::BTreeMap), [`BTreeSet`](collections::BTreeSet), [`HashMap`](collections::HashMap), [`HashSet`](collections::HashMap) }.
//!
//! Box only has minimal methods/traits implemented so far.
//!
//! HashMap and HashSet are imported from the hashbrown crate.

//!# Features
//!
//! This crate supports the following cargo features:
//! - `serde` : enables serialisation of [`BTreeMap`](collections::BTreeMap) and [`BTreeSet`](collections::BTreeSet) via serde crate.
//! - `unsafe-optim` : Enable unsafe optimisations in release mode.
//! - `dynbox` : enables Boxing of dyn values, requires nightly toolchain.

/// Memory allocation.
pub mod alloc {
    pub use allocator_api2::alloc::*;
}

/// Containers: BTreeMap, BTreeSet, HashMap and HashSet.
pub mod collections;

/// [`Vec`] similar to [`std::vec::Vec`].
pub mod vec;
pub use vec::Vec;

/// [`String`] similar to [`std::string::String`]
pub mod string;
pub use string::String;

/// [`Box`] similar to [`std::boxed::Box`].
pub mod boxed;
pub use boxed::Box;

#[cfg(test)]
mod testing {
    pub mod crash_test;
    pub mod macros;
}
