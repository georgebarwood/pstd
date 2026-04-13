#![deny(missing_docs)]
#![cfg_attr(feature = "dynbox", feature(unsize))]
#![cfg_attr(feature = "dynbox", feature(coerce_unsized))]
//#![cfg_attr(feature = "dynbox", feature(core_intrinsics))]

//! Crate with parts of Rust std library ( different implementations, features not yet stabilised etc ),
//! in particular [`Box`], [`Vec`], [`Rc`], [`String`] and [collections]::{ [`BTreeMap`](collections::BTreeMap), [`BTreeSet`](collections::BTreeSet), [`HashMap`](collections::HashMap), [`HashSet`](collections::HashMap) }.
//!
//! Box, Rc and String currently only have minimal methods/traits implemented.
//!
//! HashMap and HashSet are imported from the hashbrown crate.
//!
//! [`RcStr`] is a reference-counted string based on [`RcSlice`].
//!
//! The [`localalloc`] module has allocator implementations.
//!
//!# Features
//!
//! This crate supports the following cargo features:
//! - `serde` : enables serialisation of [`BTreeMap`](collections::BTreeMap) and [`BTreeSet`](collections::BTreeSet) via serde crate.
//! - `unsafe-optim` : Enable unsafe optimisations in release mode.
//! - `dynbox` : enables Boxing of dyn values, requires nightly toolchain.
//! - `log-bump` : prints details of local allocation when thread terminates.

/// Memory allocation.
pub mod alloc {
    pub use allocator_api2::alloc::*;
}

/// Allocator implementations.
pub mod localalloc;

/// Containers: BTreeMap, BTreeSet, HashMap and HashSet.
pub mod collections;

/// [`Vec`] similar to [`std::vec::Vec`].
pub mod vec;
pub use vec::{Vec, VecA};

/// [`String`] similar to [`std::string::String`]
pub mod string;
pub use string::{String, StringA};

/// [`Box`] similar to [`std::boxed::Box`].
pub mod boxed;
pub use boxed::{Box, BoxA};

/// [`Rc`] similar to [`std::rc::Rc`].
pub mod rc;
pub use rc::{Rc, RcA, RcSlice, RcSliceA, RcStr, RcStrA};

#[cfg(test)]
mod testing {
    pub mod crash_test;
    pub mod macros;
}
