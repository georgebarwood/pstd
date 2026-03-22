#![deny(missing_docs)]
#![cfg_attr(feature = "dynbox", feature(unsize))]
#![cfg_attr(feature = "dynbox", feature(coerce_unsized))]
// #![cfg_attr(feature = "dynbox", feature(dispatch_from_dyn))]
// #![feature(layout_for_ptr)]

//#![feature(coerce_unsized)]
//#![feature(unsize)]

//! Crate with parts of Rust std library ( different implementations, features not yet stabilised etc ), in particular [boxed::Box], [`collections::BTreeMap`], [`collections::BTreeSet`] and [`vec::Vec`].
//!
//! Box only has minimal methods/traits implemented so far.

//!# Features
//!
//! This crate supports the following cargo features:
//! - `serde` : enables serialisation of [`collections::BTreeMap`] and [`collections::BTreeSet`] via serde crate.
//! - `unsafe-optim` : Enable unsafe optimisations in release mode.
//! - `dynbox` : enables Boxing of dyn values, requires nightly toolchain.

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
