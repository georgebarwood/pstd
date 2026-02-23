pub mod btree_map;

pub use btree_map::BTreeMap;

pub mod btree_set;

pub use btree_set::BTreeSet;

mod merge_iter;

/// The error type for `try_reserve` methods.
pub struct TryReserveError {
    // kind: TryReserveErrorKind,
}
