pub mod btree_map;

pub use btree_map::BTreeMap;

pub mod btree_set;

pub use btree_set::BTreeSet;

mod merge_iter;

/// The error type for `try_reserve` methods.
#[derive(Debug)]
pub struct TryReserveError {
    pub(crate) kind: TryReserveErrorKind,
}

impl TryReserveError {
    /// Details about the allocation that caused the error
    #[must_use]
    pub fn kind(&self) -> TryReserveErrorKind {
        self.kind.clone()
    }
}

/// Details of the allocation that caused a `TryReserveError`
#[derive(Debug, Clone)]
pub enum TryReserveErrorKind {
    /// Error due to the computed capacity exceeding the collection's maximum
    /// (usually `isize::MAX` bytes).
    CapacityOverflow,

    /// The memory allocator returned an error
    AllocError {
        /// The layout of allocation request that failed
        layout: std::alloc::Layout,
    },
}
