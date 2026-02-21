cargo test
cargo doc
cargo clippy
cargo fmt
cargo llvm-cov
cargo llvm-cov --html --open --doctests
cargo asm
cargo doc --open --document-private-items
cargo test vec -- --nocapture
cargo miri test vec -- --nocapture
rustup default stable
rustup default nightly