Wrapping-in-range arithmetic for custom ranges via the [`WrappingInRange`] type

[<img alt="github" src="https://img.shields.io/badge/github-nik-rev/wrapping-in-range-8da0cb?style=for-the-badge&labelColor=555555&logo=github" height="20">](https://github.com/nik-rev/wrapping-in-range)
[<img alt="crates.io" src="https://img.shields.io/crates/v/wrapping-in-range.svg?style=for-the-badge&color=fc8d62&logo=rust" height="20">](https://crates.io/crates/wrapping-in-range)
[<img alt="docs.rs" src="https://img.shields.io/badge/docs.rs-wrapping-in-range-66c2a5?style=for-the-badge&labelColor=555555&logo=docs.rs" height="20">](https://docs.rs/wrapping-in-range)

These arithmetic operations act just like `std`'s `.wrapping_sub()`, `.wrapping_add()`, etc. but for a custom user-provided range.

# Examples

```rust
use std::ops::Range;
use wrapping_in_range::WrappingInRange;

let w = |i: i16| WrappingInRange(i, -1..=1);

assert_eq!(
    [-2, -1, 0, 1, 2].map(|i| w(i) - 1),
    [ 0, 1, -1, 0, 1]
);
assert_eq!(
    [-2, -1, 0,  1, 2].map(|i| w(i) + 1),
    [-1,  0, 1, -1, 0]
);
```
