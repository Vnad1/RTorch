# RTorch — tooling notes (honest limitations)

This documents which verification tools are usable on this crate and which are
NOT, so future runs do not take a silent "0 tests" as a pass.

## Miri — NOT usable on the `rtorch` crate (as of this state)

`cargo +nightly-x86_64-pc-windows-gnu miri test` reports **`running 0 tests`**
for every target (lib + all bins), exit 0. That is **not** a pass and **not** a
UB-clean result — it means Miri collected no tests at all, so it checked nothing.

Root cause (combination on this host):
- `nightly-x86_64-pc-windows-gnu` + `[profile.release] panic = "abort"`.
- `build.rs` runs an external C++ compiler (`g++`) and the Vulkan SDK as a
  subprocess — subprocess execution is disallowed in the Miri interpreter, and
  `build.rs` output is not reproducible under Miri.
- The crate is `#[cfg(windows)]` with FFI (`LoadLibraryW` / `GetProcAddress` /
  `transmute_copy`), which Miri does not model.

**Conclusion:** Miri results for this crate should be treated as
"inconclusive / not run", never as evidence of absence of UB. Memory safety here
is instead established by (a) static review of every `unsafe` site (the FFI
symbol loader enforces a size check before `transmute_copy`), and (b) runtime
dynamic edge tests.

## AddressSanitizer — NOT available on this toolchain

`cargo +nightly-x86_64-pc-windows-gnu test -Zsanitizer=address` fails with
`error: unknown -Z flag specified: sanitizer`. The installed nightly Gnu build
does not enable the `sanitizer` flag, and ASan on `x86_64-pc-windows-gnu` is not
supported anyway. Use the MSVC toolchain (`cl /fsanitize=address,undefined`) if
you need a native ASan/UBSan pass; that is how the C++ engine is sanitized.

## What IS used

- `cargo check` — compile-only.
- `cargo clippy --bin rtorch` — lint; no correctness errors from the current
  `main.rs` changes.
- `cargo test --release` — the executable test suite (integration + unit),
  including `references_prebuilt_formula_dll` for the DLL-reference path.
- Dynamic edge tests of the real `rtorch.exe`: missing DLL, non-formula DLL,
  empty input, missing input, CPU/GPU via a prebuilt formula DLL — all fail
  cleanly (no panic / no access violation).

See the existing `tests/` suite and the sanitizer note in the README for what
the project relies on for correctness.
