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
- `cargo clippy --all-targets` — lint; the parity test's hardcoded `FRAC_PI_*`
  approximation was caught and fixed (approximate-value lint). Zero errors after.
- `cargo test --all` — executable test suite; exit 0 (rtw includes
  `rejects_bad_magic`/`rejects_truncated`, formula-abi edge cases, autograd,
  tensor parity, Striker integration).
- Dynamic edge tests of the real `rtorch.exe`: missing DLL, non-formula DLL,
  empty input, missing input, CPU/GPU via a prebuilt formula DLL — all fail
  cleanly (no panic / no access violation).

## Full-unsafe static review (substitute for Miri)

Every `unsafe` block (78 across `src/{formula,main,vk}.rs`) was reviewed. The
three FFI symbol loaders (`formula.rs::load_symbol`, `main.rs::resolve_fn`,
`vk.rs::sym`) all enforce a **pointer-size contract** before `transmute_copy`
(`size_of::<T>() == size_of::<*mut c_void>()`, else `None`) — no size-mismatched
transmute, so no UB from a non-pointer-sized `T`. Note this is the same pattern
as Miri would check; Miri itself is unavailable (above).

## cargo-fuzz — NOT usable offline

`crates.io` returns **403** on this host (network-restricted), so `cargo-fuzz`
cannot be installed. No fuzzing.

## Known documentation code mismatch (non-defect, technical debt)

`src/error.rs` states the library "returns errors instead of `panic!/expect!` on
bad input", but `src/autograd.rs` / `src/gpu_tensor.rs` use `panic!` for
defensive shape/gather/grad-length checks (all with upstream validation, so not
UB). Converting these to `Result` is an API-level change that would ripple into
Striker; recorded here rather than done, per the no-big-refactor rule.

---

See the existing `tests/` suite and the sanitizer note in the README for what
the project relies on for correctness.
