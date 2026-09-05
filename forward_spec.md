# forward_spec — RTorch formula forward-pass specification

This is the authoritative spec for the **forward-only** formula protocol
(scheme **D2 option (a)**): formulas and `model.dll` carry the forward pass only;
**training does NOT go through the formula**. Backward/autograd is owned by the
model framework (Striker) via RTorch's autograd (`rtorch::autograd` /
`rtorch::gvar`), not by the formula ABI.

## Why forward-only

RTorch's formula ABI (`rtorch.h`) is **stateless and forward-only**: a formula
declares `rtorch_output_size` + `rtorch_compute` (CPU) and optionally
`rtorch_gpu_kernel` / `rtorch_gpu_groups` (GPU). There is no `rtorch_backward`.
Trying to run training through `model.dll` would force the formula to also own
gradient/parameter state and a backward pass — which the framework owns.

So the boundary is settled (D2 → (a)):

- **formula / model.dll** = the **forward pass** only (inference / a single
  compute step). Input blobs in → output blob out.
- **training (forward + backward + optimizer)** = in the model framework
  (Striker), using RTorch's autograd (`Var` graph on CPU, `GVar` on GPU) and
  the optimizers (`Adam` / `AdamG`).

A `model.rtw` / `memory.rtw` container is data + the forward kernel; the trainable
weights and the memory state are **not** threaded through the formula ABI for a
backward pass. Persistence and scheduling are owned by the model framework.

## The forward ABI (from `rtorch.h`)

```c
// How many output bytes the formula needs for the given inputs.
unsigned long long rtorch_output_size(int n_in, const rtorch_blob* in, int device);

// Forward compute. Write results into out->data (out->len bytes available).
// Return 0 on success, non-zero on failure.
int rtorch_compute(int n_in, const rtorch_blob* in, rtorch_blob* out, int device);

// Optional GPU forward: a GLSL compute kernel (binding 0..n-1 inputs, binding n
// output). If exported, `--device gpu` uses this instead of host
// `rtorch_compute`.
const char* rtorch_gpu_kernel(void);
void rtorch_gpu_groups(int* gx, int* gy, int* gz);
```

`rtorch_blob { const void* data; size_t len; }` holds raw bytes (little-endian,
host-ordered). `device`: `0` = CPU host, `>0` = best accelerator (Vulkan compute).

## Method

- The formula decides its own input/output byte layout. It is **pure**: given the
  same input blobs + device, it produces the same output bytes (subject to the
  usual f32 vs f64 host/accelerator precision difference).
- The pipeline (`rtorch::formula::run`) compiles the `.cpp` (or loads a `.dll`),
  resolves the entry points, and dispatches on the requested device.
- **No state, no parameter blob, no backward** in the forward ABI. If a formula
  needs parameters, they are supplied as extra input blobs and used only in the
  forward computation.

## Parity

The forward pass must agree between `device = 0` (CPU) and `device = 1` (GPU)
within the precision implied by the data type (never bit-exact between f64 host
and f32 accelerator). `formula_trig.cpp` is the reference forward formula: it
implements both the CPU `rtorch_compute` and the GPU `rtorch_gpu_kernel`, so its
outputs can be compared across devices. See the parity test.

## Related

- `RTW.md` — the `.rtw` container (result / kernel / model / memory).
- `src/formula.rs` — the library pipeline (`rtorch::formula::run`).
- `rtorch.h` — the public forward ABI header.
- The Striker model framework owns training (forward via this formula is not used
  for backprop).
