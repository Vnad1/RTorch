// RTorch Workfile (.rtw) — a self-contained container for results and compute
// kernels. Not tied to PyTorch/any runtime: pure little-endian binary that any
// language can read/write. Kinds:
//   result (0) : packed data tensor(s) -> shape/dtype/data
//   kernel (1) : embedded compute kernel (GLSL source) -> runnable by `rtorch x.rtw`
//   model  (2) : frozen weights (named params + optional Adam state) -> Model payload
//   memory (3) : Striker memory state (fragment list) -> Memory payload
// Format:
//   magic  "RTW1" (4B)
//   u16 version
//   u8  kind    (0=result, 1=kernel, 2=model, 3=memory)
//   u8  dtype   (0=fp32,1=fp16,2=fp8,3=fp4,4=int32,5=bytes)
//   u32 rank
//   u32 shape[rank]
//   u64 data_len  (byte length of the data payload, little-endian)
//   bytes data[data_len]              (all kinds; content depends on `kind`)
//   u8  flags   (0x01 = has_kernel)
//   u64 klen  + bytes kernel          (if flags & 0x01)
//   magic  "RTEND" (5B)
use std::io::{self, Read, Write};

pub const DTYPE_FP32: u8 = 0;
pub const DTYPE_FP16: u8 = 1;
pub const DTYPE_FP8: u8 = 2; // E4M3
pub const DTYPE_FP4: u8 = 3; // E2M1
pub const DTYPE_INT32: u8 = 4;
pub const DTYPE_BYTES: u8 = 5;

pub const KIND_RESULT: u8 = 0;
pub const KIND_KERNEL: u8 = 1;
pub const KIND_MODEL: u8 = 2; // 冻结权重(权重+结构元数据)
pub const KIND_MEMORY: u8 = 3; // Striker 记忆 state(Frag 列表序列化, 动态增长)

const FLAG_HAS_KERNEL: u8 = 0x01;

pub fn dtype_width(d: u8) -> usize {
    match d {
        DTYPE_FP32 | DTYPE_INT32 => 4,
        DTYPE_FP16 => 2,
        DTYPE_FP8 => 1,
        DTYPE_FP4 => 1, // packed 1 byte for v1 simplicity
        _ => 1,
    }
}

pub fn dtype_name(d: u8) -> &'static str {
    match d {
        DTYPE_FP32 => "fp32",
        DTYPE_FP16 => "fp16",
        DTYPE_FP8 => "fp8(E4M3)",
        DTYPE_FP4 => "fp4(E2M1)",
        DTYPE_INT32 => "int32",
        _ => "bytes",
    }
}

#[derive(Debug, Clone)]
pub struct Rtw {
    pub kind: u8,
    pub dtype: u8,
    pub shape: Vec<u32>,
    pub data: Vec<u8>,
    pub kernel: Option<Vec<u8>>,
}

impl Rtw {
    pub fn count(&self) -> u64 {
        self.data.len() as u64 / dtype_width(self.dtype) as u64
    }
}

fn put_u16(w: &mut Vec<u8>, v: u16) {
    w.extend_from_slice(&v.to_le_bytes());
}
fn put_u32(w: &mut Vec<u8>, v: u32) {
    w.extend_from_slice(&v.to_le_bytes());
}
fn put_u64(w: &mut Vec<u8>, v: u64) {
    w.extend_from_slice(&v.to_le_bytes());
}

fn rd_u16(r: &mut &[u8]) -> io::Result<u16> {
    if r.len() < 2 {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "rtw: short u16",
        ));
    }
    let b: [u8; 2] = [r[0], r[1]];
    *r = &r[2..];
    Ok(u16::from_le_bytes(b))
}
fn rd_u32(r: &mut &[u8]) -> io::Result<u32> {
    if r.len() < 4 {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "rtw: short u32",
        ));
    }
    let b: [u8; 4] = [r[0], r[1], r[2], r[3]];
    *r = &r[4..];
    Ok(u32::from_le_bytes(b))
}
fn rd_u64(r: &mut &[u8]) -> io::Result<u64> {
    if r.len() < 8 {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "rtw: short u64",
        ));
    }
    let b: [u8; 8] = [r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7]];
    *r = &r[8..];
    Ok(u64::from_le_bytes(b))
}

/// Serialize an Rtw container to bytes.
///
/// Byte layout (little-endian), offsets relative to the container start:
///
/// ```text
/// [0..4]   magic    "RTW1"
/// [4..6]   u16      version (=1)
/// [6]      u8       kind
/// [7]      u8       dtype
/// [8..12]  u32      rank
/// [12..]   u32[rank] shape (rank * 4 bytes)
/// [..]     u64      data_len (byte length of data payload)
/// [..]     byte[]   data payload
/// [..]     u8       flags (0x01 = has_kernel)
/// [..]     u64      kernel_len + byte[] kernel   (only if flags & 0x01)
/// [..]     char[5]  "RTEND"
/// ```
pub fn encode(rtw: &Rtw) -> Vec<u8> {
    let mut w = Vec::new();
    w.extend_from_slice(b"RTW1"); // magic, offset 0..4
    put_u16(&mut w, 1); // version, offset 4..6
    w.push(rtw.kind); // kind, offset 6
    w.push(rtw.dtype); // dtype, offset 7
    put_u32(&mut w, rtw.shape.len() as u32); // rank, offset 8..12
    for &s in &rtw.shape {
        put_u32(&mut w, s); // shape[rank], offset 12..12+4*rank
    }
    put_u64(&mut w, rtw.data.len() as u64); // data_len (bytes), offset ...

    w.extend_from_slice(&rtw.data); // data payload
    let mut flags = 0u8;
    if rtw.kernel.is_some() {
        flags |= FLAG_HAS_KERNEL;
    }
    w.push(flags); // flags, offset ...
    if let Some(k) = &rtw.kernel {
        put_u64(&mut w, k.len() as u64); // kernel_len
        w.extend_from_slice(k); // kernel bytes
    }
    w.extend_from_slice(b"RTEND"); // end magic, offset ...
    w
}

/// Parse an Rtw container from bytes, validating magic and trimming trailing.
///
/// Mirrors `encode`'s layout (see the layout block there). Read order:
/// magic, version, kind, dtype, rank, shape, data_len, data, flags,
/// kernel (only when the has-kernel flag is set), RTEND.
pub fn decode(bytes: &[u8]) -> io::Result<Rtw> {
    if bytes.len() < 12 || &bytes[0..4] != b"RTW1" {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "rtw: bad or missing magic",
        ));
    }
    let mut r: &[u8] = &bytes[4..];
    let version = rd_u16(&mut r)?; // [4..6] version
    if version != 1 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("rtw: unsupported version {version}"),
        ));
    }
    if r.len() < 2 {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "rtw: missing kind/dtype",
        ));
    }
    let kind = r[0]; // [6] kind
    r = &r[1..];
    let dtype = r[0]; // [7] dtype
    r = &r[1..];
    let rank = rd_u32(&mut r)? as usize; // [8..12] rank
    if r.len() < rank * 4 {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "rtw: short shape",
        ));
    }
    let mut shape = Vec::with_capacity(cap_for(rank, 4, r.len()));
    for _ in 0..rank {
        shape.push(rd_u32(&mut r)?); // [12..] shape[rank]
    }
    let data_len = rd_u64(&mut r)? as usize; // data_len (bytes)
    if r.len() < data_len {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "rtw: short data",
        ));
    }
    let data = r[..data_len].to_vec(); // data payload
    r = &r[data_len..];
    if r.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "rtw: missing flags",
        ));
    }
    let flags = r[0]; // flags
    r = &r[1..];
    let mut kernel = None;
    if flags & FLAG_HAS_KERNEL != 0 {
        let klen = rd_u64(&mut r)? as usize; // kernel_len
        if r.len() < klen {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "rtw: short kernel",
            ));
        }
        kernel = Some(r[..klen].to_vec()); // kernel bytes
        r = &r[klen..];
    }
    // end magic (optional tolerance)
    if r.len() >= 5 && &r[..5] == b"RTEND" {
        // ok
    }
    Ok(Rtw {
        kind,
        dtype,
        shape,
        data,
        kernel,
    })
}

/// Read a whole file into bytes.
pub fn read_file(path: &std::path::Path) -> io::Result<Vec<u8>> {
    let mut f = std::fs::File::open(path)?;
    let mut buf = Vec::new();
    f.read_to_end(&mut buf)?;
    Ok(buf)
}

/// Write bytes to a file.
pub fn write_file(path: &std::path::Path, bytes: &[u8]) -> io::Result<()> {
    let mut f = std::fs::File::create(path)?;
    f.write_all(bytes)
}

// ============================================================================
// Structured payloads for kind=MODEL and kind=MEMORY (self-describing net payload).
// These are REAL round-trippable formats (not placeholders): a model carries
// metadata + named parameter tensors (shape/dtype/data) + optional Adam state;
// a memory carries a list of state fragments.
// ============================================================================

#[derive(Clone, Debug)]
pub struct NamedTensor {
    pub name: String,
    pub shape: Vec<u32>,
    pub dtype: u8,
    pub data: Vec<f32>,
}

#[derive(Clone, Debug, Default)]
pub struct OptState {
    pub m: Vec<Vec<f32>>,
    pub v: Vec<Vec<f32>>,
    pub t: u64,
}

#[derive(Clone, Debug, Default)]
pub struct Model {
    pub name: String,
    pub version: u32,
    pub params: Vec<NamedTensor>,
    pub opt: Option<OptState>,
}

#[derive(Clone, Debug)]
pub struct MemoryFragment {
    pub id: u64,
    pub state: Vec<f32>,
    pub strength: f32,
}

#[derive(Clone, Debug, Default)]
pub struct Memory {
    pub fragments: Vec<MemoryFragment>,
}

fn put_bytes(w: &mut Vec<u8>, b: &[u8]) {
    put_u32(w, b.len() as u32);
    w.extend_from_slice(b);
}
fn rd_bytes(r: &mut &[u8]) -> io::Result<Vec<u8>> {
    let n = rd_u32(r)? as usize;
    if r.len() < n {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "rtw: short bytes",
        ));
    }
    let b = r[..n].to_vec();
    *r = &r[n..];
    Ok(b)
}
fn put_str(w: &mut Vec<u8>, s: &str) {
    put_bytes(w, s.as_bytes());
}
fn rd_str(r: &mut &[u8]) -> io::Result<String> {
    let b = rd_bytes(r)?;
    String::from_utf8(b).map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "rtw: bad utf8"))
}
fn put_f32s(w: &mut Vec<u8>, v: &[f32]) {
    let mut b = Vec::with_capacity(v.len() * 4);
    for x in v {
        b.extend_from_slice(&x.to_le_bytes());
    }
    put_bytes(w, &b);
}
fn rd_f32s(r: &mut &[u8]) -> io::Result<Vec<f32>> {
    let b = rd_bytes(r)?;
    if b.len() % 4 != 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "rtw: bad f32 block",
        ));
    }
    Ok(b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

/// Encode a Model into a self-describing byte payload (for RTW kind=MODEL).
pub fn encode_model(m: &Model) -> Vec<u8> {
    let mut w = Vec::new();
    put_str(&mut w, &m.name);
    put_u32(&mut w, m.version);
    put_u32(&mut w, m.params.len() as u32);
    for p in &m.params {
        put_str(&mut w, &p.name);
        put_u32(&mut w, p.shape.len() as u32);
        for &d in &p.shape {
            put_u32(&mut w, d);
        }
        w.push(p.dtype);
        put_f32s(&mut w, &p.data);
    }
    match &m.opt {
        Some(o) => {
            w.push(1);
            put_u64(&mut w, o.t);
            for list in [&o.m, &o.v] {
                put_u32(&mut w, list.len() as u32);
                for t in list {
                    put_f32s(&mut w, t);
                }
            }
        }
        None => w.push(0),
    }
    w
}

/// A conservative pre-allocation capacity for a list whose length was read from
/// untrusted payload bytes. `count` is the claimed element count; `min_entry` is
/// a hard lower bound (bytes) each element needs. The returned capacity never
/// exceeds what `remaining` bytes could possibly hold, so a hostile/truncated
/// payload cannot drive a multi-GB `with_capacity` allocation (which would abort
/// the process). The loop still validates each field via bounds checks, so an
/// implausible count just means the entries fail to parse — never a huge alloc.
///
/// `min_entry` is a per-element lower bound in bytes; pass 0 to disable the clamp.
///
/// Safety: the divide only runs when `min_entry > 0` (a 0 min_entry makes
/// `count*0 = 0` never exceed `remaining`, so `count` is returned instead), so
/// there is no division by zero.
fn cap_for(count: usize, min_entry: usize, remaining: usize) -> usize {
    // Clamp a claimed element count against the bytes actually remaining. If
    // min_entry is 0, count*0 = 0 never exceeds `remaining`, so `count` is kept
    // (no clamp). Otherwise, if count*min_entry overflows or can't fit in
    // `remaining`, clamp to what `remaining` bytes could hold — which keeps a
    // hostile huge count from driving a giant `with_capacity`/`abort`. The divide
    // only runs when min_entry > 0 (a 0 min_entry yields the `count` branch), so
    // no division by zero.
    if count.checked_mul(min_entry).is_none_or(|needed| needed > remaining) {
        remaining / min_entry
    } else {
        count
    }
}

/// Decode a Model payload produced by encode_model.
pub fn decode_model(bytes: &[u8]) -> io::Result<Model> {
    let mut r: &[u8] = bytes;
    let name = rd_str(&mut r)?;
    let version = rd_u32(&mut r)?;
    let nparams = rd_u32(&mut r)? as usize;
    // Clamp capacity to what the remaining bytes could plausibly hold (a hostile
    // count must not drive a huge with_capacity). Each param needs >= ~4 bytes.
    let mut params = Vec::with_capacity(cap_for(nparams, 4, r.len()));
    for _ in 0..nparams {
        let pname = rd_str(&mut r)?;
        let rank = rd_u32(&mut r)? as usize;
        if r.len() < rank * 4 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "rtw: short shape",
            ));
        }
        let mut shape = Vec::with_capacity(rank);
        for _ in 0..rank {
            shape.push(rd_u32(&mut r)?);
        }
        if r.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "rtw: short dtype",
            ));
        }
        let dtype = r[0];
        r = &r[1..];
        let data = rd_f32s(&mut r)?;
        params.push(NamedTensor {
            name: pname,
            shape,
            dtype,
            data,
        });
    }
    if r.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "rtw: short opt flag",
        ));
    }
    let has_opt = r[0];
    r = &r[1..];
    let opt = if has_opt != 0 {
        let t = rd_u64(&mut r)?;
        let mut read_lists = |r: &mut &[u8]| -> io::Result<Vec<Vec<f32>>> {
            let n = rd_u32(r)? as usize;
            let mut out = Vec::with_capacity(cap_for(n, 4, r.len()));
            for _ in 0..n {
                out.push(rd_f32s(r)?);
            }
            Ok(out)
        };
        let m = read_lists(&mut r)?;
        let v = read_lists(&mut r)?;
        Some(OptState { m, v, t })
    } else {
        None
    };
    Ok(Model {
        name,
        version,
        params,
        opt,
    })
}

/// Encode a Memory into a self-describing byte payload (for RTW kind=MEMORY).
pub fn encode_memory(m: &Memory) -> Vec<u8> {
    let mut w = Vec::new();
    put_u32(&mut w, m.fragments.len() as u32);
    for f in &m.fragments {
        put_u64(&mut w, f.id);
        put_f32s(&mut w, &f.state);
        w.extend_from_slice(&f.strength.to_le_bytes());
    }
    w
}

/// Decode a Memory payload produced by encode_memory.
pub fn decode_memory(bytes: &[u8]) -> io::Result<Memory> {
    let mut r: &[u8] = bytes;
    let n = rd_u32(&mut r)? as usize;
    // Clamp capacity so a hostile fragment count can't drive a huge alloc; each
    // fragment needs >= 8 bytes (id + state-len + strength).
    let mut frags = Vec::with_capacity(cap_for(n, 8, r.len()));
    for _ in 0..n {
        let id = rd_u64(&mut r)?;
        let state = rd_f32s(&mut r)?;
        if r.len() < 4 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "rtw: short strength",
            ));
        }
        let s: [u8; 4] = [r[0], r[1], r[2], r[3]];
        let strength = f32::from_le_bytes(s);
        r = &r[4..];
        frags.push(MemoryFragment {
            id,
            state,
            strength,
        });
    }
    Ok(Memory { fragments: frags })
}

/// Wrap a Model into a full RTW container (kind=MODEL).
pub fn model_rtw(m: &Model) -> Rtw {
    Rtw {
        kind: KIND_MODEL,
        dtype: DTYPE_BYTES,
        shape: vec![m.params.len() as u32],
        data: encode_model(m),
        kernel: None,
    }
}

/// Wrap a Memory into a full RTW container (kind=MEMORY).
pub fn memory_rtw(m: &Memory) -> Rtw {
    Rtw {
        kind: KIND_MEMORY,
        dtype: DTYPE_BYTES,
        shape: vec![m.fragments.len() as u32],
        data: encode_memory(m),
        kernel: None,
    }
}
