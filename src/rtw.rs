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

/// RTW format version (semver string). RTW is a format described by the RTW
/// protocol; this is the format version, independent of any RTorch release.
/// RTW is part of RTorch (no separate "RTW" product); the artifact's own
/// version and its "library it came from" live in the Manifest.
pub const RTW_FORMAT_VERSION: &str = "0.0.1";

/// Magic that marks the optional Manifest block at the head of the `data`
/// payload. A legacy RTW whose `data` does not start with this magic is treated
/// as having no Manifest (format v1), so an old RTW remains readable.
const MANIFEST_MAGIC: [u8; 4] = *b"RTMF";

/// The self-describing Manifest carried at the head of an RTW's `data` payload.
/// RTorch only reads this to answer three things — which library this artifact
/// came from, where it is located, and whether the RTW can parse itself into a
/// form the runtime understands. It does NOT resolve dependencies, download, or
/// interpret third-party semantics (those belong to an upper runtime/package
/// manager).
#[derive(Debug, Clone)]
pub struct Manifest {
    /// Stable, globally-unique artifact id (reverse-DNS namespace):
    /// e.g. "com.example.model". This is the "which library" answer.
    pub artifact_id: String,
    /// Where this artifact/file lives (path/URL), i.e. the "location" answer.
    pub location: String,
    /// Format version this artifact uses (defaults to RTW_FORMAT_VERSION).
    pub format_version: String,
    /// Capabilities this artifact requires/declares (optional, informational).
    pub requires: Vec<String>,
}

impl Default for Manifest {
    fn default() -> Self {
        Manifest {
            artifact_id: String::new(),
            location: String::new(),
            format_version: RTW_FORMAT_VERSION.to_string(),
            requires: Vec::new(),
        }
    }
}

/// Encode a Manifest as the JSON string carried at the head of `data`.
/// Hand-written (no serde/third-party deps). Produces a compact object:
///   {"artifact_id":..,"location":..,"format_version":..,"requires":[..]}
pub fn encode_manifest(m: &Manifest) -> Vec<u8> {
    let mut s = String::from("{");
    s.push_str("\"artifact_id\":");
    s.push_str(&json_str(&m.artifact_id));
    s.push_str(",\"location\":");
    s.push_str(&json_str(&m.location));
    s.push_str(",\"format_version\":");
    s.push_str(&json_str(&m.format_version));
    s.push_str(",\"requires\":[");
    for (i, r) in m.requires.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        s.push_str(&json_str(r));
    }
    s.push_str("]}");
    s.into_bytes()
}

/// Parse a Manifest JSON string back into a [`Manifest`]. Loose: unknown keys are
/// ignored (forward-compatible); only the fields RTorch understands are read.
/// Returns `Err` on malformed JSON structure.
pub fn parse_manifest(bytes: &[u8]) -> io::Result<Manifest> {
    let text = std::str::from_utf8(bytes)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "rtw: manifest not utf8"))?;
    let mut m = Manifest::default();
    parse_manifest_fields(text, &mut m)?;
    Ok(m)
}

/// A tiny JSON string literal encoder (escaping `"`, `\`, and common controls).
fn json_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn is_ws(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\n' | b'\r')
}

fn read_json_string(bytes: &[u8], start: usize) -> io::Result<(String, usize)> {
    // bytes[start] == '"'
    let mut out = String::new();
    let mut i = start + 1;
    while i < bytes.len() {
        let c = bytes[i] as char;
        if c == '"' {
            return Ok((out, i + 1));
        }
        if c == '\\' {
            if i + 1 < bytes.len() {
                let esc = bytes[i + 1] as char;
                let ch = match esc {
                    '"' => '"',
                    '\\' => '\\',
                    '/' => '/',
                    'n' => '\n',
                    't' => '\t',
                    'r' => '\r',
                    'b' => '\u{8}',
                    'f' => '\u{c}',
                    other => other,
                };
                out.push(ch);
                i += 2;
                continue;
            }
            return Err(io::Error::new(io::ErrorKind::InvalidData, "rtw: manifest bad escape"));
        }
        // Decode multi-byte UTF-8 char (non-ASCII ids/locations).
        match std::str::from_utf8(&bytes[i..]) {
            Ok(s) => {
                let ch = s.chars().next().unwrap_or('\u{FFFD}');
                out.push(ch);
                i += ch.len_utf8();
            }
            Err(_) => {
                out.push(bytes[i] as char);
                i += 1;
            }
        }
    }
    Err(io::Error::new(io::ErrorKind::UnexpectedEof, "rtw: manifest unterminated string"))
}

/// Scan a Manifest JSON object for the fields RTorch understands. Unknown keys
/// are ignored (forward-compatible); this reads `artifact_id`, `location`,
/// `format_version` (scalar strings) and `requires` (array of strings).
fn parse_manifest_fields(text: &str, m: &mut Manifest) -> io::Result<()> {
    let bytes = text.as_bytes();
    let mut i = 0;

    // Find each top-level `"key":` and decode its value.
    while i < bytes.len() {
        if bytes[i] != b'"' {
            i += 1;
            continue;
        }
        let (key, next) = read_json_string(bytes, i)?;
        i = next;
        // Find ':' after the key, then skip it.
        while i < bytes.len() && is_ws(bytes[i]) {
            i += 1;
        }
        if i >= bytes.len() || bytes[i] != b':' {
            break;
        }
        i += 1; // skip ':'
        while i < bytes.len() && is_ws(bytes[i]) {
            i += 1;
        }
        if i >= bytes.len() {
            break;
        }
        // The value is either a string (scalar) or an array (requires).
        if bytes[i] == b'"' && key != "requires" {
            let (val, nx) = read_json_string(bytes, i)?;
            match key.as_str() {
                "artifact_id" => m.artifact_id = val,
                "location" => m.location = val,
                "format_version" => m.format_version = val,
                _ => {}
            }
            i = nx;
        } else if bytes[i] == b'[' && key == "requires" {
            // Collect every string inside the requires array.
            let mut j = i + 1;
            while j < bytes.len() && bytes[j] != b']' {
                if bytes[j] == b'"' {
                    let (v, nx) = read_json_string(bytes, j)?;
                    m.requires.push(v);
                    j = nx;
                } else {
                    j += 1;
                }
            }
            i = j + 1;
        } else {
            // Skip to next comma / unknown structure.
            i += 1;
        }
    }
    Ok(())
}


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
    /// Optional self-describing Manifest carried at the head of `data`.
    pub manifest: Option<Manifest>,
}

impl Rtw {
    pub fn count(&self) -> u64 {
        self.data.len() as u64 / dtype_width(self.dtype) as u64
    }
}

impl Rtw {
    /// The self-describing Manifest, if present. RTorch only reads this to answer
    /// *which library* / *where* / *can RTW parse itself* — nothing more.
    pub fn manifest(&self) -> Option<&Manifest> {
        self.manifest.as_ref()
    }

    /// A short human description: the library this artifact came from (artifact_id),
    /// its location, and whether it parsed (i.e. RTW could "translate itself").
    pub fn describe(&self) -> String {
        match &self.manifest {
            Some(m) => format!(
                "artifact_id={} location={} format_version={}",
                if m.artifact_id.is_empty() { "<unknown>" } else { &m.artifact_id },
                if m.location.is_empty() { "<unknown>" } else { &m.location },
                if m.format_version.is_empty() { RTW_FORMAT_VERSION } else { &m.format_version }
            ),
            None => format!(
                "artifact_id=<legacy-no-manifest> location=<unknown> format_version=1"
            ),
        }
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

/// The theoretical maximum RTW artifact size allowed by the format/protocol
/// (32 TiB). This is a **format-level** capability — it does NOT mean a device
/// must be able to load 32 TiB, nor does it grant the decoder an unconditional
/// right to allocate that much. The decoder enforces stricter runtime bounds
/// (payload bytes + resource sanity) separately; this constant only caps what
/// the format can represent.
pub const RTW_MAX_SIZE: u64 = 32 * 1024 * 1024 * 1024 * 1024; // 32 TiB = 35,184,372,088,832

/// Validate an element `count` read from untrusted payload bytes before using it
/// as a pre-allocation capacity. The decoder never trusts a length field; it
/// walks a validation chain:
///
///   1. `count * min_bytes` must not overflow `usize` (integer-overflow check).
///   2. `count * min_bytes` must not exceed the format maximum `RTW_MAX_SIZE`
///      (a payload declaring more than the format can represent is rejected).
///   3. `count * min_bytes` must fit within the *actual* remaining payload bytes
///      (a count that implies more data than the artifact holds is a malformed /
///      hostile payload and is rejected — it is NOT clamped, so we can tell a
///      deliberately-oversized count from a legitimately large artifact).
///
/// On success returns the capacity (= `count`). On any check failing, returns an
/// error so the decoder rejects the payload instead of allocating.
///
/// `min_bytes == 0` disables the byte-size checks and returns `count` unchanged
/// (for fields the caller trusts to be bounded separately).
fn validate_capacity(count: usize, min_bytes: usize, remaining: usize) -> io::Result<usize> {
    if min_bytes == 0 {
        return Ok(count);
    }
    // 1. Integer overflow: count * min_bytes must be representable.
    let needed = count.checked_mul(min_bytes).ok_or_else(|| {
        io::Error::new(io::ErrorKind::InvalidData, "rtw: count * element_size overflow")
    })?;
    // 2. Format-level cap: the artifact cannot claim more than RTW_MAX_SIZE bytes.
    if needed as u64 > RTW_MAX_SIZE {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "rtw: payload exceeds RTW_MAX_SIZE (32 TiB)",
        ));
    }
    // 3. Payload boundary: the claimed data must fit in the actual remaining
    //    bytes. Oversized-but-not-overflowing counts are rejected here, which is
    //    how a hostile count is distinguished from a valid large artifact.
    if needed > remaining {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "rtw: count exceeds payload size",
        ));
    }
    Ok(count)
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
    // data_len covers the whole data region: the optional manifest block (if
    // any) plus the payload bytes.
    let manifest_block_len: usize = match &rtw.manifest {
        Some(m) => 8 + encode_manifest(m).len(), // magic(4) + len(4) + json
        None => 0,
    };
    put_u64(&mut w, (rtw.data.len() + manifest_block_len) as u64); // data_len (bytes), offset ...

    // The manifest (if any) is carried at the HEAD of the data payload, framed
    // with the MANIFEST_MAGIC. A legacy RTW (no manifest) has data that does not
    // start with MANIFEST_MAGIC, so it stays readable.
    if let Some(m) = &rtw.manifest {
        let json = encode_manifest(m);
        w.extend_from_slice(&MANIFEST_MAGIC);
        put_u32(&mut w, json.len() as u32);
        w.extend_from_slice(&json);
    }
    w.extend_from_slice(&rtw.data); // data payload (after optional manifest block)
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
    let mut shape = Vec::with_capacity(validate_capacity(rank, 4, r.len())?);
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
    // Extract the optional Manifest block at the head of `data` (if it carries
    // the MANIFEST_MAGIC). A legacy RTW's data does not, so it stays compatible.
    let (manifest, data) = split_manifest(data);

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
        manifest,
    })
}

/// If `data` starts with `MANIFEST_MAGIC`, split off the framed Manifest and
/// return `(Some(manifest), remaining_payload)`; otherwise `(None, data)`.
fn split_manifest(data: Vec<u8>) -> (Option<Manifest>, Vec<u8>) {
    if data.len() < 8 || &data[0..4] != &MANIFEST_MAGIC {
        return (None, data);
    }
    let mlen = u32::from_le_bytes([data[4], data[5], data[6], data[7]]) as usize;
    if data.len() < 8 + mlen {
        return (None, data); // truncated manifest marker; treat as no manifest
    }
    match parse_manifest(&data[8..8 + mlen]) {
        Ok(m) => (Some(m), data[8 + mlen..].to_vec()),
        Err(_) => (None, data), // malformed manifest; treat payload as-is
    }
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

/// Decode a Model payload produced by encode_model.
pub fn decode_model(bytes: &[u8]) -> io::Result<Model> {
    let mut r: &[u8] = bytes;
    let name = rd_str(&mut r)?;
    let version = rd_u32(&mut r)?;
    let nparams = rd_u32(&mut r)? as usize;
    let mut params = Vec::with_capacity(validate_capacity(nparams, 4, r.len())?);
    for _ in 0..nparams {
        let pname = rd_str(&mut r)?;
        let rank = rd_u32(&mut r)? as usize;
        if r.len() < rank * 4 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "rtw: short shape",
            ));
        }
        let mut shape = Vec::with_capacity(validate_capacity(rank, 4, r.len())?);
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
        let read_lists = |r: &mut &[u8]| -> io::Result<Vec<Vec<f32>>> {
            let n = rd_u32(r)? as usize;
            let mut out = Vec::with_capacity(validate_capacity(n, 4, r.len())?);
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
    let mut frags = Vec::with_capacity(validate_capacity(n, 8, r.len())?);
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
        manifest: None,
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
        manifest: None,
    }
}

#[cfg(test)]
mod rtw_decode_tests {
    use super::*;

    #[test]
    fn validate_capacity_overflow_rejected() {
        // count * min_bytes overflows usize -> Err (must not allocate).
        let big = usize::MAX / 4; // * 8 overflows on 64-bit
        let r = validate_capacity(big, 8, usize::MAX);
        assert!(r.is_err(), "overflow must be rejected");

        let r2 = validate_capacity(big, 0, usize::MAX); // min_bytes 0 -> no clamp
        assert_eq!(r2.unwrap(), big);
    }

    #[test]
    fn validate_capacity_rtw_max_size_rejected() {
        // count * min_bytes exceeds RTW_MAX_SIZE (32 TiB) -> Err.
        let count = (RTW_MAX_SIZE / 8) as usize + 1; // just over 32 TiB / 8
        let r = validate_capacity(count, 8, usize::MAX);
        assert!(r.is_err(), "over RTW_MAX_SIZE must be rejected");

        // At exactly RTW_MAX_SIZE (with min_bytes 1) is allowed if it fits.
        let r_ok = validate_capacity(RTW_MAX_SIZE as usize, 1, usize::MAX);
        assert!(r_ok.is_ok());
    }

    #[test]
    fn validate_capacity_payload_boundary_rejected() {
        // count fits in format cap but exceeds actual payload bytes -> Err.
        let r = validate_capacity(1000, 8, 16); // needs 8000 bytes, only 16 present
        assert!(r.is_err(), "count exceeding payload must be rejected");
    }

    #[test]
    fn validate_capacity_legit_large_allowed() {
        // A count that actually fits within the remaining payload is allowed
        // (a legitimately large artifact is not rejected just for being big).
        let r = validate_capacity(100, 8, 1000); // 100*8 = 800 <= 1000
        assert_eq!(r.unwrap(), 100);
    }
}
