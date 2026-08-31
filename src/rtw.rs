// RTorch Workfile (.rtw) — a self-contained container for results and compute
// kernels. Not tied to PyTorch/any runtime: pure little-endian binary that any
// language can read/write. Two kinds (v1):
//   result : packed data tensor(s) -> shape/dtype/data
//   kernel : embedded compute kernel (GLSL source) -> runnable by `rtorch x.rtw`
// Format:
//   magic  "RTW1" (4B)
//   u16 version
//   u8  kind    (0=result, 1=kernel)
//   u8  dtype   (0=fp32,1=fp16,2=fp8,3=fp4,4=int32,5=bytes)
//   u32 rank
//   u32 shape[rank]
//   u64 count
//   bytes data[count*width]         (kind=result)
//   u8  flags   (0x01 = has_kernel)
//   u64 klen  + bytes kernel        (if flags & 0x01)
//   magic  "RTEND" (5B)
use std::io::{self, Read, Write};

pub const DTYPE_FP32: u8 = 0;
pub const DTYPE_FP16: u8 = 1;
pub const DTYPE_FP8: u8 = 2;   // E4M3
pub const DTYPE_FP4: u8 = 3;   // E2M1
pub const DTYPE_INT32: u8 = 4;
pub const DTYPE_BYTES: u8 = 5;

pub const KIND_RESULT: u8 = 0;
pub const KIND_KERNEL: u8 = 1;
pub const KIND_MODEL: u8 = 2;   // 冻结权重(权重+结构元数据)
pub const KIND_MEMORY: u8 = 3;  // Striker 记忆 state(Frag 列表序列化, 动态增长)

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

fn put_u16(w: &mut Vec<u8>, v: u16) { w.extend_from_slice(&v.to_le_bytes()); }
fn put_u32(w: &mut Vec<u8>, v: u32) { w.extend_from_slice(&v.to_le_bytes()); }
fn put_u64(w: &mut Vec<u8>, v: u64) { w.extend_from_slice(&v.to_le_bytes()); }

fn rd_u16(r: &mut &[u8]) -> io::Result<u16> {
    if r.len() < 2 { return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "rtw: short u16")); }
    let b: [u8; 2] = [r[0], r[1]]; *r = &r[2..]; Ok(u16::from_le_bytes(b))
}
fn rd_u32(r: &mut &[u8]) -> io::Result<u32> {
    if r.len() < 4 { return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "rtw: short u32")); }
    let b: [u8; 4] = [r[0], r[1], r[2], r[3]]; *r = &r[4..]; Ok(u32::from_le_bytes(b))
}
fn rd_u64(r: &mut &[u8]) -> io::Result<u64> {
    if r.len() < 8 { return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "rtw: short u64")); }
    let b: [u8; 8] = [r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7]]; *r = &r[8..]; Ok(u64::from_le_bytes(b))
}

/// Serialize an Rtw container to bytes.
pub fn encode(rtw: &Rtw) -> Vec<u8> {
    let mut w = Vec::new();
    w.extend_from_slice(b"RTW1");
    put_u16(&mut w, 1); // version
    w.push(rtw.kind);
    w.push(rtw.dtype);
    put_u32(&mut w, rtw.shape.len() as u32);
    for &s in &rtw.shape {
        put_u32(&mut w, s);
    }
    put_u64(&mut w, rtw.data.len() as u64);
    w.extend_from_slice(&rtw.data);
    let mut flags = 0u8;
    if rtw.kernel.is_some() { flags |= FLAG_HAS_KERNEL; }
    w.push(flags);
    if let Some(k) = &rtw.kernel {
        put_u64(&mut w, k.len() as u64);
        w.extend_from_slice(k);
    }
    w.extend_from_slice(b"RTEND");
    w
}

/// Parse an Rtw container from bytes, validating magic and trimming trailing.
pub fn decode(bytes: &[u8]) -> io::Result<Rtw> {
    if bytes.len() < 12 || &bytes[0..4] != b"RTW1" {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "rtw: bad or missing magic"));
    }
    let mut r: &[u8] = &bytes[4..];
    let version = rd_u16(&mut r)?;
    if version != 1 {
        return Err(io::Error::new(io::ErrorKind::InvalidData, format!("rtw: unsupported version {version}")));
    }
    let kind = r[0]; r = &r[1..];
    let dtype = r[0]; r = &r[1..];
    let rank = rd_u32(&mut r)? as usize;
    if r.len() < rank * 4 { return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "rtw: short shape")); }
    let mut shape = Vec::with_capacity(rank);
    for _ in 0..rank {
        shape.push(rd_u32(&mut r)?);
    }
    let data_len = rd_u64(&mut r)? as usize;
    if r.len() < data_len { return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "rtw: short data")); }
    let data = r[..data_len].to_vec();
    r = &r[data_len..];
    let flags = r[0]; r = &r[1..];
    let mut kernel = None;
    if flags & FLAG_HAS_KERNEL != 0 {
        let klen = rd_u64(&mut r)? as usize;
        if r.len() < klen { return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "rtw: short kernel")); }
        kernel = Some(r[..klen].to_vec());
        r = &r[klen..];
    }
    // end magic (optional tolerance)
    if r.len() >= 5 && &r[..5] == b"RTEND" {
        // ok
    }
    Ok(Rtw { kind, dtype, shape, data, kernel })
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
