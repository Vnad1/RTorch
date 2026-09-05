//! RTW whitelist — restrict which `.rtw` files RTorch is willing to load.
//!
//! The whitelist is a JSON config file (`{"allowed": ["C:\\abs\\a.rtw", ...]}`)
//! whose entries are **exact absolute paths** to `.rtw` files. RTorch refuses to
//! load any `.rtw` whose canonical path is not on the whitelist. This is the
//! *path-level* gate, applied before `rtw::read_file`/`rtw::decode`, so an
//! off-list `.rtw` never reaches the decoder.
//!
//! Security model: the whitelist is **default-deny**. If no config is provided
//! (the `RTORCH_WHITELIST` env var is unset), the whitelist is empty and every
//! `.rtw` is rejected. A caller must explicitly trust a path by listing it.
//!
//! JSON parsing is hand-written (no serde / third-party deps) so RTorch stays
//! dependency-free. It supports exactly the shape this module reads:
//! a top-level object with an `"allowed"` key whose value is an array of strings.
//! Unknown keys are ignored; strings are the only values extracted.

use crate::error::{RtorchError, Result};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Default)]
pub struct Whitelist {
    /// Canonical absolute paths that may be loaded.
    allowed: Vec<PathBuf>,
}

impl Whitelist {
    /// Load a whitelist from `RTORCH_WHITELIST` (the JSON file path). If the env
    /// var is unset, returns an **empty** whitelist (default-deny — nothing is
    /// allowed). Any error reading/parsing the file is surfaced as a
    /// `WhitelistError` (it must not silently fall open).
    pub fn from_env() -> Result<Whitelist> {
        match std::env::var("RTORCH_WHITELIST") {
            Ok(p) if !p.is_empty() => Whitelist::load(Path::new(&p)),
            _ => Ok(Whitelist::default()),
        }
    }

    /// Load a whitelist JSON file. `{}` / missing `allowed` yields an empty list.
    pub fn load(path: &Path) -> Result<Whitelist> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| RtorchError::whitelist(format!("read {}: {e}", path.display())))?;
        Whitelist::load_from_text(&text)
    }

    /// Load a whitelist from an in-memory JSON string (also used by tests).
    pub fn load_from_text(text: &str) -> Result<Whitelist> {
        let allowed = parse_whitelist_json(text).map_err(RtorchError::whitelist)?;
        Ok(Whitelist { allowed })
    }

    /// Whether `path` is allowed. Comparison is against canonicalized absolute
    /// paths, so a relative or differently-formed input still matches if it
    /// resolves to the same file. On Windows, path comparison is case-insensitive.
    pub fn is_allowed(&self, path: &Path) -> bool {
        if self.allowed.is_empty() {
            return false;
        }
        let target = normalize(path);
        let target = match target {
            Some(t) => t,
            None => return false,
        };
        self.allowed.iter().any(|a| paths_equal(&strip_verbatim(a.clone()), &target))
    }

    /// Number of allowed paths (useful for error messages / introspection).
    pub fn len(&self) -> usize {
        self.allowed.len()
    }

    pub fn is_empty(&self) -> bool {
        self.allowed.is_empty()
    }
}

/// Canonicalize a path to an absolute form; returns None if it can't be resolved
/// (e.g. the file doesn't exist yet — still resolvable for a non-existent path by
/// making it absolute relative to cwd).
fn normalize(path: &Path) -> Option<PathBuf> {
    let abs = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir().ok()?.join(path)
    };
    // Canonicalize to resolve symlinks / `..`, but strip the Windows verbatim
    // `\\?\` prefix that canonicalize returns — otherwise a whitelist entry written
    // as a normal `D:\...` path never matches a `\\?\D:\...` target.
    let canon = std::fs::canonicalize(&abs);
    match canon {
        Ok(c) => Some(strip_verbatim(c)),
        Err(_) => Some(abs),
    }
}

/// Strip the Windows "\\?\" verbatim prefix (and any trailing separator) that
/// `std::fs::canonicalize` may produce, so path comparison is consistent with the
/// plain paths users write in the whitelist.
#[cfg(windows)]
fn strip_verbatim(p: PathBuf) -> PathBuf {
    let s = p.to_string_lossy();
    if let Some(rest) = s.strip_prefix(r"\\?\") {
        PathBuf::from(rest)
    } else {
        p
    }
}

#[cfg(not(windows))]
fn strip_verbatim(p: PathBuf) -> PathBuf {
    p
}

#[cfg(windows)]
fn paths_equal(a: &Path, b: &Path) -> bool {
    a.to_string_lossy().eq_ignore_ascii_case(&b.to_string_lossy())
}

#[cfg(not(windows))]
fn paths_equal(a: &Path, b: &Path) -> bool {
    a == b
}

// ---------------------------------------------------------------------------
// Hand-written JSON parsing (minimal).
// ---------------------------------------------------------------------------

/// Parse the whitelist JSON and return the list of allowed path strings.
///
/// We support exactly the shape we read:
///   `{ "allowed": [ "C:\\abs\\a.rtw", "C:\\abs\\b.rtw" ] }`
/// The scanner looks for the `"allowed"` key, then the following `[`, and
/// collects every string literal inside that array. Other keys/values are
/// ignored. Loosely valid JSON is required, but unknown content is skipped.
fn parse_whitelist_json(text: &str) -> std::result::Result<Vec<PathBuf>, String> {
    let bytes = text.as_bytes();
    let mut i = 0;

    // 1) Find the `"allowed"` key (a `"allowed"` string followed by `:`).
    let key = b"allowed";
    let mut pos = None;
    while i < bytes.len() {
        if bytes[i] == b'"' {
            let (val, next) = read_json_string(bytes, i)?;
            if val.as_bytes() == key {
                // Ensure the next non-ws char is ':'
                let mut j = next;
                while j < bytes.len() && is_ws(bytes[j]) {
                    j += 1;
                }
                if j < bytes.len() && bytes[j] == b':' {
                    pos = Some(j);
                    break;
                }
            }
            i = next;
        } else {
            i += 1;
        }
    }
    let colon = match pos {
        Some(c) => c,
        None => return Ok(Vec::new()), // no `allowed` key -> empty list
    };

    // 2) Skip ws + ':' then find the '[' opening the allowed array.
    let mut j = colon + 1;
    while j < bytes.len() && is_ws(bytes[j]) {
        j += 1;
    }
    if j >= bytes.len() || bytes[j] != b'[' {
        return Err("expected '[' after \"allowed\"".to_string());
    }
    j += 1;

    // 3) Collect every string literal inside the array.
    let mut paths = Vec::new();
    while j < bytes.len() {
        match bytes[j] {
            b'"' => {
                let (val, next) = read_json_string(bytes, j)?;
                paths.push(PathBuf::from(val));
                j = next;
            }
            b']' => break, // end of allowed array
            _ => j += 1,
        }
    }
    Ok(paths)
}

fn is_ws(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\n' | b'\r')
}

fn read_json_string(bytes: &[u8], start: usize) -> std::result::Result<(String, usize), String> {
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
            return Err("unterminated escape".to_string());
        }
        // Decode a (possibly multi-byte, non-ASCII) UTF-8 char instead of reading
        // a single byte as a char — a literal byte cast would corrupt paths with
        // non-ASCII characters.
        match std::str::from_utf8(&bytes[i..]) {
            Ok(s) => {
                let ch = s.chars().next().unwrap_or('\u{FFFD}');
                let adv = ch.len_utf8();
                out.push(ch);
                i += adv;
            }
            Err(_) => {
                // Invalid UTF-8 byte: keep it verbatim to avoid panicking.
                out.push(bytes[i] as char);
                i += 1;
            }
        }
    }
    Err("unterminated string".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_allowed_array() {
        let json = r#"{ "allowed": ["C:\\abs\\a.rtw", "C:\\abs\\b.rtw"] }"#;
        let paths = parse_whitelist_json(json).expect("parse");
        assert_eq!(paths.len(), 2);
        // Note: `\\` in JSON becomes a single `\` in the decoded string.
        assert_eq!(paths[0], PathBuf::from("C:\\abs\\a.rtw"));
        assert_eq!(paths[1], PathBuf::from("C:\\abs\\b.rtw"));
    }

    #[test]
    fn empty_when_no_allowed_key() {
        assert!(parse_whitelist_json(r#"{ "other": 1 }"#).expect("parse").is_empty());
        assert!(parse_whitelist_json(r#"[]"#).expect("parse").is_empty());
        assert!(parse_whitelist_json(r#""#).expect("parse").is_empty());
    }

    #[test]
    fn ignores_other_keys_before_allowed() {
        let json = r#"{ "version": 1, "allowed": ["C:\\x.rtw"] }"#;
        let paths = parse_whitelist_json(json).expect("parse");
        assert_eq!(paths, vec![PathBuf::from("C:\\x.rtw")]);
    }

    #[test]
    fn unquoted_or_bad_string_is_error() {
        // A genuinely unterminated string (no closing quote before EOF) is an
        // error, not a silent empty/shorted list.
        assert!(parse_whitelist_json(r#"{ "allowed": ["C:\\x"#).is_err());
    }

    #[test]
    fn default_deny_when_empty() {
        let wl = Whitelist::default();
        assert!(wl.is_empty());
        assert!(!wl.is_allowed(Path::new("C:\\anything.rtw")));
    }

    #[test]
    fn exact_path_match_after_normalize() {
        // Write a real temp file so canonicalize succeeds, then confirm an exact
        // path matches while a sibling does not.
        let dir = std::env::temp_dir().join(format!("wl_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let a = dir.join("a.rtw");
        let b = dir.join("b.rtw");
        std::fs::write(&a, b"RTW1").unwrap();
        std::fs::write(&b, b"RTW1").unwrap();
        let a_canon = std::fs::canonicalize(&a).unwrap();

        // JSON strings must escape backslashes (`\` -> `\\`); the parser decodes
        // `\\` back to `\`. So build the nested path with escaped separators.
        let a_json = a_canon.display().to_string().replace('\\', "\\\\");
        let json = format!(r#"{{ "allowed": ["{}"] }}"#, a_json);
        let wl = Whitelist::load_from_text(&json).unwrap();
        assert!(wl.is_allowed(Path::new(&a_canon)));
        // A sibling file is not allowed.
        assert!(!wl.is_allowed(&b));
        // A relative path resolving to `a` also matches (normalization).
        let rel = dir.join("a.rtw");
        assert!(wl.is_allowed(&rel));
    }

    #[test]
    fn parses_non_ascii_path() {
        // A path with non-ASCII characters (e.g. Chinese) must round-trip intact.
        // JSON: { "allowed": ["C:\\模型\\文件.rtw"] } (Windows backslashes escaped).
        let json = "{\"allowed\":[\"C:\\\\模型\\\\文件.rtw\"]}";
        let paths = parse_whitelist_json(json).expect("parse");
        assert_eq!(paths, vec![PathBuf::from("C:\\模型\\文件.rtw")]);
    }

    #[test]
    fn is_allowed_matches_existing_file_path() {
        // Replicate the CLI formula-DLL case: the whitelist stores a real file's
        // canonical path; querying it (absolute or relative) must match.
        let repo = concat!(env!("CARGO_MANIFEST_DIR"));
        let dll = std::path::Path::new(repo).join("delta.dll");
        if !dll.exists() {
            return; // formula dll not present in this checkout; skip
        }
        let canon = std::fs::canonicalize(&dll).unwrap();
        // Store the canonical path (which may carry a `\\?\` verbatim prefix) and
        // query it — is_allowed must match regardless of the prefix, so a whitelist
        // written by hand (plain `D:\...`) and the target normalize consistently.
        let json = format!(r#"{{ "allowed": ["{}"] }}"#, canon.display().to_string().replace('\\', "\\\\"));
        let wl = Whitelist::load_from_text(&json).unwrap();
        // Absolute query.
        assert!(wl.is_allowed(&canon), "absolute path must be allowed");
        // A plain (no verbatim prefix) absolute query must also match: the target
        // normalizes consistently even if the whitelist entry carries `\\?\`.
        let plain = canon.to_string_lossy().replace(r"\\?\", "");
        let plain = std::path::PathBuf::from(plain);
        assert!(wl.is_allowed(&plain), "plain absolute path must be allowed");
    }
}
