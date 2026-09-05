//! Interactive trust gate for loading `.rtw` and formula DLLs.
//!
//! This is a **CLI-level** trust decision (used by `rtorch.exe`), NOT a library
//! gate. The library (`rtw::decode`, `formula::run`) stays pure and never spawns
//! anything. When `rtorch.exe` is about to load a `.rtw` or a formula DLL, it asks:
//!
//!   1. **Whitelist first** — if the path is listed in `RTORCH_WHITELIST`, allow
//!      it silently (no prompt).
//!   2. **Remembered** — if this process already trusted the same path, allow it.
//!   3. **Otherwise** — spawn a temporary `cmd.exe` window that prints
//!      "Do you trust this library?" (from <path>, name <name>) and reads y/n.
//!      The window closes when the user answers; the main program continues.
//!      `y` trusts (and remembers), `n` rejects.
//!
//! The trust set is per-process only (a `RefCell<HashSet<PathBuf>>`), so a path is
//! prompted at most once per process run.

use crate::whitelist::Whitelist;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

#[derive(Debug)]
pub struct TrustGate {
    whitelist: Option<Whitelist>,
    remembered: Mutex<std::collections::HashSet<PathBuf>>,
}

impl Default for TrustGate {
    fn default() -> Self {
        TrustGate {
            whitelist: Whitelist::from_env().ok(),
            remembered: Mutex::new(std::collections::HashSet::new()),
        }
    }
}

impl TrustGate {
    /// Build a gate, loading the whitelist from env (if any).
    pub fn from_env() -> Self {
        TrustGate::default()
    }

    /// Decide whether `path` may be loaded. Returns `Ok(true)` if trusted,
    /// `Ok(false)` if the user answered `n` (or never got a chance to answer),
    /// and `Err` if the trust check itself failed (e.g. can't spawn the prompt).
    ///
    /// `kind` is a human label ("RTW container" or "formula DLL") shown in the
    /// prompt. `name` is the file name and `from` the full path.
    pub fn check(&self, path: &Path, kind: &str) -> std::result::Result<bool, String> {
        // 1) Whitelist first.
        if let Some(wl) = &self.whitelist {
            if !wl.is_empty() && wl.is_allowed(path) {
                return Ok(true);
            }
        }
        // 2) Remembered in this process.
        if self.is_remembered(path) {
            return Ok(true);
        }
        // 3) Prompt (temporary cmd window). Print the library identity safely (to
        // stderr, not a shell) so the user knows what they're authorizing; the cmd
        // window only asks the fixed y/n question (no injection surface).
        let name = path
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| path.display().to_string());
        let from = path.display().to_string();
        eprintln!("[rtorch] ask: trust this {kind}? From: {from}, Name: {name}");
        match ask_trust_cmd() {
            Ok(true) => {
                self.remember(path);
                Ok(true)
            }
            Ok(false) => Ok(false),
            Err(e) => Err(e),
        }
    }

    fn is_remembered(&self, path: &Path) -> bool {
        if let Ok(guard) = self.remembered.lock() {
            guard.contains(path)
        } else {
            false
        }
    }

    fn remember(&self, path: &Path) {
        if let Ok(mut guard) = self.remembered.lock() {
            guard.insert(path.to_path_buf());
        }
    }
}

/// Spawn a temporary `cmd.exe` window that asks y/n and returns the answer.
///
/// SECURITY: the cmd script contains **only fixed text** — `from`, `name`, and
/// `kind` are never interpolated into the command line, so a hostile path or file
/// name cannot inject shell metacharacters (`&`, `|`, `^`, etc.). The identity of
/// the library is printed by the caller (via `eprintln!`, which is not a shell) so
/// the user still sees what they are being asked to trust.
///
/// The window closes when the user answers; only `y`/`Y` is accepted as "yes", and
/// the result is surfaced through the process exit code (`exit /b 0` for yes,
/// `exit /b 1` otherwise).
///
/// Non-interactive safety: if `RTORCH_TRUST_PROMPT` is set to `0`, the interactive
/// prompt is skipped and the answer is "no" (fail-closed). This is used for
/// non-TTY / pipeline / automated contexts where a human cannot answer — the
/// safe default is to reject rather than hang or silently allow.
fn ask_trust_cmd() -> std::result::Result<bool, String> {
    if std::env::var("RTORCH_TRUST_PROMPT").map(|v| v == "0").unwrap_or(false) {
        return Ok(false); // non-interactive: reject
    }
    // `kind`, `from`, `name` are all printed to stderr by `check` (not a shell).
    // Here the cmd script is **fully static** — no caller-supplied bytes touch it,
    // so there is no command-injection surface.
    let script =
        "echo Do you trust this library? (y/n)& set /p ANS=^>& echo ^%ANS^%|findstr /i \"^y\" >nul && (exit /b 0)& exit /b 1";

    #[cfg(windows)]
    {
        let status = std::process::Command::new("cmd.exe")
            .arg("/c")
            .arg(script)
            .status()
            .map_err(|e| format!("spawn trust prompt: {e}"))?;
        // exit 0 = yes (y/Y), anything else = no.
        return Ok(status.code() == Some(0));
    }

    #[cfg(not(windows))]
    {
        let _ = script;
        Err("trust prompt only supported on Windows".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: a gate with an in-memory whitelist (no env dependency).
    fn gate_with_whitelist(json: &str) -> TrustGate {
        let wl = crate::whitelist::Whitelist::load_from_text(json).expect("whitelist");
        TrustGate {
            whitelist: Some(wl),
            remembered: Mutex::new(std::collections::HashSet::new()),
        }
    }

    #[test]
    fn whitelisted_path_is_allowed_without_prompt() {
        // A whitelisted path is allowed even when the prompt is disabled and
        // hasn't been remembered — proving whitelist is checked first.
        let dev_null = std::path::Path::new("C:\\trust_test\\allowed.rtw");
        let json = r#"{"allowed":["C:\\trust_test\\allowed.rtw"]}"#;
        let gate = gate_with_whitelist(json);
        assert!(gate.check(dev_null, "RTW container").unwrap());
    }

    #[test]
    fn non_whitelisted_prompt_disabled_rejects() {
        // Non-whitelisted + prompt disabled (non-interactive) -> reject.
        let dev_null = std::path::Path::new("C:\\trust_test\\other.rtw");
        let json = r#"{"allowed":["C:\\trust_test\\allowed.rtw"]}"#;
        let gate = gate_with_whitelist(json);
        // Simulate non-interactive by not setting RTORCH_TRUST_PROMPT pointer...
        // We can't run cmd.exe in a unit test; but the env guard is what we test:
        // with prompt disabled, ask_trust_cmd returns false.
        unsafe { std::env::set_var("RTORCH_TRUST_PROMPT", "0") };
        let r = gate.check(dev_null, "RTW container").unwrap();
        assert!(!r, "non-whitelisted non-interactive must reject");
        unsafe { std::env::remove_var("RTORCH_TRUST_PROMPT") };
    }

    #[test]
    fn empty_whitelist_is_not_full_deny_when_promptable() {
        // An empty whitelist still allows the remember path; but here with the
        // prompt disabled it rejects (fail-closed). This documents the default.
        let dev_null = std::path::Path::new("C:\\anything.rtw");
        let gate = TrustGate::default();
        unsafe { std::env::set_var("RTORCH_TRUST_PROMPT", "0") };
        assert!(!gate.check(dev_null, "RTW container").unwrap());
        unsafe { std::env::remove_var("RTORCH_TRUST_PROMPT") };
    }
}
