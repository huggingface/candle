//! Performance profiler for WASM
//!
//! Tracks timing and memory usage across different parts of the model.

use std::cell::RefCell;
use std::collections::HashMap;
use wasm_bindgen::prelude::*;

thread_local! {
    static PROFILER: RefCell<Profiler> = RefCell::new(Profiler::new());
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct ProfileEntry {
    pub name: String,
    pub count: usize,
    pub total_ms: f64,
    pub min_ms: f64,
    pub max_ms: f64,
    pub avg_ms: f64,
    pub last_ms: f64,
}

pub struct Profiler {
    entries: HashMap<String, ProfileData>,
    enabled: bool,
    stack: Vec<(String, f64)>,
}

#[derive(Debug, Clone)]
struct ProfileData {
    count: usize,
    total_ms: f64,
    min_ms: f64,
    max_ms: f64,
    last_ms: f64,
}

impl Profiler {
    fn new() -> Self {
        Self {
            entries: HashMap::new(),
            enabled: true,
            stack: Vec::new(),
        }
    }

    fn start(&mut self, name: &str) {
        if !self.enabled {
            return;
        }
        let time = js_sys::Date::now();
        self.stack.push((name.to_string(), time));
    }

    fn end(&mut self, name: &str) {
        if !self.enabled {
            return;
        }

        let end_time = js_sys::Date::now();

        if let Some((start_name, start_time)) = self.stack.pop() {
            if start_name != name {
                web_sys::console::warn_1(
                    &format!(
                        "Profiler mismatch: expected '{}', got '{}'",
                        start_name, name
                    )
                    .into(),
                );
                return;
            }

            let elapsed = end_time - start_time;

            let entry = self.entries.entry(name.to_string()).or_insert(ProfileData {
                count: 0,
                total_ms: 0.0,
                min_ms: f64::INFINITY,
                max_ms: 0.0,
                last_ms: 0.0,
            });

            entry.count += 1;
            entry.total_ms += elapsed;
            entry.min_ms = entry.min_ms.min(elapsed);
            entry.max_ms = entry.max_ms.max(elapsed);
            entry.last_ms = elapsed;
        }
    }

    fn get_entries(&self) -> Vec<ProfileEntry> {
        let mut entries: Vec<_> = self
            .entries
            .iter()
            .map(|(name, data)| ProfileEntry {
                name: name.clone(),
                count: data.count,
                total_ms: data.total_ms,
                min_ms: data.min_ms,
                max_ms: data.max_ms,
                avg_ms: data.total_ms / data.count as f64,
                last_ms: data.last_ms,
            })
            .collect();

        entries.sort_by(|a, b| b.total_ms.partial_cmp(&a.total_ms).unwrap());
        entries
    }

    fn reset(&mut self) {
        self.entries.clear();
        self.stack.clear();
    }

    fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }
}

// Public API
pub fn profile_start(name: &str) {
    PROFILER.with(|p| p.borrow_mut().start(name));
}

pub fn profile_end(name: &str) {
    PROFILER.with(|p| p.borrow_mut().end(name));
}

pub fn profile_reset() {
    PROFILER.with(|p| p.borrow_mut().reset());
}

pub fn profile_set_enabled(enabled: bool) {
    PROFILER.with(|p| p.borrow_mut().set_enabled(enabled));
}

// RAII guard for automatic profiling
pub struct ProfileGuard {
    name: String,
}

impl ProfileGuard {
    pub fn new(name: &str) -> Self {
        profile_start(name);
        Self {
            name: name.to_string(),
        }
    }
}

impl Drop for ProfileGuard {
    fn drop(&mut self) {
        profile_end(&self.name);
    }
}

// Macro for easy profiling
#[macro_export]
macro_rules! profile_scope {
    ($name:expr) => {
        let _guard = $crate::profiler::ProfileGuard::new($name);
    };
}

// WASM exports
#[wasm_bindgen]
pub struct ProfileStats {
    entries: Vec<ProfileEntry>,
}

#[wasm_bindgen]
impl ProfileStats {
    #[wasm_bindgen(getter)]
    pub fn json(&self) -> String {
        serde_json::to_string(&self.entries).unwrap_or_default()
    }
}

#[wasm_bindgen]
pub fn profile_get_stats() -> ProfileStats {
    let entries = PROFILER.with(|p| p.borrow().get_entries());
    ProfileStats { entries }
}

#[wasm_bindgen]
pub fn profile_print_stats() {
    let entries = PROFILER.with(|p| p.borrow().get_entries());

    web_sys::console::log_1(&"".into());
    web_sys::console::log_1(&"═══════════════════════════════════════════════════════".into());
    web_sys::console::log_1(&"                    PERFORMANCE PROFILE                 ".into());
    web_sys::console::log_1(&"═══════════════════════════════════════════════════════".into());

    if entries.is_empty() {
        web_sys::console::log_1(&"No profiling data collected.".into());
        return;
    }

    let total_time: f64 = entries.iter().map(|e| e.total_ms).sum();

    web_sys::console::log_1(
        &format!(
            "{:<30} {:>8} {:>10} {:>10} {:>10} {:>10}",
            "Section", "Count", "Total(ms)", "Avg(ms)", "Min(ms)", "Max(ms)"
        )
        .into(),
    );
    web_sys::console::log_1(&"───────────────────────────────────────────────────────".into());

    for entry in &entries {
        let percent = (entry.total_ms / total_time) * 100.0;
        web_sys::console::log_1(
            &format!(
                "{:<30} {:>8} {:>10.2} {:>10.3} {:>10.3} {:>10.3}  ({:.1}%)",
                entry.name,
                entry.count,
                entry.total_ms,
                entry.avg_ms,
                entry.min_ms,
                entry.max_ms,
                percent
            )
            .into(),
        );
    }

    web_sys::console::log_1(&"───────────────────────────────────────────────────────".into());
    web_sys::console::log_1(&format!("TOTAL TIME: {:.2}ms", total_time).into());
    web_sys::console::log_1(&"═══════════════════════════════════════════════════════".into());
}

#[wasm_bindgen]
pub fn profile_enable(enabled: bool) {
    profile_set_enabled(enabled);
    if enabled {
        web_sys::console::log_1(&"✅ Profiler ENABLED".into());
    } else {
        web_sys::console::log_1(&"❌ Profiler DISABLED".into());
    }
}

#[wasm_bindgen]
pub fn profile_clear() {
    profile_reset();
    web_sys::console::log_1(&"Profiler CLEARED".into());
}

// Memory tracking
#[wasm_bindgen]
pub fn get_memory_info() -> String {
    let memory = web_sys::window()
        .and_then(|w| w.performance())
        .and_then(|p| js_sys::Reflect::get(&p, &"memory".into()).ok())
        .and_then(|m| {
            let used = js_sys::Reflect::get(&m, &"usedJSHeapSize".into())
                .ok()
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0);
            let total = js_sys::Reflect::get(&m, &"totalJSHeapSize".into())
                .ok()
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0);
            let limit = js_sys::Reflect::get(&m, &"jsHeapSizeLimit".into())
                .ok()
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0);
            Some((used, total, limit))
        });

    if let Some((used, total, limit)) = memory {
        format!(
            "Used: {:.2} MB / Total: {:.2} MB / Limit: {:.2} MB ({:.1}%)",
            used / 1_048_576.0,
            total / 1_048_576.0,
            limit / 1_048_576.0,
            (used / limit) * 100.0
        )
    } else {
        "Memory info not available".to_string()
    }
}

#[wasm_bindgen]
pub fn log_memory() {
    let info = get_memory_info();
    web_sys::console::log_1(&format!("Memory: {}", info).into());
}

#[wasm_bindgen]
pub fn get_wasm_memory_info() -> String {
    #[cfg(target_arch = "wasm32")]
    {
        let pages = core::arch::wasm32::memory_size(0);
        let bytes = pages as f64 * 65536.0;
        let mb = bytes / (1024.0 * 1024.0);

        format!("WASM Memory: {:.2} MB ({} pages of 64KB)", mb, pages)
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        "Not WASM".to_string()
    }
}

#[wasm_bindgen]
pub fn log_wasm_memory() {
    let info = get_wasm_memory_info();
    web_sys::console::log_1(&format!("{}", info).into());
}