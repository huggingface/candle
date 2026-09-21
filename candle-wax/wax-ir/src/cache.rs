//! Compiled wax IR module cache.
//!
//! A full frontend run is a relatively expensive step of the pipeline. We can cache the resulting
//! [`crate::Module`] based on its' inputs, here represented by [`CompileKey`].
//!
//! The cache is both frontend and backend neutral. The only requirement for caching a module is
//! that you provide the [`CompileKey`] and a closure/function that produces a
//! [`Result<crate::Module, _>`].
//!
//! A [`crate::Module`] owns a pliron [`pliron::context::Context`] (which is not `Sync`), so we use
//! a thread-local static cache rather than something shared like a `Mutex`.
//!
//! Failed compilations are not cached.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

/// Unique key of a module.
#[derive(Hash, Eq, PartialEq, Clone, Debug)]
pub struct CompileKey {
    /// Compile-time SHA-256 of source. The producer closure handed to [`cached_module`] must
    /// compile the source this hash covers.
    pub source_hash: &'static str,
    /// The module name, as handed to the compiler.
    pub module: &'static str,
    /// The entry point this key compiles.
    pub entry: &'static str,
    /// Generic arguments, in declared order.
    pub generics: Vec<String>,
    /// Argument name to strides, in declared order.
    pub strides: Vec<(String, Vec<i32>)>,
}

thread_local! {
    /// Compiled modules, keyed on the frontend's inputs.
    static MODULES: RefCell<HashMap<CompileKey, Rc<crate::Module>>> =
        RefCell::new(HashMap::new());
}

/// Cached compilation of a module.
///
/// Cache uses the provided `key` for lookup.
/// `compile` runs only on a cache miss. On a hit the stored module is returned and the closure
/// is dropped without running.
/// A miss that errors stores nothing.
///
/// Cache is thread-local.
pub fn cached_module<F, E>(key: &CompileKey, compile: F) -> Result<Rc<crate::Module>, E>
where
    F: FnOnce() -> Result<crate::Module, E>,
{
    if let Some(m) = MODULES.with(|c| c.borrow().get(key).cloned()) {
        return Ok(m);
    }
    let m = compile()?;
    let rc = Rc::new(m);
    MODULES.with(|c| c.borrow_mut().insert(key.clone(), rc.clone()));
    Ok(rc)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    fn key(tag: &'static str) -> CompileKey {
        CompileKey {
            source_hash: "test-source",
            module: "kernels",
            entry: tag,
            generics: vec!["f16".into()],
            strides: vec![("z".into(), vec![8, 1])],
        }
    }

    #[test]
    fn module_is_cached_correctly() {
        let runs = Cell::new(0);
        let f = || -> Result<crate::Module, ()> {
            runs.set(runs.get() + 1);
            Ok(crate::Module::new("t"))
        };
        let k = key("a");
        let a = cached_module(&k, f).unwrap();
        let again = cached_module(&k, || Ok::<crate::Module, ()>(crate::Module::new("t"))).unwrap();
        assert_eq!(runs.get(), 1, "a hit must not recompile");
        assert_eq!(
            Rc::as_ptr(&a),
            Rc::as_ptr(&again),
            "a hit must be the same module"
        );
    }

    #[test]
    fn different_keys_compile_independently() {
        // different key is a different compilation, and keys never leak across.
        let runs = Cell::new(0);
        let a = cached_module(&key("a"), || -> Result<crate::Module, ()> {
            runs.set(runs.get() + 1);
            Ok(crate::Module::new("t"))
        })
        .unwrap();
        let b = cached_module(&key("b"), || -> Result<crate::Module, ()> {
            runs.set(runs.get() + 1);
            Ok(crate::Module::new("t"))
        })
        .unwrap();
        assert_eq!(runs.get(), 2);
        assert_ne!(Rc::as_ptr(&a), Rc::as_ptr(&b));
    }

    #[test]
    fn failed_compilation_is_not_cached() {
        let runs = Cell::new(0);
        // First run fails. Next call must retry compilation.
        let flaky = || {
            runs.set(runs.get() + 1);
            if runs.get() == 1 {
                Err("boom")
            } else {
                Ok(crate::Module::new("t"))
            }
        };
        let k = key("flaky");
        assert!(cached_module(&k, flaky).is_err());
        let m = cached_module(&k, flaky).expect("the retry succeeds and is stored");
        assert_eq!(runs.get(), 2);
        let again = cached_module(&k, || unreachable!("must be a hit") as Result<_, ()>).unwrap();
        assert_eq!(Rc::as_ptr(&m), Rc::as_ptr(&again));
    }
}
