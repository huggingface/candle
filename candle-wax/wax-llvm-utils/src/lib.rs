//! LLVM bitcode, read back by LLVM itself.

use llvm_sys::bit_reader::LLVMParseBitcodeInContext2;
use llvm_sys::core::*;
use llvm_sys::prelude::*;
use llvm_sys::LLVMAttributeFunctionIndex;
use std::ffi::CStr;

/// A declaration or definition, as LLVM read it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Func {
    pub name: String,
    /// 0 for a declaration.
    pub blocks: u32,
    pub attr_kinds: Vec<u32>,
}

impl Func {
    pub fn is_declaration(&self) -> bool {
        self.blocks == 0
    }
    /// Does the function carry the named enum attribute, e.g. `convergent`?
    ///
    /// Panics on a name LLVM does not know.
    pub fn has_attr(&self, name: &str) -> bool {
        let kind =
            unsafe { LLVMGetEnumAttributeKindForName(name.as_ptr() as *const _, name.len()) };
        assert!(kind != 0, "LLVM does not know an attribute called `{name}`");
        self.attr_kinds.contains(&kind)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Global {
    pub name: String,
    /// The printed value type, e.g. `[1056 x float]`.
    pub ty: String,
    pub addrspace: u32,
}

/// Everything the assertions ask about a lowered kernel.
#[derive(Debug, Clone)]
pub struct Parsed {
    pub triple: String,
    pub funcs: Vec<Func>,
    pub globals: Vec<Global>,
}

impl Parsed {
    pub fn declarations(&self) -> impl Iterator<Item = &Func> {
        self.funcs.iter().filter(|f| f.is_declaration())
    }
    pub fn definitions(&self) -> impl Iterator<Item = &Func> {
        self.funcs.iter().filter(|f| !f.is_declaration())
    }
    pub fn global(&self, name: &str) -> Option<&Global> {
        self.globals.iter().find(|g| g.name == name)
    }
}

unsafe fn name_of(v: LLVMValueRef) -> String {
    let mut n = 0;
    let p = LLVMGetValueName2(v, &mut n);
    CStr::from_ptr(p).to_string_lossy().into_owned()
}

unsafe fn cstr(p: *mut std::os::raw::c_char) -> String {
    let s = CStr::from_ptr(p).to_string_lossy().into_owned();
    LLVMDisposeMessage(p);
    s
}

/// Parse LLVM bitcode. An `Err` here means LLVM could not read the LLVM a backend wrote.
pub fn parse(bc: &[u8]) -> Result<Parsed, String> {
    unsafe {
        let ctx = LLVMContextCreate();
        let buf = LLVMCreateMemoryBufferWithMemoryRange(
            bc.as_ptr() as *const _,
            bc.len(),
            c"m".as_ptr(),
            0,
        );
        let mut m: LLVMModuleRef = std::ptr::null_mut();
        if LLVMParseBitcodeInContext2(ctx, buf, &mut m) != 0 {
            LLVMContextDispose(ctx);
            return Err("LLVM could not parse the bitcode we emitted".into());
        }

        let triple = CStr::from_ptr(LLVMGetTarget(m))
            .to_string_lossy()
            .into_owned();

        let mut globals = Vec::new();
        let mut g = LLVMGetFirstGlobal(m);
        while !g.is_null() {
            globals.push(Global {
                name: name_of(g),
                ty: cstr(LLVMPrintTypeToString(LLVMGlobalGetValueType(g))),
                addrspace: LLVMGetPointerAddressSpace(LLVMTypeOf(g)),
            });
            g = LLVMGetNextGlobal(g);
        }

        let mut funcs = Vec::new();
        let mut f = LLVMGetFirstFunction(m);
        while !f.is_null() {
            // Enumerate the function's own attribute set by index, which is how a named
            // attribute like `convergent` is reachable through the C API.
            let n = LLVMGetAttributeCountAtIndex(f, LLVMAttributeFunctionIndex);
            let mut raw = vec![std::ptr::null_mut(); n as usize];
            if n > 0 {
                LLVMGetAttributesAtIndex(f, LLVMAttributeFunctionIndex, raw.as_mut_ptr());
            }
            let attr_kinds = raw
                .into_iter()
                .filter(|a| !a.is_null())
                .map(|a| LLVMGetEnumAttributeKind(a))
                .collect();
            funcs.push(Func {
                name: name_of(f),
                blocks: LLVMCountBasicBlocks(f),
                attr_kinds,
            });
            f = LLVMGetNextFunction(f);
        }

        LLVMContextDispose(ctx);
        Ok(Parsed {
            triple,
            funcs,
            globals,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::parse;

    #[test]
    fn bytes_that_are_not_bitcode_are_an_error_not_a_panic() {
        assert!(parse(b"").is_err());
        assert!(parse(b"MTLB and then some").is_err());
        assert!(parse(&[0u8; 64]).is_err());
    }
}
