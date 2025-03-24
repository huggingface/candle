## Kernel Files (`.pwgsl`)  

The WGPU kernels are located in the `candle-wgpu-kernels` crate as `.pwgsl` files.  
A `.pwgsl` file is a WGSL shader file that is preprocessed using a C-like preprocessor, which supports:  

### File Inclusion  
```c
#include "FILENAME"  
```
Inserts the content of another file at this location.  

### Macro Definitions  

#### Simple Defines  
```c
#define KEY 42  
```
Replaces all occurrences of `KEY` with `42`. Only whole words are replaced:  
- ✅ `if KEY == 42` → `if 42 == 42`  
- ❌ `if KEYS == 42` (unchanged, as `KEYS` is not a whole match)  

#### Function-Like Defines  
```c
#define MAX(a, b) (a > b) ? a : b  
```
Allows function-like macros. Example:  
```c
MAX(2+2, 42)  
```
Expands to:  
```c
(2+2 > 42) ? 2+2 : 42  
```  

#### Computed Defines  
```c
#definec DEFINE_NAME EXPRESSION  
```
A computed define evaluates the given mathematical expression at compile time. Example:  
```c
#define KEY 42  
#definec MyDefine (42 + KEY)  
let x = MyDefine;  
```
Expands to:  
```c
let x = 84;  
```
Since `42 + KEY` → `42 + 42` → `84`.  

### Conditional Compilation  

The preprocessor supports the following C-like conditional directives:  

```c
#if CONDITION           // Evaluates to true if CONDITION is met  
#ifdef DEFINE_NAME      // Evaluates to true if DEFINE_NAME is defined  
#ifndef DEFINE_NAME     // Evaluates to true if DEFINE_NAME is NOT defined  
#elif CONDITION         // Alternative condition if the previous condition fails  
#elifdef DEFINE_NAME    // Alternative condition if DEFINE_NAME is defined  
#elifndef DEFINE_NAME   // Alternative condition if DEFINE_NAME is NOT defined  
#endif                  // Ends the conditional block  
```

This preprocessor enhances `.pwgsl` files by enabling more dynamic and reusable shader code. 🚀  