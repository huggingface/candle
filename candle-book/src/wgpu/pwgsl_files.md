## Kernel Files (`.pwgsl`)

The WGPU kernels are located in the `candle-wgpu-kernels` crate as `.pwgsl` files.

A `.pwgsl` file is a WGSL shader file that is preprocessed using a C-like preprocessor.
This allows writing reusable and configurable shader code using includes, macros, and conditional compilation.

---

## File Inclusion

```c
#include "FILENAME"
```

Inserts the contents of another file at this location.

---

## Macro Definitions

### Simple Defines

```c
#define KEY 42
```

Replaces all occurrences of `KEY` with `42`. Only whole identifiers are replaced:

* ✅ `if KEY == 42` → `if 42 == 42`
* ❌ `if KEYS == 42` (unchanged, `KEYS` is not a full match)

---

### Function-Like Defines

```c
#define MAX(a, b) (a > b) ? a : b
```

Allows function-like macros.

Example:

```c
MAX(2+2, 42)
```

Expands to:

```c
(2+2 > 42) ? 2+2 : 42
```

---

### Computed Defines

```c
#definec DEFINE_NAME EXPRESSION
```

A computed define evaluates a mathematical expression at preprocess time.

Example:

```c
#define KEY 42
#definec MyDefine (42 + KEY)
let x = MyDefine;
```

Expands to:

```c
let x = 84;
```

---

## Conditional Compilation

The preprocessor supports the following C-like conditional directives:

```c
#if CONDITION            // True if CONDITION evaluates to non-zero
#ifdef DEFINE_NAME       // True if DEFINE_NAME is defined
#ifndef DEFINE_NAME      // True if DEFINE_NAME is NOT defined
#elif CONDITION          // Alternative condition
#elifdef DEFINE_NAME     // Alternative if DEFINE_NAME is defined
#elifndef DEFINE_NAME   // Alternative if DEFINE_NAME is NOT defined
#endif                   // Ends the conditional block
```

---

## Multi-Line Preprocessor Blocks (`#pp_begin` / `#pp_end`)

In addition to normal `#define`, the preprocessor supports **multi-line macro blocks** using:

```c
#pp_begin DEFINENAME(data, tid)
    ...
#pp_end
```

This defines a macro named `DEFINENAME` with parameters (`data`, `tid`) whose body can span **multiple lines**.

Key properties:

* Works like a function-like `#define`, but allows multi-line bodies.
* The body may contain **other preprocessor directives** (`#if`, `#define`, `#include`, etc.).
* All inner preprocessing is executed using the **context and state at the time the macro is expanded**, not when it is defined.

This makes `#pp_begin` useful for:

* Defining reusable shader kernels or code templates
* Generating complex control flow or binding logic
* Sharing blocks of code that depend on compile-time configuration