use crate::{Compiler, MacroOptions, Result, ShaderKind, SourceLanguage};
use heck::ToSnakeCase;
use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use std::{
    cell::Cell,
    env, fs,
    io::Write,
    iter::Iterator,
    path::{Path, PathBuf},
    process::{self, Command, Stdio},
    sync::atomic::{AtomicUsize, Ordering},
};

pub(super) fn compile_shader(
    options: &MacroOptions,
    source: &str,
    working_dir: &Path,
    shader_kind: Option<ShaderKind>,
    source_language: Option<SourceLanguage>,
    entry_point: Option<&str>,
    compiler: Option<Compiler>,
    macro_defines: &[(String, String)],
) -> Result<(Vec<u8>, Vec<String>), String> {
    let source_language = source_language.unwrap_or(options.global_source_language);
    let compiler = compiler
        .or(options.global_compiler)
        .unwrap_or(source_language.default_compiler());

    let mut command = Command::new(compiler.as_command());

    command.current_dir(working_dir);

    let vulkano_include_dir = init_include_dir()?;
    let temp_dir = &TempDir::new()?.0;
    let dependencies_file = temp_dir.join("deps.d");

    match compiler {
        Compiler::Shaderc => {
            command.arg("-x").arg(source_language.as_str());
            let target_env = options.vulkan_version.as_shaderc_target_env();
            command.arg(format!("--target-env={}", target_env));
            let target_spv = options.spirv_version.as_shaderc_target_spv();
            command.arg(format!("--target-spv={}", target_spv));

            if let Some(shader_kind) = shader_kind {
                command.arg(format!("-fshader-stage={}", shader_kind.as_shaderc_stage()));
            }

            if let Some(entry_point) = entry_point {
                command.arg(format!("-fentry-point={}", entry_point));
            }

            // vulkano.glsl dir first, then user include directories.
            command.arg("-I").arg(&vulkano_include_dir);
            set_common_options(&mut command, options, macro_defines);

            command.arg("-MD");
            command.arg("-MF").arg(&dependencies_file);
            command.arg("-o").arg("-");
            command.arg("-");
        }
        Compiler::Slangc => {
            command.arg("-lang").arg(source_language.as_str());
            command.arg("-target").arg("spirv");
            let profile = options.spirv_version.as_slangc_profile();
            command.arg("-profile").arg(profile);

            if let Some(shader_kind) = shader_kind {
                command.arg("-stage").arg(shader_kind.as_slangc_stage());
            }

            if let Some(entry_point) = entry_point {
                command.arg("-entry").arg(entry_point);
            }

            // Make sure the matrix majorness is consistent across languages.
            command.arg("-matrix-layout-column-major");

            // slangc renames an entry point specified by the `-entry` option to `main` unless this
            // option is specified for some reason.
            command.arg("-fvk-use-entrypoint-name");

            // vulkano.glsl dir first, working dir for module imports, then user include
            // directories.
            command.arg("-I").arg(&vulkano_include_dir);
            command.arg("-I").arg(working_dir);
            set_common_options(&mut command, options, macro_defines);

            command.arg("-depfile").arg(&dependencies_file);
            command.arg("--");
            command.arg("-");
        }
    }

    let mut child = command
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("failed to call {}: {e}", compiler.as_command()))?;

    child
        .stdin
        .take()
        .ok_or_else(|| format!("failed to open {} stdin", compiler.as_command()))?
        .write_all(source.as_bytes())
        .map_err(|e| format!("failed to write to {} stdin: {e}", compiler.as_command()))?;

    let output = child
        .wait_with_output()
        .map_err(|e| format!("failed to wait for {}: {e}", compiler.as_command()))?;

    if !output.status.success() {
        return Err(format!(
            "{} failed:\n{}",
            compiler.as_command(),
            String::from_utf8_lossy(&output.stderr),
        ));
    }

    let content = &fs::read_to_string(&dependencies_file)
        .map_err(|e| format!("failed to read dependencies file: {e}"))?;
    let input_files = parse_deps_file(content, &vulkano_include_dir, working_dir).map_err(|e| {
        let content = content
            .lines()
            .flat_map(|line| ["    ", line])
            .collect::<String>();

        format!("failed to parse dependencies file: {e}\nfile content:\n{content}")
    })?;

    Ok((output.stdout, input_files))
}

fn init_include_dir() -> Result<PathBuf, String> {
    // The include directory only needs to be written anew when the contents have changed, so we
    // key the temp dir on the hash of its contents.
    let hash = &include_str!("../include-sha256sums.sha256")[0..64];
    let path = env::temp_dir().join(format!("vulkano-shaders-include-{hash}"));

    // If the path proper exists, we know that all include files must have finished being written.
    if path.exists() {
        return Ok(path);
    }

    // Otherwise, we have to write the include files to a different, unique directory such that we
    // don't race with other invocations of the macro.
    let process_id = process::id();
    let thread_id = thread_id();
    let frag_path = path.with_extension(format!("frag-{process_id}-{thread_id}"));

    fs::create_dir_all(&frag_path)
        .map_err(|e| format!("failed to create vulkano include dir: {e}"))?;
    fs::write(
        frag_path.join("vulkano.glsl"),
        include_str!("../include/vulkano.glsl"),
    )
    .map_err(|e| format!("failed to write vulkano.glsl: {e}"))?;
    fs::write(
        frag_path.join("vulkano.slang"),
        include_str!("../include/vulkano.slang"),
    )
    .map_err(|e| format!("failed to write vulkano.slang: {e}"))?;

    // Only after writing is done can we attempt to rename the directory to the path proper.
    if let Err(e) = fs::rename(&frag_path, &path) {
        let _ = fs::remove_dir_all(frag_path);

        // If the path exists this time around, another process/thread has beaten us to the punch.
        if !path.exists() {
            return Err(format!("failed to create vulkano include dir: {e}"));
        }
    }

    Ok(path)
}

struct TempDir(PathBuf);

impl TempDir {
    fn new() -> Result<Self, String> {
        let process_id = process::id();
        let thread_id = thread_id();
        let path = env::temp_dir().join(format!("vulkano-shaders-{process_id}-{thread_id}"));

        fs::create_dir_all(&path).map_err(|e| format!("failed to create vulkano temp dir: {e}"))?;

        Ok(TempDir(path))
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn thread_id() -> usize {
    static COUNTER: AtomicUsize = AtomicUsize::new(0);

    thread_local! {
        static CURRENT: Cell<Option<usize>> = const { Cell::new(None) };
    }

    if let Some(id) = CURRENT.get() {
        return id;
    }

    let id = COUNTER
        .try_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
        .expect("we somehow managed to have more than `usize::MAX` threads");

    CURRENT.set(Some(id));

    id
}

fn set_common_options(
    command: &mut Command,
    options: &MacroOptions,
    macro_defines: &[(String, String)],
) {
    if cfg!(feature = "shaderc-debug") {
        command.arg("-g");
    }

    for dir in &options.include_directories {
        command.arg("-I").arg(dir);
    }

    for (name, value) in &options.global_macro_defines {
        command.arg(format!("-D{name}={value}"));
    }

    for (name, value) in macro_defines {
        command.arg(format!("-D{name}={value}"));
    }
}

/// Parses a Makefile-format dependency file produced by glslc `-MF`, returning the list of
/// included file paths. The format is `target: source dep1 dep2 ...`.
fn parse_deps_file(
    content: &str,
    vulkano_dir: &Path,
    working_dir: &Path,
) -> Result<Vec<String>, String> {
    fn take_while(input: &str, predicate: impl FnMut(char) -> bool) -> (&str, &str) {
        let index = input.len() - input.trim_start_matches(predicate).len();

        input.split_at(index)
    }

    fn take_until(input: &str, predicate: impl FnMut(char) -> bool) -> (&str, &str) {
        let Some(index) = input.find(predicate) else {
            return (input, "");
        };

        input.split_at(index)
    }

    fn is_space(c: char) -> bool {
        c == ' '
    }

    fn is_space_or_eol(c: char) -> bool {
        c == ' ' || c == '\n'
    }

    fn found(input: &str) -> String {
        if let Some(c) = input.chars().next() {
            format!("`{}`", c.escape_default())
        } else {
            "EOF".to_owned()
        }
    }

    let column = |input: &str| content.len() - input.len() + 1;

    // Everything after the first `:` is in the format `source dep1 dep2 ...`.
    let Some((start, input)) = content.split_once(':') else {
        return Ok(Vec::new());
    };

    if start.contains('\n') {
        return Err("expected no new lines before the `:`".to_owned());
    }

    // Skip the source.
    let Some(input) = input.strip_prefix(" <stdin>") else {
        let column = column(input);
        let found = found(input);
        return Err(format!("1:{column}: expected ` <stdin>`, found {found}"));
    };

    if input.starts_with('\n') {
        return Ok(Vec::new());
    }

    let Some(input) = input.strip_prefix(' ') else {
        let column = column(input);
        let found = found(input);
        return Err(format!("1:{column}: expected `\\n` or ` `, found {found}"));
    };

    let mut paths = Vec::new();
    let mut path = String::new();
    let mut input = input;

    // HACK: The Makefile format doesn't support path quoting, and shaderc just outputs paths that
    // contain spaces unquoted. Meaning that we don't know if a space is an actual delimiter or
    // part of a path. What we do is that we treat a file extension followed by a single space as
    // an actual delimiter, and treat any other space as part of a path. This rules out file and/or
    // directory names that contain an extension followed by space(s), as well as file names
    // without an extension.
    loop {
        let (spaces, rest) = take_while(input, is_space);
        let (token, rest) = take_until(rest, is_space_or_eol);
        let has_extension = Path::new(token).extension().is_some();

        path.push_str(spaces);
        path.push_str(token);

        if has_extension {
            let normalized = {
                let mut result = normalize_str(path.replace("\\:", ":"));
                if !Path::new(&result).is_absolute() {
                    result =
                        normalize_str(PathBuf::from_iter([working_dir, &PathBuf::from(result)]))
                }
                result
            };

            if !Path::new(&normalized).starts_with(vulkano_dir) {
                if !Path::new(&normalized).is_file() {
                    return Err(format!(
                        "expected `{normalized}` to be a file existing on the file system",
                    ));
                }

                paths.push(normalized);
            }

            path = String::new();
        }

        input = rest;

        if input.starts_with('\n') {
            if !path.is_empty() {
                let column = column(input);
                return Err(format!(
                    "1:{column}: expected a file extension, found `\\n`",
                ));
            }

            break;
        }

        if has_extension {
            input = input.strip_prefix(' ').unwrap();
        }
    }

    Ok(paths)
}

fn normalize_str(path: impl AsRef<Path>) -> String {
    fn inner(path: &Path) -> String {
        path.components()
            .collect::<PathBuf>()
            .into_os_string()
            .into_string()
            .unwrap()
    }

    inner(path.as_ref())
}

pub(super) fn generate_shaders(
    name: Option<&str>,
    words: &[u32],
    input_paths: Vec<String>,
) -> Result<TokenStream> {
    let include_bytes = input_paths.into_iter().map(|s| {
        quote! {
            // Using `include_bytes` here ensures that changing the shader will force recompilation.
            // The bytes themselves can be optimized out by the compiler as they are unused.
            ::std::include_bytes!( #s )
        }
    });

    let load_name = if let Some(name) = &name {
        format_ident!("load_{}", name.to_snake_case())
    } else {
        format_ident!("load")
    };
    let try_load_name = format_ident!("try_{load_name}");
    let load_unchecked_name = format_ident!("{load_name}_unchecked");
    let words_name = format_ident!("{}_WORDS", load_name.to_string().to_uppercase());

    let shaders_code = quote! {
        const _: &[&[u8]] = &[ #( #include_bytes ),* ];

        static #words_name: &[u32] = &[ #( #words ),* ];

        /// Loads the shader as a `ShaderModule`, panicking on a validation error.
        #[allow(unsafe_code)]
        #[inline]
        #[track_caller]
        pub unsafe fn #load_name(
            device: &::std::sync::Arc<::vulkano::device::Device>,
        ) -> ::std::result::Result<
            ::std::sync::Arc<::vulkano::shader::ShaderModule>,
            ::vulkano::VulkanError,
        > {
            match unsafe { #try_load_name(device) } {
                ::std::result::Result::Ok(shader_module) => {
                    ::std::result::Result::Ok(shader_module)
                }
                ::std::result::Result::Err(err) => {
                    ::std::result::Result::Err(::vulkano::Validated::unwrap(err))
                }
            }
        }

        /// Loads the shader as a `ShaderModule`.
        #[allow(unsafe_code)]
        #[inline]
        pub unsafe fn #try_load_name(
            device: &::std::sync::Arc<::vulkano::device::Device>,
        ) -> ::std::result::Result<
            ::std::sync::Arc<::vulkano::shader::ShaderModule>,
            ::vulkano::Validated<::vulkano::VulkanError>,
        > {
            unsafe {
                ::vulkano::shader::ShaderModule::try_new(
                    device,
                    &::vulkano::shader::ShaderModuleCreateInfo::new(#words_name),
                )
            }
        }

        /// Loads the shader as a `ShaderModule`, skipping validation.
        #[allow(unsafe_code)]
        #[inline]
        pub unsafe fn #load_unchecked_name(
            device: &::std::sync::Arc<::vulkano::device::Device>,
        ) -> ::std::result::Result<
            ::std::sync::Arc<::vulkano::shader::ShaderModule>,
            ::vulkano::VulkanError,
        > {
            unsafe {
                ::vulkano::shader::ShaderModule::new_unchecked(
                    device,
                    &::vulkano::shader::ShaderModuleCreateInfo::new(#words_name),
                )
            }
        }
    };

    Ok(shaders_code)
}
