/// The Rust DSL, loaded into `cutile::core` as a module.
use cutile_frontend::ast::Module;

pub fn core_module() -> Module {
    let mut module = cutile_dsl::core::__module_ast_self();
    module.set_absolute_path("cutile::core".to_string());
    module
}
