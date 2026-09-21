use pliron::builtin::given_names::erase_given_names;
use pliron::context::{Context, Ptr};
use pliron::irfmt::parsers::spaced;
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::parsable::parse_from_str;
use pliron::printable::Printable;
use pliron::result::Result;
use wax_ir::types::{ScalarType, TileElementType, TileType, Type};

fn tile(e: ScalarType) -> Type {
    Type::Tile(TileType {
        shape: vec![],
        element_type: TileElementType::Scalar(e),
    })
}

fn reparse(text: &str, ctx: &mut Context) -> Result<Ptr<Operation>> {
    parse_from_str(spaced(Operation::top_level_parser()), ctx, text)
}

fn reprint(op: Ptr<Operation>, ctx: &Context) -> String {
    erase_given_names(ctx, op);
    op.disp(ctx).to_string()
}

#[test]
fn module_print_parse_round_trips_correctly() -> Result<()> {
    let mut m = wax_ir::Module::new("my_module");
    let tv = Type::TensorView(wax_ir::types::TensorViewType {
        element_type: ScalarType::F32,
        shape: vec![wax_ir::types::DYNAMIC, 64],
        strides: vec![64, 1],
    });
    m.begin_function("k", &[tile(ScalarType::I32), tv]);

    let text = m.module_op().get_operation().disp(&m.ctx).to_string();
    assert!(text.starts_with("builtin.module @my_module"));
    assert!(text.contains("tile<i32>"));
    assert!(text.contains("tensor_view<?x64xf32, strides [64, 1]>"));
    assert!(text.contains("^entry_"));

    let reparsed = reparse(&text, &mut m.ctx)?;
    let direct = reparsed.disp(&m.ctx).to_string();
    assert!(direct.starts_with("builtin.module @my_module"));
    assert!(direct.contains("tile<i32>"));
    assert!(direct.contains("tensor_view<?x64xf32, strides [64, 1]>"));
    assert!(direct.contains("^entry_"));

    let mut ctx = Context::new();
    let reparsed = reparse(&text, &mut ctx)?;
    let first = reprint(reparsed, &ctx);

    let mut ctx = Context::new();
    let reparsed = reparse(&first, &mut ctx)?;
    let second = reprint(reparsed, &ctx);

    assert_eq!(first, second, "\n{first}\nnot equal to\n{second}");

    Ok(())
}
