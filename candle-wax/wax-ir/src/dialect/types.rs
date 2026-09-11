//! The `wax` vocabulary in pliron.
//!
//! We do not re-model the entire cutile MLIR in pliron. Instead we use existing cutile-rs concepts, and
//! use those to create pliron directly.
//!
//! When interfacing with the DSL pliron's type system gives us some implicit confidence in correctness, and it let's us easily
//! parse and print for debugging purposes. More importantly we can verify the parsing with `succ`.
use crate::Type;
use pliron::context::Context;
use pliron::derive::pliron_type;
use pliron::parsable::{Parsable, ParseResult, StateStream};
use pliron::printable::{self, Printable};
use pliron::r#type::{TypeHandle, TypedHandle};

/// A tile-level type.
#[pliron_type(name = "wax.ty", generate_get = true, verifier = "succ")]
#[derive(Hash, PartialEq, Eq, Debug)]
pub struct TileTy {
    ty: Type,
}

impl TileTy {
    pub fn ty(&self) -> &Type {
        &self.ty
    }
}

impl Printable for TileTy {
    fn fmt(
        &self,
        _ctx: &Context,
        _state: &printable::State,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        write!(f, "{:?}", self.ty)
    }
}

impl Parsable for TileTy {
    type Arg = ();
    type Parsed = TypedHandle<TileTy>;
    fn parse<'a>(
        _state_stream: &mut StateStream<'a>,
        _arg: Self::Arg,
    ) -> ParseResult<'a, Self::Parsed> {
        unimplemented!("the `wax` dialect is not parsed from text")
    }
}

/// Move a type into the dialect.
///
/// For wax a `Type::Func` actually represents the entire module structure (not your typical function),
/// and so we don't create a pliron type representation for it.
pub fn to_pliron(ctx: &mut Context, ty: &Type) -> Option<TypeHandle> {
    if matches!(ty, Type::Func(_)) {
        return None;
    }
    Some(TileTy::get(ctx, ty.clone()).into())
}

/// Move a type out of the dialect.
pub fn from_pliron(ctx: &Context, h: TypeHandle) -> Option<Type> {
    TypedHandle::<TileTy>::from_handle(h, ctx)
        .ok()
        .map(|t| t.deref(ctx).ty().clone())
}
