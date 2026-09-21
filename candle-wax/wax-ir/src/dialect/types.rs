//! The `wax` vocabulary in pliron.
//!
//! We do not re-model the entire cutile MLIR in pliron. Instead we take existing cutile-rs
//! concepts, and create pliron types from them.
//!
//! When interfacing with the DSL pliron's type system gives us some implicit confidence in correctness, and it let's us easily
//! parse and print for debugging purposes.
// TODO: Add improved verification (over verifier = "succ").
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
        f.write_str(&crate::dialect::text::print(&self.ty))
    }
}

impl Parsable for TileTy {
    type Arg = ();
    type Parsed = TypedHandle<TileTy>;

    fn parse<'a>(
        state_stream: &mut StateStream<'a>,
        _arg: Self::Arg,
    ) -> ParseResult<'a, Self::Parsed> {
        use pliron::combine::{self, Parser, many1, satisfy};
        use pliron::input_err;
        use pliron::parsable::{IntoParseResult, StreamLocation};

        let mut depth = 0i32;
        let mut prev = '\0';
        many1::<String, _, _>(satisfy(move |c: char| {
            let before = core::mem::replace(&mut prev, c);
            match c {
                '<' | '[' | '(' => {
                    depth += 1;
                    true
                }
                // `->` guard
                '>' if before == '-' => true,
                '>' | ']' | ')' if depth > 0 => {
                    depth -= 1;
                    true
                }
                ',' => depth > 0,
                '>' | ']' | ')' => false,
                c if c.is_whitespace() => depth > 0,
                _ => true,
            }
        }))
        .then(|raw: String| {
            combine::parser(move |ss: &mut StateStream<'a>| {
                match crate::dialect::text::parse(raw.trim()) {
                    Some(ty) => Ok(TileTy::get(ss.state.ctx, ty)).into_parse_result(),
                    None => input_err!(ss.loc(), "not a wax type: {}", raw).into_parse_result(),
                }
            })
        })
        .parse_stream(state_stream)
        .into_result()
    }
}

/// Move a type into the dialect.
///
/// For wax a `Type::Func` actually represents the entire module structure (not your typical function),
/// and so we don't create a pliron type representation for it.
pub fn to_pliron(ctx: &Context, ty: &Type) -> Option<TypeHandle> {
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

/// Add dialect spelling of `ScalarType`.
/// Not a pliron type, but we embed them within pliron types.
impl Printable for crate::types::ScalarType {
    fn fmt(
        &self,
        _ctx: &Context,
        _state: &printable::State,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        f.write_str(crate::dialect::text::scalar_name(self))
    }
}

impl Parsable for crate::types::ScalarType {
    type Arg = ();
    type Parsed = Self;

    fn parse<'a>(
        state_stream: &mut StateStream<'a>,
        _arg: Self::Arg,
    ) -> ParseResult<'a, Self::Parsed> {
        use pliron::combine::{Parser, many1, satisfy};
        use pliron::input_err;
        use pliron::parsable::{IntoParseResult, StreamLocation};

        let loc = state_stream.loc();
        let (text, _) = many1::<String, _, _>(satisfy(|c: char| c.is_ascii_alphanumeric()))
            .parse_stream(&mut *state_stream)
            .into_result()?;
        match crate::dialect::text::parse_scalar(&text) {
            Some(s) => Ok(s).into_parse_result(),
            None => input_err!(loc, "`{}` is not a scalar type", text)?,
        }
    }
}
