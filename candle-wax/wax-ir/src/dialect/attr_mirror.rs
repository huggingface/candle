//! Attributes of an op.
use crate::Attribute;
use pliron::context::Context;
use pliron::parsable::{Parsable, ParseResult, StateStream};
use pliron::printable::{self, Printable};

/// Newtype that lets us insert our attributes into pliron's `AttibuteDict`.
/// Lookup as `dict.get::<WaxAttr>(&key)`.
#[pliron::derive::pliron_attr(name = "wax.attr", verifier = "succ")]
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct WaxAttr(pub Attribute);

impl Printable for WaxAttr {
    fn fmt(
        &self,
        _ctx: &Context,
        _state: &printable::State,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        write!(f, "{:?}", self.0)
    }
}

impl Parsable for WaxAttr {
    type Arg = ();
    type Parsed = Self;
    fn parse<'a>(
        _state_stream: &mut StateStream<'a>,
        _arg: Self::Arg,
    ) -> ParseResult<'a, Self::Parsed> {
        unimplemented!("the `wax` dialect is not parsed from text")
    }
}
