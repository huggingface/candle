//! Attributes of an op.
use crate::Attribute;
use pliron::context::Context;
use pliron::parsable::{Parsable, ParseResult, StateStream};
use pliron::printable::{self, Printable};

// TODO: This just nests wax attributes inside pliron attributes for no reason. Flatten.
#[pliron::derive::pliron_attr(name = "wax.attrs", verifier = "succ")]
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct WaxAttrs(pub Vec<(String, Attribute)>);

impl Printable for WaxAttrs {
    fn fmt(
        &self,
        _ctx: &Context,
        _state: &printable::State,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        write!(f, "{:?}", self.0)
    }
}

impl Parsable for WaxAttrs {
    type Arg = ();
    type Parsed = Self;
    fn parse<'a>(
        _state_stream: &mut StateStream<'a>,
        _arg: Self::Arg,
    ) -> ParseResult<'a, Self::Parsed> {
        unimplemented!("the `wax` dialect is not parsed from text")
    }
}
