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
        f.write_str(&super::text::print_attr(&self.0))
    }
}

impl Parsable for WaxAttr {
    type Arg = ();
    type Parsed = Self;

    fn parse<'a>(
        state_stream: &mut StateStream<'a>,
        _arg: Self::Arg,
    ) -> ParseResult<'a, Self::Parsed> {
        use pliron::combine::{self, Parser, many1, satisfy};
        use pliron::input_err;
        use pliron::parsable::IntoParseResult;
        use pliron::parsable::StreamLocation;

        let mut depth = 0i32;
        let mut in_str = false;
        let mut prev = '\0';
        many1::<String, _, _>(satisfy(move |c: char| {
            let before = core::mem::replace(&mut prev, c);
            match c {
                '"' => {
                    in_str = !in_str;
                    true
                }
                _ if in_str => true,
                '<' | '[' | '(' | '{' => {
                    depth += 1;
                    true
                }
                // `->` is not a closing bracket.
                '>' if before == '-' => true,
                '>' | ']' | ')' | '}' if depth > 0 => {
                    depth -= 1;
                    true
                }
                // Comma inside a nested construct not end.
                ',' => depth > 0,
                '>' | ']' | ')' | '}' => false,
                c if c.is_whitespace() => depth > 0,
                _ => true,
            }
        }))
        .then(|raw: String| {
            combine::parser(move |ss: &mut StateStream<'a>| {
                match super::text::parse_attr(raw.trim()) {
                    Some(a) => Ok(WaxAttr(a)).into_parse_result(),
                    None => {
                        input_err!(ss.loc(), "not a wax attribute: {}", raw).into_parse_result()
                    }
                }
            })
        })
        .parse_stream(state_stream)
        .into_result()
    }
}
