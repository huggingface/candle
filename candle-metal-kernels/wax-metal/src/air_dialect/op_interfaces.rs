//! Op interfaces of the `air` dialect.

use pliron::builtin::attributes::{IntegerAttr, TypeAttr};
use pliron::context::Context;
use pliron::derive::op_interface;
use pliron::op::Op;
use pliron::result::Result;
use pliron::r#type::{TypeHandle, Typed};
use pliron::{dict_key, verify_err};
use thiserror::Error;

dict_key!(
    /// Attribute key for the element type of a GEP or alloca.
    ATTR_KEY_AIR_ELEM,
    "air_elem"
);

#[derive(Error, Debug)]
#[error("Element type missing on Op")]
pub struct AirElemTypeErr;

/// Get and set `air_elem` element type attribute on an op.
#[op_interface]
pub trait AirElemType {
    fn elem_type(&self, ctx: &Context) -> TypeHandle
    where
        Self: Sized,
    {
        self.get_operation()
            .deref(ctx)
            .attributes
            .get::<TypeAttr>(&ATTR_KEY_AIR_ELEM)
            .expect("element type missing or is of incorrect type")
            .get_type(ctx)
    }

    fn set_elem_type(&self, ctx: &Context, ty: TypeHandle)
    where
        Self: Sized,
    {
        self.get_operation()
            .deref_mut(ctx)
            .attributes
            .set(ATTR_KEY_AIR_ELEM.clone(), TypeAttr::new(ty));
    }

    fn verify(op: &dyn Op, ctx: &Context) -> Result<()>
    where
        Self: Sized,
    {
        let opr = op.get_operation().deref(ctx);
        if opr.attributes.get::<TypeAttr>(&ATTR_KEY_AIR_ELEM).is_none() {
            return verify_err!(opr.loc(), AirElemTypeErr);
        }
        Ok(())
    }
}

dict_key!(
    /// Attribute key for array length.
    ATTR_KEY_AIR_COUNT,
    "air_count"
);

/// Get and set array length attribute on an op. `[n x elem]`.
#[op_interface]
pub trait AirArrayLen {
    fn array_len(&self, ctx: &Context) -> Option<u64>
    where
        Self: Sized,
    {
        self.get_operation()
            .deref(ctx)
            .attributes
            .get::<IntegerAttr>(&ATTR_KEY_AIR_COUNT)
            .map(|a| a.value().to_u64())
    }

    fn set_array_len(&self, ctx: &Context, count: IntegerAttr)
    where
        Self: Sized,
    {
        self.get_operation()
            .deref_mut(ctx)
            .attributes
            .set(ATTR_KEY_AIR_COUNT.clone(), count);
    }

    fn verify(_op: &dyn Op, _ctx: &Context) -> Result<()>
    where
        Self: Sized,
    {
        Ok(())
    }
}
