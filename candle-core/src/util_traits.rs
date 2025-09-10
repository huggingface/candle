//! Utility traits that can be used to apply different behaviors based on a condition.
mod sealed {
    pub trait Sealed {}
    impl Sealed for super::True {}
    impl Sealed for super::False {}
}

pub trait CondValue: sealed::Sealed {}
pub struct True;
pub struct False;
impl CondValue for True {}
impl CondValue for False {}

pub trait Condition {
    type Value: CondValue;
}

pub struct IsSame<T, U> {
    marker: std::marker::PhantomData<(T, U)>,
}

impl<T> Condition for IsSame<T, T> {
    type Value = True;
}
