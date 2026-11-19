use crate::descriptor_set::DescriptorSetWithOffsets;

/// A collection of descriptor set objects.
pub unsafe trait DescriptorSetsCollection {
    fn into_vec(self) -> Vec<DescriptorSetWithOffsets>;
}

unsafe impl DescriptorSetsCollection for () {
    #[inline]
    fn into_vec(self) -> Vec<DescriptorSetWithOffsets> {
        vec![]
    }
}

unsafe impl<T> DescriptorSetsCollection for T
where
    T: Into<DescriptorSetWithOffsets>,
{
    fn into_vec(self) -> Vec<DescriptorSetWithOffsets> {
        vec![self.into()]
    }
}

unsafe impl<T> DescriptorSetsCollection for Vec<T>
where
    T: Into<DescriptorSetWithOffsets>,
{
    fn into_vec(self) -> Vec<DescriptorSetWithOffsets> {
        self.into_iter().map(|x| x.into()).collect()
    }
}

macro_rules! impl_collection {
    ($first:ident $(, $others:ident)+) => (
        unsafe impl<$first$(, $others)+> DescriptorSetsCollection for ($first, $($others),+)
            where $first: Into<DescriptorSetWithOffsets>
                  $(, $others: Into<DescriptorSetWithOffsets>)*
        {
            #[inline]
            #[allow(non_snake_case)]
            fn into_vec(self) -> Vec<DescriptorSetWithOffsets> {
                let ($first, $($others,)*) = self;
                vec![$first.into() $(, $others.into())+]
            }
        }

        impl_collection!($($others),+);
    );

    ($i:ident) => ();
}

impl_collection!(Z, Y, X, W, V, U, T, S, R, Q, P, O, N, M, L, K, J, I, H, G, F, E, D, C, B, A);
