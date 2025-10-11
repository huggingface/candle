macro_rules! ops{
    ($($name:ident),+) => {

        pub mod contiguous {
        pub struct Kernel(pub &'static str);
        $(
        pub mod $name {
            use super::Kernel;
            pub const FLOAT: Kernel = Kernel(concat!(stringify!($name), "_f32"));
            pub const HALF: Kernel = Kernel(concat!(stringify!($name), "_f16"));
            pub const BFLOAT: Kernel = Kernel(concat!(stringify!($name), "_bf16"));
            pub const I64: Kernel = Kernel(concat!(stringify!($name), "_i64"));
            pub const U32: Kernel = Kernel(concat!(stringify!($name), "_u32"));
            pub const U8: Kernel = Kernel(concat!(stringify!($name), "_u8"));
        }
        )+
            pub mod copy {
                use super::Kernel;
                pub const FLOAT: Kernel = Kernel("copy_f32");
                pub const HALF: Kernel = Kernel("copy_f16");
                pub const BFLOAT: Kernel = Kernel("copy_bf16");
                pub const I64: Kernel = Kernel("copy_i64");
                pub const U32: Kernel = Kernel("copy_u32");
                pub const U8: Kernel = Kernel("copy_u8");
            }
        }

        pub mod contiguous_tiled {
        pub struct Kernel(pub &'static str);
        $(
        pub mod $name {
            use super::Kernel;
            pub const FLOAT: Kernel = Kernel(concat!(stringify!($name), "_f32_tiled"));
            pub const HALF: Kernel = Kernel(concat!(stringify!($name), "_f16_tiled"));
            pub const BFLOAT: Kernel = Kernel(concat!(stringify!($name), "_bf16_tiled"));
            pub const I64: Kernel = Kernel(concat!(stringify!($name), "_i64_tiled"));
            pub const U32: Kernel = Kernel(concat!(stringify!($name), "_u32_tiled"));
            pub const U8: Kernel = Kernel(concat!(stringify!($name), "_u8_tiled"));
        }
        )+
            pub mod copy {
                use super::Kernel;
                pub const FLOAT: Kernel = Kernel("copy_f32_tiled");
                pub const HALF: Kernel = Kernel("copy_f16_tiled");
                pub const BFLOAT: Kernel = Kernel("copy_bf16_tiled");
                pub const I64: Kernel = Kernel("copy_i64_tiled");
                pub const U32: Kernel = Kernel("copy_u32_tiled");
                pub const U8: Kernel = Kernel("copy_u8_tiled");
            }
        }

        pub mod strided {
        pub struct Kernel(pub &'static str);
        $(
        pub mod $name {
            use super::Kernel;
            pub const FLOAT: Kernel = Kernel(concat!(stringify!($name), "_f32_strided"));
            pub const HALF: Kernel = Kernel(concat!(stringify!($name), "_f16_strided"));
            pub const BFLOAT: Kernel = Kernel(concat!(stringify!($name), "_bf16_strided"));
            pub const I64: Kernel = Kernel(concat!(stringify!($name), "_i64_strided"));
            pub const U32: Kernel = Kernel(concat!(stringify!($name), "_u32_strided"));
            pub const U8: Kernel = Kernel(concat!(stringify!($name), "_u8_strided"));
        }
        )+
            pub mod copy {
                use super::Kernel;
                pub const FLOAT: Kernel = Kernel("copy_f32_strided");
                pub const HALF: Kernel = Kernel("copy_f16_strided");
                pub const BFLOAT: Kernel = Kernel("copy_bf16_strided");
                pub const I64: Kernel = Kernel("copy_i64_strided");
                pub const U32: Kernel = Kernel("copy_u32_strided");
                pub const U8: Kernel = Kernel("copy_u8_strided");
            }
        }
    };
}
pub(crate) use ops;
