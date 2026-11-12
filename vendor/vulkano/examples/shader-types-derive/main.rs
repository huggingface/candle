// This example demonstrates how to put derives onto Rust structs generated from the shader types
// through the `custom_derives` option of the `shader!` macro.

// The `shader!` macro is capable of generating Rust structs representing each type found in the
// shader source. These structs are generated in the same module where the macro was called.
//
// By default each type only has `Clone` and `Copy` derives. For ergonomic purposes you may want to
// add more derives for each type. For example built-in derive macros such as `Default` or `Debug`.
//
// The only way we can add derive macros to these generated types is if the `shader!` macro
// generates the derive attribute with the wanted derives, hence there's a macro option for it.

use ron::ser::PrettyConfig;
use serde::{Deserialize, Serialize};
use std::fmt::{Debug, Display, Error as FmtError, Formatter};
use vulkano::padded::Padded;

vulkano_shaders::shader! {
    ty: "compute",
    src: r"
        #version 450

        struct Foo {
            float x;
            vec3 z;
        };

        struct Bar {
            vec2 y;
            Foo foo;
        };

        layout(push_constant) uniform PushConstantData {
            int multiple;
        } pc;

        layout(set = 0, binding = 1) buffer Bars {
            Bar bar[];
        };

        void main() {}
    ",
    custom_derives: [Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize],
}

// In the example above the macro generates `Clone`, `Copy`, `Debug`, `Default`, `PartialEq`,
// derives for each declared type (`PushConstantData`, `Foo` and `Bar`) in the shader, and it also
// applies derives of the `Serialize` and `Deserialize` traits from serde. However, it doesn't
// apply any of these to `Bars` since that type does not a have size known at compile time.

// Some traits are not meant to be derived, such as `Display`, but we can still implement them
// manually.
impl Display for Foo {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> Result<(), FmtError> {
        Debug::fmt(self, formatter)
    }
}

fn main() {
    // Prints "Foo { x: 0.0, z: [100.0, 200.0, 300.0] }".
    println!(
        "{:?}",
        Foo {
            z: [100.0, 200.0, 300.0],
            ..Default::default()
        },
    );

    let mut bar = Bar {
        // The `Padded` wrapper here is padding the following field, `foo`.
        y: Padded([5.1, 6.2]),
        // Fills all fields with zeroes.
        ..Default::default()
    };

    assert_eq!(
        Bar {
            // `Padded<T, N>` implements `From<T>`, so you can construct it this way as well.
            y: [5.1, 6.2].into(),
            ..Default::default()
        },
        bar,
    );

    assert_ne!(Bar::default(), bar);

    // `Padded` dereferences into the wrapped type, so we can easily access the underlying data.
    *bar.foo.x = 125.0;

    // Since we put `Serialize` and `Deserialize` traits to the derives list we can serialize and
    // deserialize shader data.

    let serialized = ron::ser::to_string_pretty(&bar, PrettyConfig::default()).unwrap();

    println!("Serialized Bar: {serialized}");

    let deserialized = ron::from_str::<Bar>(&serialized).unwrap();

    assert_eq!(*deserialized.foo.x, 125.0);
}
