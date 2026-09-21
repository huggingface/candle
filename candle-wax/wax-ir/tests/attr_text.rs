use wax_ir::attr::{
    Attribute, Bounded, DenseElements, DivBy, FloatBits, OptimizationHints, SameElements,
};
use wax_ir::dialect::text::{parse_attr, print_attr};
use wax_ir::types::{ScalarType, Type};

fn i32_ty() -> Type {
    Type::Scalar(ScalarType::I32)
}

fn attributes() -> Vec<Attribute> {
    let leaf = vec![
        Attribute::Integer(-7, i32_ty()),
        Attribute::Integer(i64::MIN, i32_ty()),
        Attribute::Float(FloatBits::new(1.5), Type::Scalar(ScalarType::F64)),
        Attribute::Float(FloatBits::new(f64::NAN), Type::Scalar(ScalarType::F64)),
        Attribute::Float(
            FloatBits::new(f64::NEG_INFINITY),
            Type::Scalar(ScalarType::F64),
        ),
        Attribute::Bool(true),
        Attribute::Bool(false),
        Attribute::Type(i32_ty()),
        Attribute::String("hello".to_string()),
        Attribute::String("\"hello\", world\\! >>><>".to_string()),
        Attribute::String(String::new()),
        Attribute::Array(vec![]),
        Attribute::DenseElements(DenseElements {
            element_type: i32_ty(),
            shape: vec![2, 2],
            data: vec![1, 0, 0, 0, 2, 0, 0, 0],
        }),
        Attribute::DenseElements(DenseElements {
            element_type: i32_ty(),
            shape: vec![],
            data: vec![],
        }),
        Attribute::DenseI32Array(vec![]),
        Attribute::DenseI32Array(vec![1, 0, 0]),
        Attribute::DivBy(DivBy {
            divisor: 16,
            every: None,
            along: None,
        }),
        Attribute::DivBy(DivBy {
            divisor: 16,
            every: Some(4),
            along: Some(0),
        }),
        Attribute::SameElements(SameElements { values: vec![] }),
        Attribute::SameElements(SameElements {
            values: vec![1, 2, 3],
        }),
        Attribute::Dictionary(vec![]),
        Attribute::OptimizationHints(OptimizationHints { entries: vec![] }),
        Attribute::OptimizationHints(OptimizationHints {
            entries: vec![
                (
                    "unroll".into(),
                    vec![("count".into(), Attribute::Integer(4, i32_ty()))],
                ),
                ("empty".into(), vec![]),
            ],
        }),
        Attribute::Bounded(Bounded { lb: None, ub: None }),
        Attribute::Bounded(Bounded {
            lb: Some(0),
            ub: None,
        }),
        Attribute::Bounded(Bounded {
            lb: Some(-4),
            ub: Some(42),
        }),
    ];
    // Nested attributes
    let mut all = leaf.clone();
    all.push(Attribute::Array(leaf.clone()));
    all.push(Attribute::Array(vec![Attribute::Array(leaf.clone())]));
    all.push(Attribute::Dictionary(
        leaf.iter()
            .enumerate()
            .map(|(i, a)| (format!("k{i}"), a.clone()))
            .collect(),
    ));
    all
}

#[test]
fn attributes_round_trip() {
    for a in attributes() {
        let text = print_attr(&a);
        match parse_attr(&text) {
            Some(back) => assert_eq!(back, a, "{text} parsed, but incorrectly"),
            None => panic!("{text} printed, but not parsed"),
        }
        assert_eq!(
            print_attr(&parse_attr(&text).unwrap()),
            text,
            "round trip failed"
        );
    }
}

#[test]
fn none_prints_question_mark() {
    assert_eq!(
        print_attr(&Attribute::DivBy(DivBy {
            divisor: 16,
            every: None,
            along: None
        })),
        "div_by<16, every ?, along ?>"
    );
    assert_eq!(
        print_attr(&Attribute::Bounded(Bounded {
            lb: None,
            ub: Some(42)
        })),
        "bounded<?, 42>"
    );
}

#[test]
fn malformed_attr_is_not_parsed() {
    for malformed in [
        "",
        "nope<?>",
        "bounded<>",
        "dense<shape [2]>",
        "int<5>",
        "list<",
    ] {
        assert_eq!(
            parse_attr(malformed),
            None,
            "{malformed} is malformed and should not parse."
        );
    }
}
