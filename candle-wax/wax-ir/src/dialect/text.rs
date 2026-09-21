//! `wax` as text. One printer, one parser, roundtrip correct.
//!
// TODO: Using native pliron types instead of enums would let us use `#[pliron::derive::format]`
// for most of this.

use crate::attr::{
    Attribute, Bounded, DenseElements, DivBy, FloatBits, OptimizationHints, SameElements,
};
use crate::types::{
    DYNAMIC, FuncType, GatherScatterViewType, PaddingValue, PartitionViewType, PointerType,
    ScalarType, StridedViewType, TensorViewType, TileElementType, TileType, Type,
};

/// A dimension or stride. Integer if known, `?` if not known ([`DYNAMIC`]).
// TODO: Evaluate if simpler to keep this inside i64 `list`/`parse_list`/`split_dims`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Dim(pub i64);

impl core::fmt::Display for Dim {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        if self.0 == DYNAMIC {
            f.write_str("?")
        } else {
            write!(f, "{}", self.0)
        }
    }
}

impl Dim {
    fn parse(s: &str) -> Option<Dim> {
        if s == "?" {
            return Some(Dim(DYNAMIC));
        }
        s.parse::<i64>().ok().map(Dim)
    }
}

/// Every scalar's IR spelling, Ordered by length.
///
/// Order used by parser to ensure prefix of one type is not interpreted as another (i1 vs i16).
const SCALARS: &[(&str, ScalarType)] = &[
    ("f8e8m0fnu", ScalarType::F8E8M0FNU),
    ("f8e4m3fn", ScalarType::F8E4M3FN),
    ("f4e2m1fn", ScalarType::F4E2M1FN),
    ("f8e5m2", ScalarType::F8E5M2),
    ("bf16", ScalarType::BF16),
    ("tf32", ScalarType::TF32),
    ("f16", ScalarType::F16),
    ("f32", ScalarType::F32),
    ("f64", ScalarType::F64),
    ("i16", ScalarType::I16),
    ("i32", ScalarType::I32),
    ("i64", ScalarType::I64),
    ("i1", ScalarType::I1),
    ("i4", ScalarType::I4),
    ("i8", ScalarType::I8),
];

pub fn scalar_name(s: &ScalarType) -> &'static str {
    SCALARS
        .iter()
        .find(|(_, v)| v == s)
        .map(|(n, _)| *n)
        .expect("every ScalarType has a spelling")
}

pub(crate) fn parse_scalar(s: &str) -> Option<ScalarType> {
    SCALARS
        .iter()
        .find(|(n, _)| *n == s)
        .map(|(_, v)| v.clone())
}

/// Dimensions as a prefix. `32x16x`. Empty at rank 0.
fn dims(v: &[i64]) -> String {
    v.iter().map(|&d| format!("{}x", Dim(d))).collect()
}

fn list(v: &[i64]) -> String {
    v.iter()
        .map(|&d| Dim(d).to_string())
        .collect::<Vec<_>>()
        .join(", ")
}

fn list32(v: &[i32]) -> String {
    v.iter()
        .map(|d| d.to_string())
        .collect::<Vec<_>>()
        .join(", ")
}

fn pad_name(p: &PaddingValue) -> &'static str {
    match p {
        PaddingValue::Zero => "zero",
        PaddingValue::NegZero => "neg_zero",
        PaddingValue::Nan => "nan",
        PaddingValue::PosInf => "pos_inf",
        PaddingValue::NegInf => "neg_inf",
    }
}

fn parse_pad(s: &str) -> Option<PaddingValue> {
    Some(match s {
        "zero" => PaddingValue::Zero,
        "neg_zero" => PaddingValue::NegZero,
        "nan" => PaddingValue::Nan,
        "pos_inf" => PaddingValue::PosInf,
        "neg_inf" => PaddingValue::NegInf,
        _ => return None,
    })
}

/// Type as text
pub fn print(t: &Type) -> String {
    match t {
        Type::Scalar(s) => scalar_name(s).to_string(),
        Type::Token => "token".to_string(),
        Type::Pointer(p) => format!("ptr<{}>", scalar_name(&p.pointee)),
        Type::Tile(tt) => format!("tile<{}{}>", dims(&tt.shape), elem(&tt.element_type)),
        Type::TensorView(tv) => format!("tensor_view<{}>", tensor_view(tv)),
        Type::PartitionView(pv) => format!(
            "partition_view<[{}] of {}{}{}>",
            list32(&pv.tile_shape),
            tensor_view(&pv.tensor_view),
            dim_map(&pv.dim_map),
            pad(&pv.padding_value),
        ),
        Type::StridedView(sv) => format!(
            "strided_view<[{}] stride [{}] of {}{}{}>",
            list32(&sv.tile_shape),
            list32(&sv.traversal_strides),
            tensor_view(&sv.tensor_view),
            dim_map(&sv.dim_map),
            pad(&sv.padding_value),
        ),
        Type::GatherScatterView(gv) => format!(
            "gather_scatter_view<[{}] of {}, sparse {}{}>",
            list32(&gv.tile_shape),
            tensor_view(&gv.tensor_view),
            gv.sparse_dim,
            pad(&gv.padding_value),
        ),
        Type::Func(ft) => format!(
            "func<({}) -> ({})>",
            ft.inputs.iter().map(print).collect::<Vec<_>>().join(", "),
            ft.results.iter().map(print).collect::<Vec<_>>().join(", "),
        ),
    }
}

fn elem(e: &TileElementType) -> String {
    match e {
        TileElementType::Scalar(s) => scalar_name(s).to_string(),
        TileElementType::Pointer(p) => format!("ptr<{}>", scalar_name(&p.pointee)),
    }
}

/// A tensor view WITHOUT its `tensor_view<..>` wrapper, so the view types can embed it.
fn tensor_view(tv: &TensorViewType) -> String {
    format!(
        "{}{}, strides [{}]",
        dims(&tv.shape),
        scalar_name(&tv.element_type),
        list(&tv.strides)
    )
}

fn dim_map(v: &[i32]) -> String {
    if v.is_empty() {
        String::new()
    } else {
        format!(", dim_map [{}]", list32(v))
    }
}

fn pad(p: &Option<PaddingValue>) -> String {
    match p {
        Some(p) => format!(", pad {}", pad_name(p)),
        None => String::new(),
    }
}

/// Parse a type from text.
pub fn parse(s: &str) -> Option<Type> {
    let s = s.trim();
    if s == "token" {
        return Some(Type::Token);
    }
    if let Some(i) = tag(s, "ptr<") {
        return Some(Type::Pointer(PointerType {
            pointee: parse_scalar(i)?,
        }));
    }
    if let Some(i) = tag(s, "tile<") {
        return parse_tile(i);
    }
    if let Some(i) = tag(s, "tensor_view<") {
        return parse_tensor_view(i).map(Type::TensorView);
    }
    if let Some(i) = tag(s, "partition_view<") {
        let (tile_shape, rest) = split_tile_shape(i)?;
        let (tv, dim_map, padding_value) = split_view_tail(rest)?;
        return Some(Type::PartitionView(PartitionViewType {
            tile_shape,
            tensor_view: tv,
            dim_map,
            padding_value,
        }));
    }
    if let Some(i) = tag(s, "strided_view<") {
        let (tile_shape, rest) = split_tile_shape(i)?;
        let rest = rest.strip_prefix("stride [")?;
        let (strides, rest) = rest.split_once("] of ")?;
        let (tv, dim_map, padding_value) = split_view_tail(rest)?;
        return Some(Type::StridedView(StridedViewType {
            tile_shape,
            traversal_strides: parse_list32(strides)?,
            tensor_view: tv,
            dim_map,
            padding_value,
        }));
    }
    if let Some(i) = tag(s, "gather_scatter_view<") {
        let (tile_shape, rest) = split_tile_shape(i)?;
        let rest = rest.strip_prefix("of ")?;
        let (tv_str, rest) = rest.split_once(", sparse ")?;
        let (sparse, padding_value) = match rest.split_once(", pad ") {
            Some((n, p)) => (n, Some(parse_pad(p)?)),
            None => (rest, None),
        };
        return Some(Type::GatherScatterView(GatherScatterViewType {
            tile_shape,
            tensor_view: parse_tensor_view(tv_str)?,
            sparse_dim: sparse.trim().parse().ok()?,
            padding_value,
        }));
    }
    if let Some(i) = tag(s, "func<") {
        let i = i.strip_prefix('(')?;
        let (inputs, rest) = i.split_once(") -> (")?;
        let results = rest.strip_suffix(')')?;
        return Some(Type::Func(FuncType {
            inputs: parse_type_list(inputs)?,
            results: parse_type_list(results)?,
        }));
    }
    parse_scalar(s).map(Type::Scalar)
}

/// Strip `prefix` and the matching `>`, tracking nesting so an inner `>` does not end it.
fn tag<'a>(s: &'a str, prefix: &str) -> Option<&'a str> {
    let rest = s.strip_prefix(prefix)?;
    let mut depth = 1usize;
    let mut prev = ' ';
    let mut in_str = false;
    for (i, c) in rest.char_indices() {
        if c == '"' && prev != '\\' {
            in_str = !in_str;
        } else if !in_str {
            match c {
                '<' => depth += 1,
                // guard against `->`
                '>' if prev != '-' => {
                    depth -= 1;
                    if depth == 0 {
                        // Close at end. `tile<i32>x` is not a type.
                        return (i == rest.len() - 1).then_some(&rest[..i]);
                    }
                }
                _ => {}
            }
        }
        prev = c;
    }
    None
}

fn parse_tile(i: &str) -> Option<Type> {
    let (shape, elem_str) = split_dims(i)?;
    let element_type = match tag(elem_str, "ptr<") {
        Some(p) => TileElementType::Pointer(Box::new(PointerType {
            pointee: parse_scalar(p)?,
        })),
        None => TileElementType::Scalar(parse_scalar(elem_str)?),
    };
    Some(Type::Tile(TileType {
        shape,
        element_type,
    }))
}

fn parse_tensor_view(i: &str) -> Option<TensorViewType> {
    let (head, strides) = i.split_once(", strides [")?;
    let strides = strides.strip_suffix(']')?;
    let (shape, elem_str) = split_dims(head)?;
    Some(TensorViewType {
        element_type: parse_scalar(elem_str)?,
        shape,
        strides: parse_list(strides)?,
    })
}

/// `32x8xf32` -> (`[32, 8]`, `f32`).
fn split_dims(s: &str) -> Option<(Vec<i64>, &str)> {
    match s.rfind('x') {
        None => Some((Vec::new(), s)),
        Some(i) => {
            let dims = s[..i]
                .split('x')
                .map(|d| Dim::parse(d).map(|d| d.0))
                .collect::<Option<Vec<_>>>()?;
            Some((dims, &s[i + 1..]))
        }
    }
}

fn split_tile_shape(s: &str) -> Option<(Vec<i32>, &str)> {
    let s = s.strip_prefix('[')?;
    let (shape, rest) = s.split_once("] ")?;
    Some((parse_list32(shape)?, rest))
}

fn split_view_tail(s: &str) -> Option<(TensorViewType, Vec<i32>, Option<PaddingValue>)> {
    let s = s.strip_prefix("of ").unwrap_or(s);
    let (s, padding_value) = match s.rsplit_once(", pad ") {
        Some((head, p)) => (head, Some(parse_pad(p)?)),
        None => (s, None),
    };
    let (s, dim_map) = match s.rsplit_once(", dim_map [") {
        Some((head, d)) => (head, parse_list32(d.strip_suffix(']')?)?),
        None => (s, Vec::new()),
    };
    Some((parse_tensor_view(s)?, dim_map, padding_value))
}

fn parse_list(s: &str) -> Option<Vec<i64>> {
    if s.trim().is_empty() {
        return Some(Vec::new());
    }
    s.split(',')
        .map(|d| Dim::parse(d.trim()).map(|d| d.0))
        .collect()
}

fn parse_list32(s: &str) -> Option<Vec<i32>> {
    if s.trim().is_empty() {
        return Some(Vec::new());
    }
    s.split(',').map(|d| d.trim().parse().ok()).collect()
}

fn parse_type_list(s: &str) -> Option<Vec<Type>> {
    if s.trim().is_empty() {
        return Some(Vec::new());
    }
    let mut out = Vec::new();
    let mut depth = 0usize;
    let mut start = 0usize;
    let mut prev = ' ';
    for (i, c) in s.char_indices() {
        match c {
            '<' | '[' | '(' => depth += 1,
            // As in `tag`: the `>` of an arrow closes nothing.
            '>' if prev == '-' => {}
            '>' | ']' | ')' => depth = depth.checked_sub(1)?,
            ',' if depth == 0 => {
                out.push(parse(&s[start..i])?);
                start = i + 1;
            }
            _ => {}
        }
        prev = c;
    }
    out.push(parse(&s[start..])?);
    Some(out)
}

fn opt_i64(v: &Option<i64>) -> String {
    match v {
        Some(i) => i.to_string(),
        None => "?".to_string(),
    }
}

fn parse_opt_i64(s: &str) -> Option<Option<i64>> {
    let s = s.trim();
    if s == "?" {
        return Some(None);
    }
    s.parse::<i64>().ok().map(Some)
}

/// Attribute as text
pub fn print_attr(a: &Attribute) -> String {
    match a {
        Attribute::Integer(v, t) => format!("int<{v} : {}>", print(t)),
        // Raw bits, not a decimal. Decimal round-trip can lose precision/NaN.
        Attribute::Float(v, t) => format!("float<{:#018x} : {}>", v.get().to_bits(), print(t)),
        Attribute::Bool(b) => format!("bool<{b}>"),
        Attribute::Type(t) => format!("type<{}>", print(t)),
        Attribute::String(s) => format!("str<{s:?}>"),
        Attribute::Array(v) => format!(
            "list<{}>",
            v.iter().map(print_attr).collect::<Vec<_>>().join(", ")
        ),
        Attribute::Dictionary(kv) => format!("dict<{}>", print_entries(kv)),
        // Upstream spelling: `array<i32: 1, 0, 0>`.
        Attribute::DenseI32Array(v) => format!("array<i32: {}>", list32(v)),
        // Upstream carries the shape as nested lists (`[[1,2],[3,4]]`).
        // Went for a simpler approach.
        Attribute::DenseElements(d) => format!(
            "dense<shape [{}], {}, data [{}]>",
            list(&d.shape),
            print(&d.element_type),
            d.data
                .iter()
                .map(|b| b.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        ),
        Attribute::DivBy(d) => format!(
            "div_by<{}, every {}, along {}>",
            d.divisor,
            opt_i64(&d.every),
            opt_i64(&d.along)
        ),
        Attribute::SameElements(s) => format!("same_elements<{}>", list(&s.values)),
        Attribute::Bounded(b) => format!("bounded<{}, {}>", opt_i64(&b.lb), opt_i64(&b.ub)),
        Attribute::OptimizationHints(h) => format!(
            "hints<{}>",
            h.entries
                .iter()
                .map(|(name, kv)| format!("group<{name:?}, {}>", print_entries(kv)))
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

fn print_entries(kv: &[(String, Attribute)]) -> String {
    format!(
        "{{{}}}",
        kv.iter()
            .map(|(k, v)| format!("{k:?} = {}", print_attr(v)))
            .collect::<Vec<_>>()
            .join(", ")
    )
}

/// Parse an attribute from text.
pub fn parse_attr(s: &str) -> Option<Attribute> {
    let s = s.trim();
    if let Some(i) = tag(s, "int<") {
        let (v, t) = i.rsplit_once(" : ")?;
        return Some(Attribute::Integer(v.trim().parse().ok()?, parse(t)?));
    }
    if let Some(i) = tag(s, "float<") {
        let (v, t) = i.rsplit_once(" : ")?;
        let bits = u64::from_str_radix(v.trim().strip_prefix("0x")?, 16).ok()?;
        return Some(Attribute::Float(
            FloatBits::new(f64::from_bits(bits)),
            parse(t)?,
        ));
    }
    if let Some(i) = tag(s, "bool<") {
        return Some(Attribute::Bool(i.trim().parse().ok()?));
    }
    if let Some(i) = tag(s, "type<") {
        return Some(Attribute::Type(parse(i)?));
    }
    if let Some(i) = tag(s, "str<") {
        return Some(Attribute::String(unquote(i)?));
    }
    if let Some(i) = tag(s, "list<") {
        if i.trim().is_empty() {
            return Some(Attribute::Array(Vec::new()));
        }
        return split_top(i)
            .into_iter()
            .map(parse_attr)
            .collect::<Option<_>>()
            .map(Attribute::Array);
    }
    if let Some(i) = tag(s, "dict<") {
        return parse_entries(i).map(Attribute::Dictionary);
    }
    if let Some(i) = tag(s, "array<") {
        return Some(Attribute::DenseI32Array(parse_list32(
            i.strip_prefix("i32:")?,
        )?));
    }
    if let Some(i) = tag(s, "dense<") {
        let i = i.strip_prefix("shape [")?;
        let (shape, rest) = i.split_once("], ")?;
        let (ty, data) = rest.split_once(", data [")?;
        return Some(Attribute::DenseElements(DenseElements {
            element_type: parse(ty)?,
            shape: parse_list(shape)?,
            data: parse_bytes(data.strip_suffix(']')?)?,
        }));
    }
    if let Some(i) = tag(s, "div_by<") {
        let parts = split_top(i);
        let [d, every, along] = parts.as_slice() else {
            return None;
        };
        return Some(Attribute::DivBy(DivBy {
            divisor: d.trim().parse().ok()?,
            every: parse_opt_i64(every.trim().strip_prefix("every ")?)?,
            along: parse_opt_i64(along.trim().strip_prefix("along ")?)?,
        }));
    }
    if let Some(i) = tag(s, "same_elements<") {
        return Some(Attribute::SameElements(SameElements {
            values: parse_list(i)?,
        }));
    }
    if let Some(i) = tag(s, "bounded<") {
        let (lb, ub) = i.split_once(',')?;
        return Some(Attribute::Bounded(Bounded {
            lb: parse_opt_i64(lb)?,
            ub: parse_opt_i64(ub)?,
        }));
    }
    if let Some(i) = tag(s, "hints<") {
        if i.trim().is_empty() {
            return Some(Attribute::OptimizationHints(OptimizationHints {
                entries: Vec::new(),
            }));
        }
        let mut entries = Vec::new();
        for g in split_top(i) {
            let g = tag(g.trim(), "group<")?;
            let (name, kv) = g.split_once(", ")?;
            entries.push((unquote(name)?, parse_entries(kv)?));
        }
        return Some(Attribute::OptimizationHints(OptimizationHints { entries }));
    }
    None
}

fn parse_entries(s: &str) -> Option<Vec<(String, Attribute)>> {
    let inner = s.trim().strip_prefix('{')?.strip_suffix('}')?;
    if inner.trim().is_empty() {
        return Some(Vec::new());
    }
    split_top(inner)
        .into_iter()
        .map(|e| {
            let (k, v) = e.split_once(" = ")?;
            Some((unquote(k.trim())?, parse_attr(v)?))
        })
        .collect()
}

fn parse_bytes(s: &str) -> Option<Vec<u8>> {
    if s.trim().is_empty() {
        return Some(Vec::new());
    }
    s.split(',').map(|b| b.trim().parse().ok()).collect()
}

/// A quoted string as `{:?}` writes it.
fn unquote(s: &str) -> Option<String> {
    let inner = s.trim().strip_prefix('"')?.strip_suffix('"')?;
    let mut out = String::new();
    let mut chars = inner.chars();
    while let Some(c) = chars.next() {
        if c != '\\' {
            out.push(c);
            continue;
        }
        match chars.next()? {
            'n' => out.push('\n'),
            't' => out.push('\t'),
            'r' => out.push('\r'),
            '0' => out.push('\0'),
            c @ ('"' | '\\' | '\'') => out.push(c),
            _ => return None,
        }
    }
    Some(out)
}

/// Split on top-level commas, so a nested list or dictionary stays in one piece.
fn split_top(s: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut depth = 0i32;
    let mut start = 0usize;
    let mut prev = ' ';
    let mut in_str = false;

    for (i, c) in s.char_indices() {
        match c {
            '"' if prev != '\\' => in_str = !in_str,
            _ if in_str => {}
            '<' | '[' | '(' | '{' => depth += 1,
            '>' if prev == '-' => {}
            '>' | ']' | ')' | '}' => depth -= 1,
            ',' if depth == 0 => {
                out.push(&s[start..i]);
                start = i + 1;
            }
            _ => {}
        }
        prev = c;
    }
    out.push(&s[start..]);
    out
}
