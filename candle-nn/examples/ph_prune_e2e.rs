//! End-to-end Candle benchmark for PH-guided KV pruning and activation routing.
//!
//! This intentionally uses real `candle::Tensor` operations for the measured
//! compute paths. The persistence code is local to the example so the benchmark
//! can be run before any public API is proposed.

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use candle::{DType, Device, Result, Tensor};
use candle_nn::ops::softmax_last_dim;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::hint::black_box;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

#[derive(Debug, Clone)]
struct Args {
    tokens: usize,
    retain: usize,
    dim: usize,
    queries: usize,
    iters: usize,
    radius: f32,
    max_simplices: usize,
    kv_cache: Option<PathBuf>,
    write_kv_cache: Option<PathBuf>,
}

impl Args {
    fn parse() -> Self {
        let mut args = Self {
            tokens: 384,
            retain: 96,
            dim: 64,
            queries: 8,
            iters: 50,
            radius: 0.72,
            max_simplices: 80_000,
            kv_cache: None,
            write_kv_cache: None,
        };
        let mut raw = std::env::args().skip(1);
        while let Some(flag) = raw.next() {
            let value = raw
                .next()
                .unwrap_or_else(|| panic!("missing value for {flag}"));
            match flag.as_str() {
                "--tokens" => args.tokens = value.parse().expect("invalid --tokens"),
                "--retain" => args.retain = value.parse().expect("invalid --retain"),
                "--dim" => args.dim = value.parse().expect("invalid --dim"),
                "--queries" => args.queries = value.parse().expect("invalid --queries"),
                "--iters" => args.iters = value.parse().expect("invalid --iters"),
                "--radius" => args.radius = value.parse().expect("invalid --radius"),
                "--max-simplices" => {
                    args.max_simplices = value.parse().expect("invalid --max-simplices")
                }
                "--kv-cache" => args.kv_cache = Some(PathBuf::from(value)),
                "--write-kv-cache" => args.write_kv_cache = Some(PathBuf::from(value)),
                _ => panic!("unknown flag {flag}"),
            }
        }
        args
    }
}

#[derive(Clone, Debug)]
struct Point {
    coords: Vec<f32>,
}

impl Point {
    fn distance(&self, other: &Self) -> f32 {
        self.coords
            .iter()
            .zip(&other.coords)
            .map(|(a, b)| {
                let d = a - b;
                d * d
            })
            .sum::<f32>()
            .sqrt()
    }
}

#[derive(Clone, Debug)]
struct PersistencePair {
    dimension: usize,
    birth: f32,
    death: f32,
}

impl PersistencePair {
    fn persistence(&self) -> f32 {
        (self.death - self.birth).max(0.0)
    }
}

#[derive(Default)]
struct PersistenceDiagram {
    pairs: Vec<PersistencePair>,
    simplex_count: usize,
    truncated: bool,
}

impl PersistenceDiagram {
    fn total(&self, dimension: usize) -> f32 {
        self.pairs
            .iter()
            .filter(|pair| pair.dimension == dimension)
            .map(PersistencePair::persistence)
            .sum()
    }
}

#[derive(Clone, Debug)]
struct Simplex {
    vertices: Vec<usize>,
    filtration: f32,
    dimension: usize,
}

#[derive(Clone, Debug)]
struct KvCacheRows {
    keys: Vec<Point>,
    values: Vec<Point>,
    queries: Vec<Point>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let device = Device::Cpu;
    let (cache, source) = if let Some(path) = &args.kv_cache {
        (
            load_kv_cache(path, &device)?,
            format!("loaded:{}", path.display()),
        )
    } else {
        (
            make_transformer_like_kv_cache(args.tokens, args.dim, args.queries),
            "generated:transformer-like".to_string(),
        )
    };
    if let Some(path) = &args.write_kv_cache {
        write_kv_cache(path, &cache, &device)?;
    }

    let keys = cache.keys;
    let values = cache.values;
    let queries = cache.queries;
    let tokens = keys.len();
    let retain = args.retain.min(tokens);
    let dim = keys.first().map(|point| point.coords.len()).unwrap_or(0);
    let query_count = queries.len();
    if retain == 0 || tokens == 0 || dim == 0 || query_count == 0 {
        candle::bail!("kv cache must have non-empty k, v, and q tensors");
    }

    let ph_start = Instant::now();
    let retained = ph_retain_indices(&keys, retain);
    let retained_points: Vec<_> = retained.iter().map(|idx| keys[*idx].clone()).collect();
    let full_landmarks = farthest_point_landmarks(&keys, retain);
    let full_diagram = persistent_homology(&full_landmarks, args.radius, args.max_simplices);
    let retained_diagram = persistent_homology(&retained_points, args.radius, args.max_simplices);
    let ph_time = ph_start.elapsed();

    let q = Tensor::from_vec(flatten_points(&queries), (query_count, dim), &device)?;
    let k = Tensor::from_vec(flatten_points(&keys), (tokens, dim), &device)?;
    let v = Tensor::from_vec(flatten_points(&values), (tokens, dim), &device)?;
    let retained_ids = Tensor::from_vec(
        retained.iter().map(|idx| *idx as u32).collect::<Vec<_>>(),
        (retain,),
        &device,
    )?;
    let k_retained = k.index_select(&retained_ids, 0)?;
    let v_retained = v.index_select(&retained_ids, 0)?;

    let dense_attention_time = repeat(args.iters, || attention(&q, &k, &v))?;
    let pruned_attention_time = repeat(args.iters, || attention(&q, &k_retained, &v_retained))?;

    let activations = Tensor::from_vec(make_activations(tokens), (tokens, 1), &device)?;
    let weights = Tensor::from_vec(
        make_activation_weights(tokens, tokens),
        (tokens, tokens),
        &device,
    )?;
    let routed = route_activation_indices(&activations.to_vec2::<f32>()?, 0.35);
    let routed_ids = Tensor::from_vec(
        routed.iter().map(|idx| *idx as u32).collect::<Vec<_>>(),
        (routed.len(),),
        &device,
    )?;
    let routed_weights = weights.index_select(&routed_ids, 1)?;
    let routed_activations = activations.index_select(&routed_ids, 0)?;

    let dense_matvec_time = repeat(args.iters, || weights.matmul(&activations))?;
    let routed_matvec_time = repeat(args.iters, || routed_weights.matmul(&routed_activations))?;

    println!("ph_prune_e2e");
    println!(
        "source={} tokens={} retain={} dim={} queries={} iters={}",
        source, tokens, retain, dim, query_count, args.iters
    );
    println!(
        "ph_ms={:.3} full_simplices={} retained_simplices={} full_truncated={} retained_truncated={} h0_loss={:.4} h1_loss={:.4} h2_loss={:.4}",
        ms(ph_time),
        full_diagram.simplex_count,
        retained_diagram.simplex_count,
        full_diagram.truncated,
        retained_diagram.truncated,
        (full_diagram.total(0) - retained_diagram.total(0)).abs(),
        (full_diagram.total(1) - retained_diagram.total(1)).abs(),
        (full_diagram.total(2) - retained_diagram.total(2)).abs()
    );
    println!(
        "kv_attention dense_ms={:.3} pruned_ms={:.3} speedup={:.2}x compression={:.2}x",
        ms(dense_attention_time),
        ms(pruned_attention_time),
        dense_attention_time.as_secs_f64() / pruned_attention_time.as_secs_f64(),
        tokens as f64 / retain as f64
    );
    println!(
        "activation_routing keep={}/{} skip={:.1}% dense_ms={:.3} routed_ms={:.3} speedup={:.2}x matmul_reduction={:.2}x",
        routed.len(),
        tokens,
        100.0 * (1.0 - routed.len() as f64 / tokens as f64),
        ms(dense_matvec_time),
        ms(routed_matvec_time),
        dense_matvec_time.as_secs_f64() / routed_matvec_time.as_secs_f64(),
        tokens as f64 / routed.len() as f64
    );
    Ok(())
}

fn attention(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
    let scale = 1.0 / (q.dim(1)? as f64).sqrt();
    let scores = (q.matmul(&k.t()?)? * scale)?;
    let probs = softmax_last_dim(&scores)?;
    probs.matmul(v)
}

fn repeat<T>(iters: usize, mut f: impl FnMut() -> Result<T>) -> Result<Duration> {
    let start = Instant::now();
    for _ in 0..iters {
        black_box(f()?);
    }
    Ok(start.elapsed())
}

fn ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

fn make_transformer_like_kv_cache(tokens: usize, dim: usize, queries: usize) -> KvCacheRows {
    let topic_count = 8usize;
    let topics: Vec<_> = (0..topic_count)
        .map(|topic| {
            (0..dim)
                .map(|d| ((topic * 31 + d * 17) as f32 * 0.073).sin())
                .collect::<Vec<_>>()
        })
        .collect();
    let mut running_context = vec![0.0f32; dim];
    let mut keys = Vec::with_capacity(tokens);
    let mut values = Vec::with_capacity(tokens);

    for token in 0..tokens {
        let segment = (token * topic_count / tokens.max(1)).min(topic_count - 1);
        let position = token as f32 / tokens.max(1) as f32;
        let outlier_gain = if token < 8 || token.is_multiple_of(127) {
            1.85
        } else {
            1.0
        };
        let mut key = Vec::with_capacity(dim);
        let mut value = Vec::with_capacity(dim);
        for d in 0..dim {
            let head = d / 16;
            let rotary = ((token as f32 + 1.0) * (d as f32 + 3.0) * 0.0019).sin();
            let local = ((token as f32 * 0.037) + (d as f32 * 0.011)).cos() * 0.18;
            let topic = topics[segment][d] * (0.42 + head as f32 * 0.03);
            running_context[d] = 0.94 * running_context[d] + 0.06 * (topic + rotary);
            let attention_sink = if token < 4 {
                0.75 + d as f32 * 0.002
            } else {
                0.0
            };
            let key_value = outlier_gain * (topic + 0.55 * rotary + local + attention_sink)
                + 0.25 * running_context[d]
                + position * 0.03;
            key.push(key_value);
            value.push(
                0.62 * key_value
                    + 0.31 * running_context[d]
                    + ((token * 13 + d * 7) as f32 * 0.017).sin() * 0.09,
            );
        }
        keys.push(Point { coords: key });
        values.push(Point { coords: value });
    }

    let queries = (0..queries)
        .map(|query| {
            let anchor = tokens.saturating_sub(1 + query * tokens.max(1) / queries.max(1));
            let source = keys.get(anchor).or_else(|| keys.last());
            let coords = (0..dim)
                .map(|d| {
                    let base = source.map(|point| point.coords[d]).unwrap_or(0.0);
                    0.78 * base + ((query * 19 + d * 5) as f32 * 0.029).sin() * 0.16
                })
                .collect();
            Point { coords }
        })
        .collect();

    KvCacheRows {
        keys,
        values,
        queries,
    }
}

fn write_kv_cache(path: &Path, cache: &KvCacheRows, device: &Device) -> Result<()> {
    let dim = cache
        .keys
        .first()
        .map(|point| point.coords.len())
        .unwrap_or(0);
    let query_dim = cache
        .queries
        .first()
        .map(|point| point.coords.len())
        .unwrap_or(dim);
    let mut tensors = HashMap::new();
    tensors.insert(
        "k",
        Tensor::from_vec(flatten_points(&cache.keys), (cache.keys.len(), dim), device)?,
    );
    tensors.insert(
        "v",
        Tensor::from_vec(
            flatten_points(&cache.values),
            (cache.values.len(), dim),
            device,
        )?,
    );
    tensors.insert(
        "q",
        Tensor::from_vec(
            flatten_points(&cache.queries),
            (cache.queries.len(), query_dim),
            device,
        )?,
    );
    candle::safetensors::save(&tensors, path)
}

fn load_kv_cache(path: &Path, device: &Device) -> Result<KvCacheRows> {
    let tensors = candle::safetensors::load(path, device)?;
    let keys = load_named_points(&tensors, &["k", "key", "keys"])?;
    let values = load_named_points(&tensors, &["v", "value", "values"])?;
    let queries = load_named_points(&tensors, &["q", "query", "queries"])?;
    validate_kv_cache(&keys, &values, &queries)?;
    Ok(KvCacheRows {
        keys,
        values,
        queries,
    })
}

fn load_named_points(tensors: &HashMap<String, Tensor>, names: &[&str]) -> Result<Vec<Point>> {
    for name in names {
        if let Some(tensor) = tensors.get(*name) {
            return tensor_to_points(tensor);
        }
    }
    candle::bail!("missing tensor; tried names {:?}", names)
}

fn tensor_to_points(tensor: &Tensor) -> Result<Vec<Point>> {
    Ok(tensor
        .to_dtype(DType::F32)?
        .to_vec2::<f32>()?
        .into_iter()
        .map(|coords| Point { coords })
        .collect())
}

fn validate_kv_cache(keys: &[Point], values: &[Point], queries: &[Point]) -> Result<()> {
    if keys.is_empty() || values.is_empty() || queries.is_empty() {
        candle::bail!("kv cache tensors must be non-empty")
    }
    if keys.len() != values.len() {
        candle::bail!(
            "k/v token count mismatch: {} != {}",
            keys.len(),
            values.len()
        )
    }
    let dim = keys[0].coords.len();
    if dim == 0 {
        candle::bail!("kv cache dimension must be non-zero")
    }
    if values.iter().any(|point| point.coords.len() != dim)
        || keys.iter().any(|point| point.coords.len() != dim)
        || queries.iter().any(|point| point.coords.len() != dim)
    {
        candle::bail!("k, v, and q tensors must all be rank-2 with matching last dimension")
    }
    Ok(())
}

#[cfg(test)]
fn make_key_points(tokens: usize, dim: usize) -> Vec<Point> {
    (0..tokens)
        .map(|idx| {
            let t = idx as f32 * std::f32::consts::TAU / (tokens.saturating_sub(16).max(1)) as f32;
            let mut coords = Vec::with_capacity(dim);
            for d in 0..dim {
                let base = match d % 4 {
                    0 => t.cos(),
                    1 => t.sin(),
                    2 => (2.0 * t).cos() * 0.35,
                    _ => (3.0 * t).sin() * 0.20,
                };
                let outlier = if idx < 16 {
                    (idx as f32 + 1.0) * 0.42 + d as f32 * 0.011
                } else {
                    0.0
                };
                coords.push(base + outlier);
            }
            Point { coords }
        })
        .collect()
}

fn flatten_points(points: &[Point]) -> Vec<f32> {
    points
        .iter()
        .flat_map(|point| point.coords.iter().copied())
        .collect()
}

fn make_activations(len: usize) -> Vec<f32> {
    (0..len)
        .map(|idx| {
            if idx < len / 4 {
                (idx as f32 * 0.17).sin() * 0.9 + (idx % 11) as f32 * 0.015
            } else {
                (idx as f32 * 0.031).sin() * 0.04
            }
        })
        .collect()
}

fn make_activation_weights(rows: usize, cols: usize) -> Vec<f32> {
    (0..rows)
        .flat_map(|r| (0..cols).map(move |c| ((r * 13 + c * 7) as f32 * 0.021).sin() * 0.5))
        .collect()
}

fn route_activation_indices(activations: &[Vec<f32>], threshold: f32) -> Vec<usize> {
    let values: Vec<_> = activations.iter().map(|row| row[0]).collect();
    values
        .iter()
        .enumerate()
        .filter_map(|(idx, value)| {
            let left = idx
                .checked_sub(1)
                .and_then(|i| values.get(i))
                .copied()
                .unwrap_or(*value);
            let right = values.get(idx + 1).copied().unwrap_or(*value);
            let curvature = (left - 2.0 * value + right).abs();
            let score = value.abs() + 0.5 * curvature;
            (score >= threshold).then_some(idx)
        })
        .collect()
}

fn ph_retain_indices(points: &[Point], retain: usize) -> Vec<usize> {
    let mut scores = h0_persistence_scores(points);
    let landmarks = farthest_point_landmark_indices(points, retain.min(points.len()));
    for (rank, idx) in landmarks.iter().enumerate() {
        scores[*idx] += (retain.saturating_sub(rank) as f32) * 0.01;
    }
    let mut ranked: Vec<_> = scores.into_iter().enumerate().collect();
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    let mut retained: Vec<_> = ranked
        .into_iter()
        .take(retain)
        .map(|(idx, _)| idx)
        .collect();
    retained.sort_unstable();
    retained
}

fn h0_persistence_scores(points: &[Point]) -> Vec<f32> {
    let mut parent: Vec<_> = (0..points.len()).collect();
    let mut birth: Vec<_> = (0..points.len()).collect();
    let mut scores = vec![0.0; points.len()];
    let mut edges = Vec::new();
    for i in 0..points.len() {
        for j in i + 1..points.len() {
            edges.push((points[i].distance(&points[j]), i, j));
        }
    }
    edges.sort_by(|a, b| a.0.total_cmp(&b.0));
    for (distance, i, j) in edges {
        let ri = find(&mut parent, i);
        let rj = find(&mut parent, j);
        if ri == rj {
            continue;
        }
        let killed = birth[ri].max(birth[rj]);
        scores[killed] = distance;
        let survivor = birth[ri].min(birth[rj]);
        parent[rj] = ri;
        birth[ri] = survivor;
    }
    for idx in 0..scores.len() {
        if scores[idx] == 0.0 {
            scores[idx] = nearest_distance(points, idx);
        }
    }
    scores
}

fn persistent_homology(
    points: &[Point],
    max_radius: f32,
    max_simplices: usize,
) -> PersistenceDiagram {
    if points.is_empty() {
        return PersistenceDiagram::default();
    }
    let (simplices, truncated) = vietoris_rips_simplices(points, max_radius, max_simplices);
    let simplex_count = simplices.len();
    let simplex_index: BTreeMap<Vec<usize>, usize> = simplices
        .iter()
        .enumerate()
        .map(|(idx, simplex)| (simplex.vertices.clone(), idx))
        .collect();
    let mut reduced_by_low: HashMap<usize, Vec<usize>> = HashMap::new();
    let mut births = HashSet::new();
    let mut killed = HashSet::new();
    let mut pairs = Vec::new();

    for (col_idx, simplex) in simplices.iter().enumerate() {
        let mut column = boundary_indices(simplex, &simplex_index);
        while let Some(low) = column.last().copied() {
            if let Some(previous) = reduced_by_low.get(&low) {
                column = xor_sorted(&column, previous);
            } else {
                break;
            }
        }
        if let Some(low) = column.last().copied() {
            reduced_by_low.insert(low, column);
            killed.insert(low);
            let born = &simplices[low];
            if simplex.filtration > born.filtration {
                pairs.push(PersistencePair {
                    dimension: born.dimension,
                    birth: born.filtration,
                    death: simplex.filtration,
                });
            }
        } else {
            births.insert(col_idx);
        }
    }

    for birth_idx in births.difference(&killed) {
        let simplex = &simplices[*birth_idx];
        if max_radius > simplex.filtration {
            pairs.push(PersistencePair {
                dimension: simplex.dimension,
                birth: simplex.filtration,
                death: max_radius,
            });
        }
    }
    PersistenceDiagram {
        pairs,
        simplex_count,
        truncated,
    }
}

fn vietoris_rips_simplices(
    points: &[Point],
    max_radius: f32,
    max_simplices: usize,
) -> (Vec<Simplex>, bool) {
    let mut simplices = Vec::new();
    let mut distances = vec![vec![0.0; points.len()]; points.len()];
    let mut truncated = false;
    for i in 0..points.len() {
        simplices.push(Simplex {
            vertices: vec![i],
            filtration: 0.0,
            dimension: 0,
        });
        for j in i + 1..points.len() {
            let distance = points[i].distance(&points[j]);
            distances[i][j] = distance;
            distances[j][i] = distance;
            if distance <= max_radius {
                simplices.push(Simplex {
                    vertices: vec![i, j],
                    filtration: distance,
                    dimension: 1,
                });
            }
        }
    }

    'triangles: for i in 0..points.len() {
        for j in i + 1..points.len() {
            for k in j + 1..points.len() {
                let filtration = distances[i][j].max(distances[i][k]).max(distances[j][k]);
                if filtration <= max_radius {
                    simplices.push(Simplex {
                        vertices: vec![i, j, k],
                        filtration,
                        dimension: 2,
                    });
                    if simplices.len() >= max_simplices {
                        truncated = true;
                        break 'triangles;
                    }
                }
            }
        }
    }

    if !truncated {
        'tetras: for i in 0..points.len() {
            for j in i + 1..points.len() {
                for k in j + 1..points.len() {
                    for l in k + 1..points.len() {
                        let filtration = distances[i][j]
                            .max(distances[i][k])
                            .max(distances[i][l])
                            .max(distances[j][k])
                            .max(distances[j][l])
                            .max(distances[k][l]);
                        if filtration <= max_radius {
                            simplices.push(Simplex {
                                vertices: vec![i, j, k, l],
                                filtration,
                                dimension: 3,
                            });
                            if simplices.len() >= max_simplices {
                                truncated = true;
                                break 'tetras;
                            }
                        }
                    }
                }
            }
        }
    }

    simplices.sort_by(|a, b| {
        a.filtration
            .total_cmp(&b.filtration)
            .then_with(|| a.dimension.cmp(&b.dimension))
            .then_with(|| a.vertices.cmp(&b.vertices))
    });
    (simplices, truncated)
}

fn boundary_indices(simplex: &Simplex, simplex_index: &BTreeMap<Vec<usize>, usize>) -> Vec<usize> {
    if simplex.dimension == 0 {
        return Vec::new();
    }
    let mut boundary = Vec::with_capacity(simplex.vertices.len());
    for removed in 0..simplex.vertices.len() {
        let mut face = simplex.vertices.clone();
        face.remove(removed);
        if let Some(idx) = simplex_index.get(&face) {
            boundary.push(*idx);
        }
    }
    boundary.sort_unstable();
    boundary
}

fn xor_sorted(a: &[usize], b: &[usize]) -> Vec<usize> {
    let mut out = Vec::with_capacity(a.len() + b.len());
    let mut i = 0;
    let mut j = 0;
    while i < a.len() || j < b.len() {
        if j == b.len() || (i < a.len() && a[i] < b[j]) {
            out.push(a[i]);
            i += 1;
        } else if i == a.len() || b[j] < a[i] {
            out.push(b[j]);
            j += 1;
        } else {
            i += 1;
            j += 1;
        }
    }
    out
}

fn farthest_point_landmarks(points: &[Point], count: usize) -> Vec<Point> {
    farthest_point_landmark_indices(points, count)
        .into_iter()
        .map(|idx| points[idx].clone())
        .collect()
}

fn farthest_point_landmark_indices(points: &[Point], count: usize) -> Vec<usize> {
    if points.is_empty() || count == 0 {
        return Vec::new();
    }
    let mut selected = vec![0usize];
    while selected.len() < count.min(points.len()) {
        let next = (0..points.len())
            .filter(|idx| !selected.contains(idx))
            .max_by(|a, b| {
                distance_to_selected(points, *a, &selected)
                    .total_cmp(&distance_to_selected(points, *b, &selected))
            });
        if let Some(idx) = next {
            selected.push(idx);
        } else {
            break;
        }
    }
    selected
}

fn distance_to_selected(points: &[Point], idx: usize, selected: &[usize]) -> f32 {
    selected
        .iter()
        .map(|selected_idx| points[idx].distance(&points[*selected_idx]))
        .fold(f32::INFINITY, f32::min)
}

fn nearest_distance(points: &[Point], idx: usize) -> f32 {
    points
        .iter()
        .enumerate()
        .filter(|(other, _)| *other != idx)
        .map(|(_, point)| points[idx].distance(point))
        .fold(f32::INFINITY, f32::min)
}

fn find(parent: &mut [usize], idx: usize) -> usize {
    if parent[idx] != idx {
        parent[idx] = find(parent, parent[idx]);
    }
    parent[idx]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn point(coords: &[f32]) -> Point {
        Point {
            coords: coords.to_vec(),
        }
    }

    #[test]
    fn persistent_homology_tracks_h0_components() {
        let points = vec![point(&[0.0]), point(&[1.0]), point(&[3.0])];
        let diagram = persistent_homology(&points, 1.1, 10_000);

        assert!(!diagram.truncated);
        assert!(diagram.total(0) > 1.0);
        assert_eq!(diagram.total(1), 0.0);
        assert_eq!(diagram.total(2), 0.0);
    }

    #[test]
    fn persistent_homology_tracks_h1_square_cycle() {
        let points = vec![
            point(&[0.0, 0.0]),
            point(&[1.0, 0.0]),
            point(&[1.0, 1.0]),
            point(&[0.0, 1.0]),
        ];
        let diagram = persistent_homology(&points, 1.1, 10_000);

        assert!(!diagram.truncated);
        assert!(diagram.total(1) > 0.09);
        assert_eq!(diagram.total(2), 0.0);
    }

    #[test]
    fn persistent_homology_tracks_h2_octahedron_shell() {
        let points = vec![
            point(&[1.0, 0.0, 0.0]),
            point(&[-1.0, 0.0, 0.0]),
            point(&[0.0, 1.0, 0.0]),
            point(&[0.0, -1.0, 0.0]),
            point(&[0.0, 0.0, 1.0]),
            point(&[0.0, 0.0, -1.0]),
        ];
        let diagram = persistent_homology(&points, 1.5, 10_000);

        assert!(!diagram.truncated);
        assert!(diagram.total(2) > 0.08);
    }

    #[test]
    fn ph_retain_indices_keeps_requested_unique_indices() {
        let points = make_key_points(64, 16);
        let retained = ph_retain_indices(&points, 16);
        let unique = retained.iter().copied().collect::<HashSet<_>>();

        assert_eq!(retained.len(), 16);
        assert_eq!(unique.len(), 16);
        assert!(retained.iter().all(|idx| *idx < points.len()));
    }

    #[test]
    fn transformer_like_kv_cache_has_contextual_outliers() {
        let cache = make_transformer_like_kv_cache(96, 32, 6);
        let early_norm = cache.keys[0]
            .coords
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            .sqrt();
        let middle_norm = cache.keys[48]
            .coords
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            .sqrt();

        assert_eq!(cache.keys.len(), 96);
        assert_eq!(cache.values.len(), 96);
        assert_eq!(cache.queries.len(), 6);
        assert!(early_norm > middle_norm * 1.2);
    }

    #[test]
    fn kv_cache_roundtrips_through_safetensors() -> Result<()> {
        let device = Device::Cpu;
        let path = std::env::temp_dir().join(format!(
            "candle-ph-prune-kv-{}-{:?}.safetensors",
            std::process::id(),
            std::thread::current().id()
        ));
        let cache = make_transformer_like_kv_cache(32, 12, 4);

        write_kv_cache(&path, &cache, &device)?;
        let loaded = load_kv_cache(&path, &device)?;
        std::fs::remove_file(&path)?;

        assert_eq!(loaded.keys.len(), cache.keys.len());
        assert_eq!(loaded.values.len(), cache.values.len());
        assert_eq!(loaded.queries.len(), cache.queries.len());
        assert!((loaded.keys[3].coords[5] - cache.keys[3].coords[5]).abs() < 1e-6);
        assert!((loaded.values[7].coords[2] - cache.values[7].coords[2]).abs() < 1e-6);
        assert!((loaded.queries[1].coords[9] - cache.queries[1].coords[9]).abs() < 1e-6);
        Ok(())
    }
}
