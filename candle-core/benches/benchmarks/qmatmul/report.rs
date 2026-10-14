//! End-of-run terminal report for the qmatmul Criterion benchmarks.
//!
//! Charts are rendered when stderr is a terminal. Supported terminals get a native
//! pixel plot; other terminals and redirected output fall back to Unicode cells.
//! Full suites also compare compute speedup with actual encoded-weight compression.
//! Criterion already prints per-benchmark numbers, so the report stays graphical.
//!
//! Run just this suite with `cargo bench -p candle-core --bench bench_main -- qmatmul`;
//! the report draws when the dtype sweep finishes, whatever else the run covers.
//! Set `CANDLE_BENCH_CHART=always` to force the cell chart in a pipe/CI log, or
//! `CANDLE_BENCH_CHART=never` to disable the report.

use malevich::pixel::Graphics;
use malevich::scale::Palette;
use malevich::{Bars, Charset, Color, ColorMode, Dash, Frame, Plot, Range, Rule, Scale, Theme};
use serde::Deserialize;
use std::collections::BTreeMap;
use std::env;
use std::fmt::Write as _;
use std::fs::File;
use std::io::{self, IsTerminal, Read, Write as _};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Mutex, OnceLock};

#[derive(Clone, Debug, Deserialize)]
struct ConfidenceInterval {
    confidence_level: f64,
    lower_bound: f64,
    upper_bound: f64,
}

#[derive(Clone, Debug, Deserialize)]
struct Estimate {
    confidence_interval: ConfidenceInterval,
    point_estimate: f64,
}

#[derive(Debug, Deserialize)]
struct Estimates {
    mean: Estimate,
    slope: Option<Estimate>,
}

#[derive(Debug, Deserialize)]
struct CargoMetadata {
    target_directory: PathBuf,
}

#[derive(Clone, Debug)]
struct BenchResult {
    family: String,
    dtype: String,
    shape: String,
    operations_per_iter: u64,
    weight_bytes: u64,
    weight_elements: u64,
    estimate: Estimate,
}

#[derive(Default)]
struct SummaryState {
    results: Vec<BenchResult>,
    errors: Vec<String>,
}

#[derive(Clone, Copy)]
enum ChartMode {
    Auto,
    Always,
    Never,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum DTypeGroup {
    Float,
    ClassicQuant,
    KQuant,
}

static STATE: OnceLock<Mutex<SummaryState>> = OnceLock::new();
static CHART_MODE: OnceLock<ChartMode> = OnceLock::new();
static CRITERION_WRITES_ESTIMATES: OnceLock<bool> = OnceLock::new();
static CRITERION_OUTPUT_DIRECTORY: OnceLock<PathBuf> = OnceLock::new();

pub(super) fn record(
    benchmark_name: &str,
    family: &str,
    dtype: &str,
    shape: &str,
    operations_per_iter: u64,
    weight_bytes: u64,
    weight_elements: u64,
) {
    if !chart_enabled() || !criterion_writes_estimates() {
        return;
    }

    let path = criterion_output_directory()
        .join(benchmark_name)
        .join("iter")
        .join("new")
        .join("estimates.json");

    let mut state = state().lock().unwrap();
    match load_estimate(&path) {
        Ok(estimate) => {
            state
                .results
                .retain(|result| result.family != family || result.dtype != dtype);
            state.results.push(BenchResult {
                family: family.to_owned(),
                dtype: dtype.to_owned(),
                shape: shape.to_owned(),
                operations_per_iter,
                weight_bytes,
                weight_elements,
                estimate,
            });
        }
        Err(error) => state.errors.push(error),
    }
}

pub(super) fn print_summary() {
    if !chart_enabled() {
        return;
    }

    let mut state = state().lock().unwrap();
    let SummaryState { results, errors } = std::mem::take(&mut *state);
    drop(state);

    if results.is_empty() && errors.is_empty() {
        return;
    }

    let mut families = BTreeMap::<String, Vec<BenchResult>>::new();
    for result in results {
        families
            .entry(result.family.clone())
            .or_default()
            .push(result);
    }

    let stderr = io::stderr();
    let mut frame = Frame::detect_for(&stderr);
    frame.width = frame.width.min(120);
    frame.height = frame.height.clamp(12, 18);

    // Pixel escapes only belong on a real terminal. In particular, `always`
    // makes redirected benchmark reports useful without leaking image protocol
    // payloads into CI logs or files. A one-result family is table-only, so it
    // does not pay for an otherwise unnecessary capability probe either. Honor
    // NO_COLOR by retaining the monochrome Unicode renderer.
    let graphics = if stderr.is_terminal()
        && frame.color != ColorMode::Plain
        && families.values().any(|results| results.len() > 1)
    {
        Graphics::detect_for(&stderr).map(ink_weight)
    } else {
        None
    };
    let mut destination = stderr.lock();

    if !families.is_empty() {
        for (family, family_results) in families {
            let rendered = render_family(&family, family_results, &frame, graphics.as_ref());
            let _ = writeln!(destination, "\n{rendered}");
        }
    }

    if !errors.is_empty() {
        let _ = writeln!(
            destination,
            "\nqmatmul chart: {} result{} could not be loaded:",
            errors.len(),
            if errors.len() == 1 { "" } else { "s" }
        );
        for error in errors {
            let _ = writeln!(destination, "  - {error}");
        }
    }
}

/// A hairline vanishes on a Retina cell, where the terminal draws its own text
/// several device pixels thick. Scaling the ink with the cell keeps the whiskers
/// and the parity rule legible at any density.
fn ink_weight(graphics: Graphics) -> Graphics {
    let cell_height = usize::from(graphics.cell_size.1);
    graphics.stroke(((cell_height + 4) / 8).clamp(2, 6) as u8)
}

fn render_family(
    family: &str,
    mut results: Vec<BenchResult>,
    frame: &Frame,
    graphics: Option<&Graphics>,
) -> String {
    results.sort_by(|left, right| {
        effective_gflops(left)
            .0
            .total_cmp(&effective_gflops(right).0)
    });

    let separator = separator(frame);
    let confidence = results[0].estimate.confidence_interval.confidence_level * 100.0;
    let title = format!(
        "{} qmatmul{separator}effective GFLOP/s{separator}{}",
        backend_name(family),
        results[0].shape,
    );
    let mut output = if results.len() == 1 {
        title
    } else {
        render_throughput_chart(&title, &results, confidence, frame, graphics)
    };

    if let Some(tradeoff) = render_tradeoff_chart(&results, separator, frame, graphics) {
        let _ = write!(output, "\n\n{tradeoff}");
    }

    // Whatever came before ends mid-line: close it, then leave a blank one.
    let _ = write!(output, "\n\n  {}\n", verdict(&results, separator));
    output
}

/// The one line the charts cannot state outright: which format won, and what it
/// cost. Criterion's own output already carries the per-benchmark timings.
fn verdict(results: &[BenchResult], separator: &str) -> String {
    let best = results
        .last()
        .expect("a family always has at least one result");
    let rate = format_rate(effective_gflops(best).0);
    match results
        .iter()
        .find(|result| result.dtype == "F32")
        .filter(|_| best.dtype != "F32")
    {
        Some(baseline) => format!(
            "{} leads at {rate} GFLOP/s{separator}{:.2}x F32 at {} bits/weight",
            best.dtype,
            baseline.estimate.point_estimate / best.estimate.point_estimate,
            format_bits_per_weight(bits_per_weight(best)),
        ),
        None => format!("{} leads at {rate} GFLOP/s", best.dtype),
    }
}

/// The one separator the report draws itself, with an ASCII frame's fallback.
fn separator(frame: &Frame) -> &'static str {
    if frame.charset == Charset::Ascii {
        " | "
    } else {
        " \u{b7} "
    }
}

fn render_throughput_chart(
    title: &str,
    results: &[BenchResult],
    confidence: f64,
    frame: &Frame,
    graphics: Option<&Graphics>,
) -> String {
    let labels = results
        .iter()
        .map(|result| result.dtype.clone())
        .collect::<Vec<_>>();
    let throughput = results
        .iter()
        .map(|result| effective_gflops(result).0)
        .collect::<Vec<_>>();
    let lower = results
        .iter()
        .map(|result| effective_gflops(result).1)
        .collect::<Vec<_>>();
    let upper = results
        .iter()
        .map(|result| effective_gflops(result).2)
        .collect::<Vec<_>>();
    let y_max = upper.iter().copied().fold(0.0_f64, f64::max) * 1.08;

    let bars = Bars::new(labels.clone(), &throughput[..]);
    let mut plot = if frame.color == ColorMode::Plain {
        Plot::new().layer(bars.label("estimate"))
    } else {
        let (groups, palette) = dtype_palette(results, frame.theme);
        Plot::new().palette(palette).layer(bars.color_by(groups))
    };
    plot = plot
        .layer(
            Range::over(labels, &lower[..], &upper[..])
                .marker(&throughput[..])
                .color(confidence_color(frame.theme))
                .label(format!("{confidence:.0}% CI")),
        )
        .title(title)
        .y_domain(0.0, y_max);

    render_plot(&plot, frame, graphics)
}

/// Speedup and compression are both multiples of F32, so one axis carries both:
/// grouped bars put each format's payoff beside the storage it bought.
fn render_tradeoff_chart(
    results: &[BenchResult],
    separator: &str,
    frame: &Frame,
    graphics: Option<&Graphics>,
) -> Option<String> {
    if results.len() < 3 {
        return None;
    }

    let baseline = results.iter().find(|result| result.dtype == "F32")?;
    let baseline_bits = bits_per_weight(baseline);
    let compute_speedup = results
        .iter()
        .map(|result| baseline.estimate.point_estimate / result.estimate.point_estimate)
        .collect::<Vec<_>>();
    let weight_compression = results
        .iter()
        .map(|result| baseline_bits / bits_per_weight(result))
        .collect::<Vec<_>>();
    let left = (0..results.len())
        .map(|index| index as f64 - 0.22)
        .collect::<Vec<_>>();
    let right = (0..results.len())
        .map(|index| index as f64 + 0.22)
        .collect::<Vec<_>>();
    let labels = results
        .iter()
        .map(|result| result.dtype.clone())
        .collect::<Vec<_>>();
    let y_max = compute_speedup
        .iter()
        .chain(&weight_compression)
        .copied()
        .fold(0.0_f64, f64::max)
        * 1.08;

    let mut compact_frame = *frame;
    compact_frame.height = compact_frame.height.min(14);
    let plot = Plot::new()
        .x_scale(Scale::bands(labels))
        .layer(
            Rule::h(1.0)
                .dash(Dash::Dotted)
                .color(MUTED_INK)
                .label("F32 parity"),
        )
        .layer(
            Bars::at(&left[..], 0.34, &compute_speedup[..])
                .color(compute_color(frame.theme))
                .label("compute speedup"),
        )
        .layer(
            Bars::at(&right[..], 0.34, &weight_compression[..])
                .color(compression_color(frame.theme))
                .label("weight compression"),
        )
        .title(format!(
            "Compute payoff vs storage cost{separator}multiples of F32"
        ))
        .y_domain(0.0, y_max);

    Some(render_plot(&plot, &compact_frame, graphics))
}

fn render_plot(plot: &Plot<'_>, frame: &Frame, graphics: Option<&Graphics>) -> String {
    match graphics {
        Some(graphics) => plot.render_pixels(frame, graphics),
        None => plot.render(frame),
    }
}

fn dtype_palette(results: &[BenchResult], theme: Theme) -> (Vec<&'static str>, Palette) {
    let groups = results
        .iter()
        .map(|result| dtype_group(&result.dtype))
        .collect::<Vec<_>>();
    let mut order = Vec::new();
    for group in &groups {
        if !order.contains(group) {
            order.push(*group);
        }
    }

    let palette = Palette::try_from_colors(
        order
            .into_iter()
            .map(|group| dtype_color(group, theme))
            .collect(),
    )
    .expect("a rendered qmatmul chart always has at least one dtype");
    (groups.into_iter().map(DTypeGroup::label).collect(), palette)
}

impl DTypeGroup {
    fn label(self) -> &'static str {
        match self {
            Self::Float => "float",
            Self::ClassicQuant => "classic quant",
            Self::KQuant => "K-quant",
        }
    }
}

fn dtype_group(dtype: &str) -> DTypeGroup {
    match dtype {
        "F32" | "F16" | "BF16" => DTypeGroup::Float,
        dtype if dtype.ends_with('K') => DTypeGroup::KQuant,
        _ => DTypeGroup::ClassicQuant,
    }
}

// The first three categorical slots of the reference data-visualization palette,
// stepped per theme. They clear the colorblind-separation and surface-contrast
// gates as a set, which a hand-picked trio of terminal hues does not.
fn dtype_color(group: DTypeGroup, theme: Theme) -> Color {
    match (theme == Theme::LIGHT, group) {
        (false, DTypeGroup::Float) => Color::Rgb(57, 135, 229),
        (false, DTypeGroup::ClassicQuant) => Color::Rgb(217, 89, 38),
        (false, DTypeGroup::KQuant) => Color::Rgb(25, 158, 112),
        (true, DTypeGroup::Float) => Color::Rgb(42, 120, 214),
        (true, DTypeGroup::ClassicQuant) => Color::Rgb(235, 104, 52),
        (true, DTypeGroup::KQuant) => Color::Rgb(27, 175, 122),
    }
}

/// Chrome ink: readable against either theme's surface, recessive against both.
const MUTED_INK: Color = Color::Rgb(137, 135, 129);

// Uncertainty and reference marks are annotation, not identity: they wear the
// theme's ink so they never compete with a series hue for attention.
fn confidence_color(theme: Theme) -> Color {
    if theme == Theme::LIGHT {
        Color::Rgb(82, 81, 78)
    } else {
        Color::Rgb(195, 194, 183)
    }
}

fn compute_color(theme: Theme) -> Color {
    dtype_color(DTypeGroup::Float, theme)
}

fn compression_color(theme: Theme) -> Color {
    dtype_color(DTypeGroup::ClassicQuant, theme)
}

fn effective_gflops(result: &BenchResult) -> (f64, f64, f64) {
    let operations = result.operations_per_iter as f64;
    let interval = &result.estimate.confidence_interval;
    (
        operations / result.estimate.point_estimate,
        operations / interval.upper_bound,
        operations / interval.lower_bound,
    )
}

fn bits_per_weight(result: &BenchResult) -> f64 {
    result.weight_bytes as f64 * 8.0 / result.weight_elements as f64
}

fn format_rate(rate: f64) -> String {
    if rate >= 100.0 {
        format!("{rate:.1}")
    } else {
        format!("{rate:.2}")
    }
}

fn format_bits_per_weight(bits: f64) -> String {
    if bits >= 10.0 {
        format!("{bits:.1}")
    } else {
        format!("{bits:.2}")
    }
}

fn backend_name(family: &str) -> String {
    let backend = family.strip_suffix("_qmatmul").unwrap_or(family);
    match backend {
        "cpu" => "CPU".to_owned(),
        "accelerate" => "Accelerate".to_owned(),
        "mkl" => "MKL".to_owned(),
        "cuda" => "CUDA".to_owned(),
        "metal" => "Metal".to_owned(),
        other => other.to_uppercase(),
    }
}

fn load_estimate(path: &Path) -> Result<Estimate, String> {
    let file = File::open(path).map_err(|error| format!("{}: {error}", path.display()))?;
    parse_estimate(file, &path.display().to_string())
}

fn parse_estimate(reader: impl Read, source: &str) -> Result<Estimate, String> {
    let estimates: Estimates =
        serde_json::from_reader(reader).map_err(|error| format!("{source}: {error}"))?;
    let estimate = estimates.slope.unwrap_or(estimates.mean);
    let interval = &estimate.confidence_interval;

    if estimate.point_estimate.is_finite()
        && estimate.point_estimate > 0.0
        && interval.lower_bound.is_finite()
        && interval.lower_bound > 0.0
        && interval.lower_bound <= estimate.point_estimate
        && interval.upper_bound.is_finite()
        && interval.upper_bound >= estimate.point_estimate
        && interval.confidence_level.is_finite()
        && (0.0..=1.0).contains(&interval.confidence_level)
    {
        Ok(estimate)
    } else {
        Err(format!(
            "{source}: Criterion produced a non-positive or non-finite estimate"
        ))
    }
}

fn state() -> &'static Mutex<SummaryState> {
    STATE.get_or_init(|| Mutex::new(SummaryState::default()))
}

fn chart_enabled() -> bool {
    match *CHART_MODE.get_or_init(chart_mode_from_environment) {
        ChartMode::Auto => io::stderr().is_terminal(),
        ChartMode::Always => true,
        ChartMode::Never => false,
    }
}

fn chart_mode_from_environment() -> ChartMode {
    match env::var("CANDLE_BENCH_CHART") {
        Ok(value) => match value.to_ascii_lowercase().as_str() {
            "always" | "1" | "true" => ChartMode::Always,
            "never" | "0" | "false" => ChartMode::Never,
            "auto" | "" => ChartMode::Auto,
            other => {
                eprintln!(
                    "qmatmul chart: ignoring CANDLE_BENCH_CHART={other:?}; use auto, always, or never"
                );
                ChartMode::Auto
            }
        },
        Err(_) => ChartMode::Auto,
    }
}

fn criterion_writes_estimates() -> bool {
    *CRITERION_WRITES_ESTIMATES.get_or_init(|| {
        let arguments = env::args().collect::<Vec<_>>();
        arguments_write_estimates(&arguments, env::var_os("CARGO_CRITERION_PORT").is_some())
    })
}

fn arguments_write_estimates(arguments: &[String], cargo_criterion: bool) -> bool {
    if cargo_criterion {
        return false;
    }

    let has_argument = |name: &str| {
        arguments
            .iter()
            .any(|argument| argument == name || argument.starts_with(&format!("{name}=")))
    };

    has_argument("--bench")
        && !has_argument("--test")
        && !has_argument("--list")
        && !has_argument("--profile-time")
        && !has_argument("--load-baseline")
        && !has_argument("--discard-baseline")
}

fn criterion_output_directory() -> &'static PathBuf {
    CRITERION_OUTPUT_DIRECTORY.get_or_init(|| {
        if let Some(path) = env::var_os("CRITERION_HOME") {
            return PathBuf::from(path);
        }
        if let Some(path) = env::var_os("CARGO_TARGET_DIR") {
            return PathBuf::from(path).join("criterion");
        }

        cargo_target_directory()
            .map(|path| path.join("criterion"))
            .unwrap_or_else(|| PathBuf::from("target/criterion"))
    })
}

fn cargo_target_directory() -> Option<PathBuf> {
    let cargo = env::var_os("CARGO")?;
    let output = Command::new(cargo)
        .args(["metadata", "--format-version", "1"])
        .output()
        .ok()?;
    let metadata: CargoMetadata = serde_json::from_slice(&output.stdout).ok()?;
    Some(metadata.target_directory)
}

#[cfg(test)]
#[allow(dead_code)]
mod tests {
    use super::*;

    fn estimate(point: f64, lower: f64, upper: f64) -> Estimate {
        Estimate {
            point_estimate: point,
            confidence_interval: ConfidenceInterval {
                confidence_level: 0.95,
                lower_bound: lower,
                upper_bound: upper,
            },
        }
    }

    fn result(dtype: &str, point: f64, lower: f64, upper: f64) -> BenchResult {
        let weight_bytes = match dtype {
            "F32" => 1_024,
            "F16" => 512,
            "Q4_0" | "Q4K" => 144,
            _ => 256,
        };
        BenchResult {
            family: "cpu_qmatmul".to_owned(),
            dtype: dtype.to_owned(),
            shape: "lhs 1x2, rhs 2x2".to_owned(),
            operations_per_iter: 2_000,
            weight_bytes,
            weight_elements: 256,
            estimate: estimate(point, lower, upper),
        }
    }

    #[test]
    fn criterion_modes_that_do_not_persist_results_are_excluded() {
        let arguments = |tail: &[&str]| {
            std::iter::once("bench_main")
                .chain(tail.iter().copied())
                .map(str::to_owned)
                .collect::<Vec<_>>()
        };

        assert!(arguments_write_estimates(
            &arguments(&["--bench", "qmatmul", "--quick"]),
            false
        ));
        assert!(!arguments_write_estimates(
            &arguments(&["--bench", "--discard-baseline"]),
            false
        ));
        assert!(!arguments_write_estimates(
            &arguments(&["--bench", "--profile-time=2"]),
            false
        ));
        assert!(!arguments_write_estimates(&arguments(&["--bench"]), true));
    }

    #[test]
    fn parser_prefers_criterion_slope_and_falls_back_to_mean() {
        let with_slope = br#"{
            "mean": {
                "point_estimate": 20.0,
                "confidence_interval": {
                    "confidence_level": 0.95,
                    "lower_bound": 18.0,
                    "upper_bound": 22.0
                }
            },
            "slope": {
                "point_estimate": 10.0,
                "confidence_interval": {
                    "confidence_level": 0.95,
                    "lower_bound": 9.0,
                    "upper_bound": 11.0
                }
            }
        }"#;
        let without_slope = br#"{
            "mean": {
                "point_estimate": 20.0,
                "confidence_interval": {
                    "confidence_level": 0.95,
                    "lower_bound": 18.0,
                    "upper_bound": 22.0
                }
            },
            "slope": null
        }"#;

        assert_eq!(
            parse_estimate(&with_slope[..], "fixture")
                .unwrap()
                .point_estimate,
            10.0
        );
        assert_eq!(
            parse_estimate(&without_slope[..], "fixture")
                .unwrap()
                .point_estimate,
            20.0
        );
    }

    #[test]
    fn throughput_inverts_the_latency_interval() {
        let result = result("Q4K", 100.0, 80.0, 125.0);
        assert_eq!(effective_gflops(&result), (20.0, 16.0, 25.0));
    }

    #[test]
    fn summary_states_the_winner_its_rate_and_what_it_cost() {
        let rendered = render_family(
            "cpu_qmatmul",
            vec![
                result("F32", 100.0, 90.0, 110.0),
                result("Q4K", 25.0, 20.0, 30.0),
            ],
            &Frame::portable(80, 16),
            None,
        );

        assert!(rendered.contains("CPU qmatmul"));
        assert!(rendered.contains("lhs 1x2, rhs 2x2"));
        assert!(rendered.contains("Q4K leads at 80.00 GFLOP/s"));
        assert!(rendered.contains("4.00x F32 at 4.50 bits/weight"));
    }

    #[test]
    fn a_single_result_states_its_rate_instead_of_drawing_one_bar() {
        let rendered = render_family(
            "cpu_qmatmul",
            vec![result("Q4K", 25.0, 20.0, 30.0)],
            &Frame::portable(80, 16),
            None,
        );

        assert!(rendered.starts_with("CPU qmatmul"));
        assert!(rendered.contains("Q4K leads at 80.00 GFLOP/s"));
        // No baseline to compare against, and no chart worth drawing for one bar.
        assert!(!rendered.contains("F32"));
        assert!(!rendered.contains("estimate"));
        assert!(rendered.lines().count() < 4);
    }

    #[test]
    fn pixel_graphics_render_a_hybrid_sixel_chart_and_keep_the_verdict() {
        use malevich::pixel::Protocol;

        let graphics = Graphics::new(Protocol::Sixel).cell_size(4, 8);
        let rendered = render_family(
            "cpu_qmatmul",
            vec![
                result("F32", 100.0, 90.0, 110.0),
                result("Q4K", 25.0, 20.0, 30.0),
            ],
            &Frame::portable(40, 12),
            Some(&graphics),
        );

        assert!(rendered.contains("\x1bP0;1;0q"));
        assert!(rendered.contains("CPU qmatmul"));
        assert!(rendered.contains("4.00x"));
        assert!(rendered.contains("80.00"));
    }

    #[test]
    fn full_suite_adds_compute_vs_actual_weight_compression() {
        let rendered = render_family(
            "cpu_qmatmul",
            vec![
                result("F32", 100.0, 90.0, 110.0),
                result("F16", 80.0, 70.0, 90.0),
                result("Q4K", 25.0, 20.0, 30.0),
            ],
            &Frame::portable(80, 16),
            None,
        );

        assert!(rendered.contains("Compute payoff vs storage cost"));
        assert!(rendered.contains("compute speedup"));
        assert!(rendered.contains("weight compression"));
        assert_eq!(bits_per_weight(&result("Q4K", 25.0, 20.0, 30.0)), 4.5);
    }

    #[test]
    fn colored_chart_uses_semantic_dark_theme_hues() {
        let mut frame = Frame::portable(80, 16);
        frame.color = ColorMode::TrueColor;
        let rendered = render_family(
            "cpu_qmatmul",
            vec![
                result("F32", 100.0, 90.0, 110.0),
                result("Q4_0", 50.0, 45.0, 55.0),
                result("Q4K", 25.0, 20.0, 30.0),
            ],
            &frame,
            None,
        );

        for rgb in ["57;135;229", "217;89;38", "25;158;112", "195;194;183"] {
            assert!(rendered.contains(rgb), "missing RGB color {rgb}");
        }
        assert!(rendered.contains("float"));
        assert!(rendered.contains("classic quant"));
        assert!(rendered.contains("K-quant"));
    }
}
