//! Application state for the interactive TUI.

use crate::analysis::{ComparisonAlignment, ComparisonMetrics, ModelMetrics, VarianceSpectrum};
use crate::models::registry::{self, RegistryEntry};
use ndarray::Array1;
use ratatui::widgets::TableState;
use std::path::{Path, PathBuf};

/// Active tab.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tab {
    Dashboard,
    Inspector,
    Compare,
    Spectrum,
    Help,
}

impl Tab {
    pub const ALL: &[Tab] = &[
        Tab::Dashboard,
        Tab::Inspector,
        Tab::Compare,
        Tab::Spectrum,
        Tab::Help,
    ];

    pub fn index(self) -> usize {
        match self {
            Tab::Dashboard => 0,
            Tab::Inspector => 1,
            Tab::Compare => 2,
            Tab::Spectrum => 3,
            Tab::Help => 4,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Tab::Dashboard => "Dashboard",
            Tab::Inspector => "Inspector",
            Tab::Compare => "Compare",
            Tab::Spectrum => "Spectrum",
            Tab::Help => "Help",
        }
    }

    pub fn next(self) -> Tab {
        Tab::ALL[(self.index() + 1) % Tab::ALL.len()]
    }

    pub fn prev(self) -> Tab {
        let len = Tab::ALL.len();
        Tab::ALL[(self.index() + len - 1) % len]
    }
}

// ── File browser ────────────────────────────────────────────────────────────

/// A filesystem entry shown in the file browser.
pub struct FsEntry {
    pub name: String,
    pub path: PathBuf,
    pub is_dir: bool,
    pub is_image: bool,
    pub size: u64,
}

/// Interactive file browser state.
pub struct FileBrowser {
    pub active: bool,
    pub current_dir: PathBuf,
    pub entries: Vec<FsEntry>,
    pub selected: usize,
    pub scroll: u16,
    pub input_active: bool,
    pub input_buffer: String,
}

const IMAGE_EXTENSIONS: &[&str] = &["jpg", "jpeg", "png", "bmp", "webp", "tiff", "tif"];

impl FileBrowser {
    pub fn new(start_dir: PathBuf) -> Self {
        let mut browser = Self {
            active: false,
            current_dir: start_dir,
            entries: Vec::new(),
            selected: 0,
            scroll: 0,
            input_active: false,
            input_buffer: String::new(),
        };
        browser.refresh();
        browser
    }

    pub fn refresh(&mut self) {
        self.entries.clear();

        // Parent directory entry
        if let Some(parent) = self.current_dir.parent() {
            self.entries.push(FsEntry {
                name: "..".to_string(),
                path: parent.to_path_buf(),
                is_dir: true,
                is_image: false,
                size: 0,
            });
        }

        if let Ok(read_dir) = std::fs::read_dir(&self.current_dir) {
            let mut dir_entries: Vec<FsEntry> = read_dir
                .filter_map(|e| e.ok())
                .filter_map(|e| {
                    let path = e.path();
                    let name = e.file_name().to_string_lossy().to_string();
                    if name.starts_with('.') {
                        return None; // skip hidden
                    }
                    let is_dir = path.is_dir();
                    let ext = path
                        .extension()
                        .and_then(|e| e.to_str())
                        .unwrap_or("")
                        .to_lowercase();
                    let is_image = IMAGE_EXTENSIONS.contains(&ext.as_str());
                    if !is_dir && !is_image {
                        return None; // only show dirs and images
                    }
                    let size = e.metadata().map(|m| m.len()).unwrap_or(0);
                    Some(FsEntry {
                        name,
                        path,
                        is_dir,
                        is_image,
                        size,
                    })
                })
                .collect();

            dir_entries.sort_by(|a, b| {
                b.is_dir
                    .cmp(&a.is_dir)
                    .then(a.name.to_lowercase().cmp(&b.name.to_lowercase()))
            });
            self.entries.extend(dir_entries);
        }

        self.selected = 0;
        self.scroll = 0;
    }

    pub fn navigate_to(&mut self, path: PathBuf) {
        if path.is_dir() {
            self.current_dir = path;
            self.refresh();
        }
    }

    pub fn go_up(&mut self) {
        if let Some(parent) = self.current_dir.parent() {
            let parent = parent.to_path_buf();
            self.navigate_to(parent);
        }
    }

    pub fn select_down(&mut self) {
        if !self.entries.is_empty() {
            self.selected = (self.selected + 1).min(self.entries.len() - 1);
        }
    }

    pub fn select_up(&mut self) {
        self.selected = self.selected.saturating_sub(1);
    }

    pub fn selected_entry(&self) -> Option<&FsEntry> {
        self.entries.get(self.selected)
    }

    pub fn toggle_input(&mut self) {
        self.input_active = !self.input_active;
        if self.input_active {
            self.input_buffer = self.current_dir.to_string_lossy().to_string();
        }
    }
}

// ── App state ───────────────────────────────────────────────────────────────

/// Full TUI application state.
pub struct App {
    pub tab: Tab,
    pub running: bool,
    pub demo_mode: bool,

    /// Model registry entries.
    pub models: Vec<RegistryEntry>,
    /// Currently selected model index.
    pub selected_model: usize,
    /// Table widget state for model list.
    pub model_table_state: TableState,

    /// Per-model analysis metrics (one per model in `models`).
    pub metrics: Vec<ModelMetrics>,
    /// Pairwise comparison metrics.
    pub comparisons: Vec<ComparisonMetrics>,
    /// Per-model variance spectra, keyed by model name.
    pub spectra: Vec<(String, VarianceSpectrum)>,

    /// Image being analysed (if any).
    pub image_path: Option<PathBuf>,
    /// Pre-resized thumbnail for terminal preview.
    pub image_thumbnail: Option<image::RgbImage>,
    /// File browser state.
    pub file_browser: FileBrowser,

    /// Scroll offsets for various views.
    pub inspector_scroll: u16,
    pub spectrum_scroll: u16,
    pub help_scroll: u16,
    pub compare_scroll: u16,
}

impl App {
    /// Create an app from real analysis data.
    pub fn new(
        image_path: Option<PathBuf>,
        metrics: Vec<ModelMetrics>,
        comparisons: Vec<ComparisonMetrics>,
        spectra: Vec<(String, VarianceSpectrum)>,
    ) -> Self {
        let models = registry::registry();
        let mut model_table_state = TableState::default();
        model_table_state.select(Some(0));
        let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));

        Self {
            tab: Tab::Dashboard,
            running: true,
            demo_mode: false,
            models,
            selected_model: 0,
            model_table_state,
            metrics,
            comparisons,
            spectra,
            image_path,
            image_thumbnail: None,
            file_browser: FileBrowser::new(cwd),
            inspector_scroll: 0,
            spectrum_scroll: 0,
            help_scroll: 0,
            compare_scroll: 0,
        }
    }

    /// Create an app pre-loaded with realistic demo data.
    pub fn demo() -> Self {
        let models = registry::registry();
        let metrics = demo_metrics();
        let comparisons = demo_comparisons();
        let spectra = demo_spectra();
        let mut model_table_state = TableState::default();
        model_table_state.select(Some(0));
        let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));

        Self {
            tab: Tab::Dashboard,
            running: true,
            demo_mode: true,
            models,
            selected_model: 0,
            model_table_state,
            metrics,
            comparisons,
            spectra,
            image_path: None,
            image_thumbnail: None,
            file_browser: FileBrowser::new(cwd),
            inspector_scroll: 0,
            spectrum_scroll: 0,
            help_scroll: 0,
            compare_scroll: 0,
        }
    }

    pub fn select_next_model(&mut self) {
        if self.models.is_empty() {
            return;
        }
        self.selected_model = (self.selected_model + 1) % self.models.len();
        self.model_table_state.select(Some(self.selected_model));
    }

    pub fn select_prev_model(&mut self) {
        if self.models.is_empty() {
            return;
        }
        let len = self.models.len();
        self.selected_model = (self.selected_model + len - 1) % len;
        self.model_table_state.select(Some(self.selected_model));
    }

    /// Return the `ModelMetrics` for the currently selected model (if any).
    pub fn selected_metrics(&self) -> Option<&ModelMetrics> {
        if self.selected_model < self.models.len() {
            let name = &self.models[self.selected_model].info.name;
            self.metrics.iter().find(|m| &m.model_name == name)
        } else {
            None
        }
    }

    /// Return the `VarianceSpectrum` for the currently selected model.
    pub fn selected_spectrum(&self) -> Option<&VarianceSpectrum> {
        if self.selected_model < self.models.len() {
            let name = &self.models[self.selected_model].info.name;
            self.spectra.iter().find(|(n, _)| n == name).map(|(_, s)| s)
        } else {
            None
        }
    }

    /// Load an image from disk for preview.
    pub fn load_image(&mut self, path: &Path) {
        if let Ok(img) = image::open(path) {
            self.image_path = Some(path.to_path_buf());
            self.image_thumbnail = Some(
                img.resize(400, 400, image::imageops::FilterType::Triangle)
                    .to_rgb8(),
            );
        }
    }
}

// ── Demo data generators ────────────────────────────────────────────────────

fn demo_metrics() -> Vec<ModelMetrics> {
    vec![
        ModelMetrics {
            model_name: "dinov2-vit-l14".into(),
            n_patches: 256,
            embed_dim: 1024,
            effective_rank: 512,
            dead_dimensions: 12,
            patch_entropy: 2.34,
            attention_gini: None,
            cls_l2_norm: Some(15.2),
            patch_norm_mean: 8.21,
            patch_norm_std: 1.13,
            top10_variance_pct: 62.3,
            components_90pct: 47,
        },
        ModelMetrics {
            model_name: "mae-vit-l16".into(),
            n_patches: 196,
            embed_dim: 1024,
            effective_rank: 398,
            dead_dimensions: 28,
            patch_entropy: 1.89,
            attention_gini: None,
            cls_l2_norm: None,
            patch_norm_mean: 6.54,
            patch_norm_std: 2.37,
            top10_variance_pct: 78.5,
            components_90pct: 31,
        },
        ModelMetrics {
            model_name: "clip-vit-l14".into(),
            n_patches: 256,
            embed_dim: 1024,
            effective_rank: 467,
            dead_dimensions: 19,
            patch_entropy: 2.15,
            attention_gini: None,
            cls_l2_norm: Some(12.8),
            patch_norm_mean: 7.92,
            patch_norm_std: 1.44,
            top10_variance_pct: 71.2,
            components_90pct: 38,
        },
        ModelMetrics {
            model_name: "ijepa-vit-h14".into(),
            n_patches: 256,
            embed_dim: 1280,
            effective_rank: 723,
            dead_dimensions: 8,
            patch_entropy: 2.51,
            attention_gini: None,
            cls_l2_norm: Some(18.7),
            patch_norm_mean: 9.45,
            patch_norm_std: 0.92,
            top10_variance_pct: 53.8,
            components_90pct: 67,
        },
        ModelMetrics {
            model_name: "siglip-so400m".into(),
            n_patches: 256,
            embed_dim: 1152,
            effective_rank: 501,
            dead_dimensions: 15,
            patch_entropy: 2.22,
            attention_gini: None,
            cls_l2_norm: Some(14.1),
            patch_norm_mean: 8.03,
            patch_norm_std: 1.28,
            top10_variance_pct: 67.9,
            components_90pct: 43,
        },
    ]
}

fn demo_comparisons() -> Vec<ComparisonMetrics> {
    let names = [
        "dinov2-vit-l14",
        "mae-vit-l16",
        "clip-vit-l14",
        "ijepa-vit-h14",
        "siglip-so400m",
    ];
    let patch_counts: [usize; 5] = [256, 196, 256, 256, 256];
    let cka = [
        [1.0, 0.63, 0.82, 0.87, 0.75],
        [0.0, 1.0, 0.45, 0.58, 0.51],
        [0.0, 0.0, 1.0, 0.69, 0.91],
        [0.0, 0.0, 0.0, 1.0, 0.72],
        [0.0, 0.0, 0.0, 0.0, 1.0],
    ];
    let knn = [
        [1.0, 0.42, 0.67, 0.71, 0.58],
        [0.0, 1.0, 0.31, 0.38, 0.35],
        [0.0, 0.0, 1.0, 0.53, 0.78],
        [0.0, 0.0, 0.0, 1.0, 0.56],
        [0.0, 0.0, 0.0, 0.0, 1.0],
    ];
    let cls_sim = [
        [1.0, -1.0, 0.84, 0.79, 0.72],
        [0.0, -1.0, -1.0, -1.0, -1.0],
        [0.0, 0.0, 1.0, 0.68, 0.88],
        [0.0, 0.0, 0.0, 1.0, 0.65],
        [0.0, 0.0, 0.0, 0.0, 1.0],
    ];

    let mut out = Vec::new();
    for i in 0..names.len() {
        for j in (i + 1)..names.len() {
            let pc_a = patch_counts[i];
            let pc_b = patch_counts[j];
            let compared = pc_a.min(pc_b);
            out.push(ComparisonMetrics {
                model_a: names[i].to_string(),
                model_b: names[j].to_string(),
                alignment: ComparisonAlignment {
                    patch_count_a: pc_a,
                    patch_count_b: pc_b,
                    compared_patch_count: compared,
                    note: if pc_a != pc_b {
                        Some(format!("Truncated to {} patches", compared))
                    } else {
                        None
                    },
                },
                cls_cosine_sim: if cls_sim[i][j] < 0.0 {
                    None
                } else {
                    Some(cls_sim[i][j])
                },
                linear_cka: cka[i][j],
                knn_overlap_k10: knn[i][j],
                mean_patch_correspondence: if i == 1 || j == 1 {
                    None
                } else {
                    Some(0.3 + 0.5 * cka[i][j])
                },
                metric_caveats: Vec::new(),
            });
        }
    }
    out
}

fn generate_spectrum(alpha: f32, k: usize) -> VarianceSpectrum {
    let raw: Vec<f32> = (0..k).map(|i| (-alpha * i as f32).exp()).collect();
    let sum: f32 = raw.iter().sum();
    let ratios: Vec<f32> = raw.iter().map(|&r| r / sum).collect();
    let cumulative: Vec<f32> = ratios
        .iter()
        .scan(0.0_f32, |acc, &r| {
            *acc += r;
            Some(*acc)
        })
        .collect();
    let components_90pct = cumulative
        .iter()
        .position(|&c| c >= 0.90)
        .map(|i| i + 1)
        .unwrap_or(k);
    let components_99pct = cumulative
        .iter()
        .position(|&c| c >= 0.99)
        .map(|i| i + 1)
        .unwrap_or(k);
    let top10_concentration: f32 = ratios.iter().take(10).sum();

    // Simulate eigenvalues scaled by a plausible total variance
    let explained_variance: Vec<f32> = raw.iter().map(|&r| r * 100.0).collect();

    VarianceSpectrum {
        explained_variance: Array1::from_vec(explained_variance),
        ratios: Array1::from_vec(ratios),
        cumulative: Array1::from_vec(cumulative),
        components_90pct,
        components_99pct,
        top10_concentration,
    }
}

fn demo_spectra() -> Vec<(String, VarianceSpectrum)> {
    vec![
        ("dinov2-vit-l14".into(), generate_spectrum(0.12, 32)),
        ("mae-vit-l16".into(), generate_spectrum(0.22, 32)),
        ("clip-vit-l14".into(), generate_spectrum(0.18, 32)),
        ("ijepa-vit-h14".into(), generate_spectrum(0.10, 32)),
        ("siglip-so400m".into(), generate_spectrum(0.16, 32)),
    ]
}
