//! Help view — educational ML reference guide.

use crate::tui::app::App;
use crate::tui::theme;
use ratatui::prelude::*;
use ratatui::widgets::*;

pub fn draw(frame: &mut Frame, area: Rect, app: &App) {
    let outer = Block::bordered()
        .title(" ML Reference Guide ")
        .title_style(theme::title_style())
        .border_type(BorderType::Rounded)
        .border_style(Style::new().fg(theme::FG_DIM));
    let inner = outer.inner(area);
    frame.render_widget(outer, area);

    let content = build_help_content();
    let paragraph = Paragraph::new(Text::from(content))
        .scroll((app.help_scroll, 0))
        .wrap(Wrap { trim: false });
    frame.render_widget(paragraph, inner);
}

fn build_help_content() -> Vec<Line<'static>> {
    let mut lines: Vec<Line<'static>> = Vec::new();

    // ── SSL Methods ─────────────────────────────────────────────────────

    section_header(&mut lines, "SELF-SUPERVISED LEARNING METHODS");

    method_entry(
        &mut lines,
        "DINO / DINOv2",
        theme::MODEL_COLORS[0],
        &[
            "Self-distillation with no labels. A student network learns to match",
            "a momentum-updated teacher's output distribution. Produces features",
            "that capture semantic similarity without any supervision.",
            "",
            "Key insight: Self-distillation creates a natural clustering of the",
            "representation space, enabling zero-shot segmentation.",
            "DINOv2 adds iBOT masked image modeling for stronger local features.",
        ],
    );

    method_entry(
        &mut lines,
        "MAE (Masked Autoencoder)",
        theme::MODEL_COLORS[1],
        &[
            "Reconstructive self-supervision: mask 75% of image patches, then",
            "train the encoder + decoder to reconstruct the missing pixels.",
            "The encoder learns to extract features that are predictive of the",
            "full image content from only 25% of the visual information.",
            "",
            "Key insight: Unlike contrastive methods, MAE forces the model to",
            "understand pixel-level structure. Encoder-only features tend to be",
            "lower-rank but capture fine-grained spatial information.",
        ],
    );

    method_entry(
        &mut lines,
        "CLIP (Contrastive Language-Image Pre-training)",
        theme::MODEL_COLORS[2],
        &[
            "Contrastive learning between image and text pairs. Images and their",
            "captions are mapped to a shared embedding space where matching pairs",
            "are pulled together and non-matching pairs pushed apart.",
            "",
            "Key insight: Text supervision injects semantic structure that pure",
            "visual self-supervision lacks. CLIP features are highly aligned",
            "with human-understandable concepts but may miss fine spatial detail.",
        ],
    );

    method_entry(
        &mut lines,
        "I-JEPA (Image Joint-Embedding Predictive Architecture)",
        theme::MODEL_COLORS[3],
        &[
            "Predicts abstract representations of masked image regions from",
            "context. Unlike MAE, it predicts in representation space rather",
            "than pixel space, avoiding low-level reconstruction shortcuts.",
            "",
            "Key insight: Prediction in latent space encourages the model to",
            "capture high-level semantics. Typically produces the most",
            "distributed (high-rank) representations among SSL methods.",
        ],
    );

    method_entry(
        &mut lines,
        "SigLIP (Sigmoid Language-Image Pre-training)",
        theme::MODEL_COLORS[4],
        &[
            "A variant of CLIP that replaces the softmax contrastive loss with",
            "a sigmoid (binary) loss. Each image-text pair is independently",
            "classified as matching or not, enabling larger batch training.",
            "",
            "Key insight: The sigmoid loss is more stable and scalable than",
            "softmax-based contrastive learning. Features are similar to CLIP",
            "but often show better calibration and downstream performance.",
        ],
    );

    // ── Metrics ─────────────────────────────────────────────────────────

    section_header(&mut lines, "UNDERSTANDING METRICS");

    metric_entry(
        &mut lines,
        "Effective Rank",
        &[
            "The number of significant dimensions in the representation.",
            "Computed by thresholding eigenvalues of the covariance matrix.",
            "",
            "  High rank  → rich, distributed features (desirable)",
            "  Low rank   → dimensional collapse (problematic)",
            "  Target     → >50% of embed_dim for healthy SSL",
        ],
    );

    metric_entry(
        &mut lines,
        "Dead Dimensions",
        &[
            "Embedding dimensions with near-zero variance across patches.",
            "These dimensions carry no discriminative information.",
            "",
            "  0 dead dims  → full utilization (optimal)",
            "  >10% dead    → significant waste, possible collapse",
        ],
    );

    metric_entry(
        &mut lines,
        "Patch Entropy",
        &[
            "Shannon entropy of patch token cluster assignments (k-means).",
            "Measures diversity of spatial representations.",
            "",
            "  High (>2.0)  → diverse, discriminative patches",
            "  Low  (<1.0)  → redundant, homogeneous patches",
        ],
    );

    metric_entry(
        &mut lines,
        "CLS L2 Norm",
        &[
            "Magnitude of the [CLS] global representation token.",
            "Not all architectures produce a CLS token (e.g., MAE encoder).",
            "",
            "  Typical range: 10–25 for ViT-L models",
            "  Very high norms may indicate feature explosion",
        ],
    );

    metric_entry(
        &mut lines,
        "Top-10 Variance %",
        &[
            "Fraction of total variance explained by the first 10 PCA components.",
            "Indicates how concentrated information is in a few directions.",
            "",
            "  >80%   → highly concentrated (potential collapse)",
            "  50–80% → moderate (typical for well-trained models)",
            "  <50%   → well-distributed (JEPA-family models)",
        ],
    );

    metric_entry(
        &mut lines,
        "Components@90%",
        &[
            "Number of PCA components needed to explain 90% of variance.",
            "Directly measures the intrinsic dimensionality of representations.",
            "",
            "  High count → high-dimensional, rich representation",
            "  Low count  → low-dimensional, potentially collapsed",
        ],
    );

    // ── Cross-model metrics ─────────────────────────────────────────────

    section_header(&mut lines, "CROSS-MODEL COMPARISON METRICS");

    metric_entry(
        &mut lines,
        "Centered Kernel Alignment (CKA)",
        &[
            "Measures similarity between two representations invariant to",
            "orthogonal transforms and isotropic scaling.",
            "",
            "  CKA = 1.0  → identical representations (up to transform)",
            "  CKA = 0.0  → completely dissimilar",
            "  >0.8       → highly similar (often same training objective)",
            "  <0.3       → fundamentally different representations",
            "",
            "CKA is the gold standard for comparing neural network features",
            "across architectures and training methods.",
        ],
    );

    metric_entry(
        &mut lines,
        "k-NN Overlap",
        &[
            "Fraction of k-nearest neighbors shared between two models.",
            "Tests whether models agree on which patches are similar.",
            "",
            "  100%  → identical neighborhood structure",
            "  >50%  → strong agreement on local geometry",
            "  <20%  → fundamentally different local structure",
        ],
    );

    metric_entry(
        &mut lines,
        "CLS Cosine Similarity",
        &[
            "Cosine similarity between the [CLS] tokens of two models.",
            "Only available when both models have CLS tokens of equal dimension.",
            "",
            "  >0.8  → aligned global representations",
            "  <0.3  → orthogonal global representations",
        ],
    );

    metric_entry(
        &mut lines,
        "Patch Correspondence",
        &[
            "Mean cosine similarity of optimally matched patch pairs (Hungarian).",
            "Measures how well spatial tokens can be aligned between models.",
            "",
            "  Only available for models with the same embedding dimension.",
        ],
    );

    // ── Keyboard shortcuts ──────────────────────────────────────────────

    section_header(&mut lines, "KEYBOARD SHORTCUTS");

    shortcut_group(
        &mut lines,
        "Navigation",
        &[
            ("Tab / →", "Next tab"),
            ("Shift+Tab / ←", "Previous tab"),
            ("1–5", "Jump to tab"),
        ],
    );

    shortcut_group(
        &mut lines,
        "Model Selection",
        &[
            ("↑ / k", "Previous model"),
            ("↓ / j", "Next model"),
            ("Enter", "Open in Inspector"),
        ],
    );

    shortcut_group(
        &mut lines,
        "Scrolling",
        &[
            ("↑ / k", "Scroll up"),
            ("↓ / j", "Scroll down"),
            ("g", "Scroll to top"),
            ("G", "Scroll to bottom"),
        ],
    );

    shortcut_group(
        &mut lines,
        "General",
        &[("?", "Toggle help"), ("q / Esc", "Quit")],
    );

    // Footer
    lines.push(Line::from(""));
    lines.push(Line::from(vec![
        Span::styled("  latent-inspector v0.1.0 — ", theme::dim_style()),
        Span::styled(
            "github.com/AbdelStark/latent-inspector",
            Style::new().fg(theme::BLUE),
        ),
    ]));
    lines.push(Line::from(""));

    lines
}

// ── Content helpers ─────────────────────────────────────────────────────────

fn section_header(lines: &mut Vec<Line<'static>>, title: &str) {
    lines.push(Line::from(""));
    let separator = "─".repeat(60);
    lines.push(Line::from(vec![Span::styled(
        format!("  {} ", title),
        Style::new().fg(theme::BLUE).bold(),
    )]));
    lines.push(Line::from(vec![Span::styled(
        format!("  {separator}"),
        Style::new().fg(theme::FG_DIM),
    )]));
    lines.push(Line::from(""));
}

fn method_entry(lines: &mut Vec<Line<'static>>, name: &str, color: Color, description: &[&str]) {
    lines.push(Line::from(vec![
        Span::styled("  ◆ ", Style::new().fg(color)),
        Span::styled(name.to_string(), Style::new().fg(color).bold()),
    ]));
    for desc in description {
        lines.push(Line::from(vec![Span::styled(
            format!("    {desc}"),
            theme::value_style(),
        )]));
    }
    lines.push(Line::from(""));
}

fn metric_entry(lines: &mut Vec<Line<'static>>, name: &str, description: &[&str]) {
    lines.push(Line::from(vec![
        Span::styled("  ▸ ", Style::new().fg(theme::CYAN)),
        Span::styled(name.to_string(), Style::new().fg(theme::CYAN).bold()),
    ]));
    for desc in description {
        if desc.starts_with("  ") {
            // Indented lines get highlight color
            lines.push(Line::from(vec![Span::styled(
                format!("    {desc}"),
                Style::new().fg(theme::FG),
            )]));
        } else {
            lines.push(Line::from(vec![Span::styled(
                format!("    {desc}"),
                theme::value_style(),
            )]));
        }
    }
    lines.push(Line::from(""));
}

fn shortcut_group(lines: &mut Vec<Line<'static>>, group: &str, shortcuts: &[(&str, &str)]) {
    lines.push(Line::from(vec![Span::styled(
        format!("  {group}"),
        Style::new().fg(theme::PURPLE).bold(),
    )]));
    for (key, desc) in shortcuts {
        lines.push(Line::from(vec![
            Span::styled(format!("    {:<20}", key), theme::key_style()),
            Span::styled(desc.to_string(), theme::value_style()),
        ]));
    }
    lines.push(Line::from(""));
}
