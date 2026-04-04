//! Inspector view — deep-dive analysis of a single model's representation.

use crate::tui::app::App;
use crate::tui::theme::{self, dim_color};
use ratatui::prelude::*;
use ratatui::widgets::*;

pub fn draw(frame: &mut Frame, area: Rect, app: &App) {
    let model_name = if app.selected_model < app.models.len() {
        &app.models[app.selected_model].info.name
    } else {
        "—"
    };

    let outer = Block::bordered()
        .title(format!(" Inspector — {} ", model_name))
        .title_style(theme::title_style())
        .border_type(BorderType::Rounded)
        .border_style(Style::new().fg(theme::FG_DIM));
    let inner = outer.inner(area);
    frame.render_widget(outer, area);

    let metrics = match app.selected_metrics() {
        Some(m) => m,
        None => {
            let msg = Paragraph::new(vec![
                Line::from(""),
                Line::from(Span::styled(
                    "  No analysis data available.",
                    theme::dim_style(),
                )),
                Line::from(""),
                Line::from(Span::styled(
                    "  Launch with --image <path> to analyse a real image,",
                    theme::dim_style(),
                )),
                Line::from(Span::styled(
                    "  or use --demo to see sample data.",
                    theme::dim_style(),
                )),
            ]);
            frame.render_widget(msg, inner);
            return;
        }
    };

    let chunks = Layout::vertical([
        Constraint::Length(18), // Metrics gauges
        Constraint::Fill(1),    // Variance spectrum
    ])
    .split(inner);

    draw_metrics_panel(frame, chunks[0], metrics, app);
    draw_variance_bars(frame, chunks[1], app);
}

fn draw_metrics_panel(frame: &mut Frame, area: Rect, m: &crate::analysis::ModelMetrics, app: &App) {
    let block = Block::bordered()
        .title(" Representation Health ")
        .title_style(theme::heading_style())
        .border_type(BorderType::Rounded)
        .border_style(Style::new().fg(theme::FG_DIM));
    let inner = block.inner(area);
    frame.render_widget(block, area);

    let color = if app.selected_model < app.models.len() {
        theme::model_color(app.selected_model)
    } else {
        theme::BLUE
    };

    let gauge_width = inner.width.saturating_sub(44) as usize;

    let lines = vec![
        gauge_line(
            "Effective Rank",
            &format!("{}/{}", m.effective_rank, m.embed_dim),
            m.effective_rank as f32 / m.embed_dim as f32,
            gauge_width,
            true,
            "Higher → richer representation",
        ),
        gauge_line(
            "Dead Dimensions",
            &format!("{}/{}", m.dead_dimensions, m.embed_dim),
            m.dead_dimensions as f32 / m.embed_dim as f32,
            gauge_width,
            false,
            "Lower → less dimensional waste",
        ),
        gauge_line(
            "Patch Entropy",
            &format!("{:.3}", m.patch_entropy),
            (m.patch_entropy / 3.0).min(1.0),
            gauge_width,
            true,
            "Higher → more diverse patches",
        ),
        gauge_line(
            "CLS L2 Norm",
            &m.cls_l2_norm
                .map(|v| format!("{:.1}", v))
                .unwrap_or_else(|| "N/A (no CLS)".into()),
            m.cls_l2_norm.map(|v| (v / 25.0).min(1.0)).unwrap_or(0.0),
            gauge_width,
            true,
            "Magnitude of global token",
        ),
        gauge_line(
            "Patch Norm μ±σ",
            &format!("{:.2} ± {:.2}", m.patch_norm_mean, m.patch_norm_std),
            (m.patch_norm_mean / 15.0).min(1.0),
            gauge_width,
            true,
            "Low σ → uniform activation",
        ),
        gauge_line(
            "Top-10 Var%",
            &format!("{:.1}%", m.top10_variance_pct),
            m.top10_variance_pct / 100.0,
            gauge_width,
            false,
            "<50% distributed, >80% concentrated",
        ),
        gauge_line(
            "Components@90%",
            &format!("{}", m.components_90pct),
            (m.components_90pct as f32 / m.embed_dim as f32).min(1.0),
            gauge_width,
            true,
            "Higher → more dims needed → richer",
        ),
        gauge_line(
            "Patch isotropy",
            &format!("{:.3}", m.patch_isotropy),
            m.patch_isotropy,
            gauge_width,
            true,
            "0=collapsed, 1=uniform spread",
        ),
        gauge_line(
            "Patch uniformity",
            &format!("{:.2}", m.patch_uniformity),
            (-m.patch_uniformity / 4.0).clamp(0.0, 1.0),
            gauge_width,
            true,
            "More negative → better spread",
        ),
        gauge_line(
            "Spatial coherence",
            &m.spatial_coherence
                .map(|v| format!("{:.3}", v))
                .unwrap_or_else(|| "N/A".into()),
            m.spatial_coherence.unwrap_or(0.0),
            gauge_width,
            true,
            "0=independent, 1=smooth/segmented",
        ),
        gauge_line(
            "RankMe",
            &format!("{:.1}", m.rankme),
            (m.rankme / m.embed_dim as f32).min(1.0),
            gauge_width,
            true,
            "Smooth eff. rank (higher=richer)",
        ),
        gauge_line(
            "Spectral decay",
            &m.spectral_decay
                .map(|v| format!("{:.2}", v))
                .unwrap_or_else(|| "N/A".into()),
            m.spectral_decay
                .map(|v| (v / 4.0).clamp(0.0, 1.0))
                .unwrap_or(0.0),
            gauge_width,
            false,
            "Lower → more uniform spread",
        ),
    ];

    let mut text_lines: Vec<Line> = Vec::new();
    for (label_line, bar_line) in &lines {
        text_lines.push(label_line.clone());
        text_lines.push(bar_line.clone());
    }

    // Add model color accent
    text_lines.insert(
        0,
        Line::from(vec![
            Span::styled("  ◆ ", Style::new().fg(color)),
            Span::styled(
                format!("{} patches × {}d embedding", m.n_patches, m.embed_dim),
                Style::new().fg(color),
            ),
        ]),
    );

    let paragraph = Paragraph::new(Text::from(text_lines)).scroll((app.inspector_scroll, 0));
    frame.render_widget(paragraph, inner);
}

fn draw_variance_bars(frame: &mut Frame, area: Rect, app: &App) {
    let block = Block::bordered()
        .title(" Variance Spectrum (PCA) ")
        .title_style(theme::heading_style())
        .border_type(BorderType::Rounded)
        .border_style(Style::new().fg(theme::FG_DIM));
    let inner = block.inner(area);
    frame.render_widget(block, area);

    let spectrum = match app.selected_spectrum() {
        Some(s) => s,
        None => return,
    };

    let color = if app.selected_model < app.models.len() {
        theme::model_color(app.selected_model)
    } else {
        theme::BLUE
    };

    let bar_area_width = inner.width.saturating_sub(30) as f32;
    let max_ratio = spectrum
        .ratios
        .iter()
        .cloned()
        .fold(0.0_f32, f32::max)
        .max(0.001);

    let mut lines: Vec<Line> = Vec::new();

    let visible = (inner.height as usize).saturating_sub(2);
    let offset = app.spectrum_scroll as usize;

    for (i, (&ratio, &cum)) in spectrum
        .ratios
        .iter()
        .zip(spectrum.cumulative.iter())
        .enumerate()
        .skip(offset)
        .take(visible)
    {
        let bar_len = ((ratio / max_ratio) * bar_area_width) as usize;
        let bar = "█".repeat(bar_len);
        let empty = (bar_area_width as usize).saturating_sub(bar_len);
        let empty_bar = "░".repeat(empty);

        // Color intensity fades with component index
        let intensity = 1.0 - (i as f32 / spectrum.ratios.len() as f32) * 0.6;
        let bar_color = dim_color(color, intensity);

        let cum_marker = if cum >= 0.90 && (i == 0 || spectrum.cumulative[i - 1] < 0.90) {
            " ← 90%"
        } else if cum >= 0.99 && (i == 0 || spectrum.cumulative[i - 1] < 0.99) {
            " ← 99%"
        } else {
            ""
        };

        lines.push(Line::from(vec![
            Span::styled(format!("  PC{:02} ", i + 1), theme::dim_style()),
            Span::styled(bar, Style::new().fg(bar_color)),
            Span::styled(empty_bar, Style::new().fg(theme::BG_PANEL)),
            Span::styled(
                format!(" {:5.1}%  {:5.1}% cum", ratio * 100.0, cum * 100.0),
                theme::dim_style(),
            ),
            Span::styled(cum_marker.to_string(), Style::new().fg(theme::GREEN)),
        ]));
    }

    // Summary line at bottom
    lines.push(Line::from(""));
    lines.push(Line::from(vec![
        Span::styled("  90% at ", theme::dim_style()),
        Span::styled(
            format!("{} components", spectrum.components_90pct),
            theme::accent_style(),
        ),
        Span::styled("  ·  99% at ", theme::dim_style()),
        Span::styled(
            format!("{} components", spectrum.components_99pct),
            theme::accent_style(),
        ),
        Span::styled("  ·  Top-10: ", theme::dim_style()),
        Span::styled(
            format!("{:.1}%", spectrum.top10_concentration * 100.0),
            theme::accent_style(),
        ),
    ]));

    let paragraph = Paragraph::new(Text::from(lines));
    frame.render_widget(paragraph, inner);
}

// ── Helpers ─────────────────────────────────────────────────────────────────

fn gauge_line(
    label: &str,
    value: &str,
    ratio: f32,
    bar_width: usize,
    good_high: bool,
    hint: &str,
) -> (Line<'static>, Line<'static>) {
    let color = theme::quality_color(ratio, good_high);
    let filled = (ratio.clamp(0.0, 1.0) * bar_width as f32).round() as usize;
    let empty = bar_width.saturating_sub(filled);
    let bar = format!("{}{}", "█".repeat(filled), "░".repeat(empty));

    let label_line = Line::from(vec![
        Span::styled(format!("  {:<18}", label), theme::dim_style()),
        Span::styled(format!("{:<18}", value), Style::new().fg(color).bold()),
        Span::styled(hint.to_string(), Style::new().fg(theme::FG_DIM).italic()),
    ]);

    let bar_line = Line::from(vec![
        Span::raw("  "),
        Span::styled(bar, Style::new().fg(color)),
    ]);

    (label_line, bar_line)
}
