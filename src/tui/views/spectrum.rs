//! Spectrum view — full PCA variance spectrum visualization.

use crate::tui::app::App;
use crate::tui::theme::{self, dim_color};
use ratatui::prelude::*;
use ratatui::widgets::*;

pub fn draw(frame: &mut Frame, area: Rect, app: &App) {
    let model_name = if app.selected_model < app.models.len() {
        app.models[app.selected_model].info.name.clone()
    } else {
        "—".to_string()
    };

    let outer = Block::bordered()
        .title(format!(" PCA Variance Spectrum — {} ", model_name))
        .title_style(theme::title_style())
        .border_type(BorderType::Rounded)
        .border_style(Style::new().fg(theme::FG_DIM));
    let inner = outer.inner(area);
    frame.render_widget(outer, area);

    let spectrum = match app.selected_spectrum() {
        Some(s) => s,
        None => {
            let msg = Paragraph::new(vec![
                Line::from(""),
                Line::from(Span::styled(
                    "  No variance spectrum data available.",
                    theme::dim_style(),
                )),
            ]);
            frame.render_widget(msg, inner);
            return;
        }
    };

    let chunks = Layout::vertical([
        Constraint::Length(3), // Model info header
        Constraint::Fill(1),   // Bar chart
        Constraint::Length(7), // Summary + interpretation
    ])
    .split(inner);

    draw_spectrum_header(frame, chunks[0], app, spectrum);
    draw_full_spectrum(frame, chunks[1], app, spectrum);
    draw_spectrum_summary(frame, chunks[2], spectrum);
}

fn draw_spectrum_header(
    frame: &mut Frame,
    area: Rect,
    app: &App,
    spectrum: &crate::analysis::VarianceSpectrum,
) {
    let color = if app.selected_model < app.models.len() {
        theme::model_color(app.selected_model)
    } else {
        theme::BLUE
    };

    let entry = &app.models[app.selected_model.min(app.models.len().saturating_sub(1))];
    let total_components = spectrum.ratios.len();

    let lines = vec![
        Line::from(vec![
            Span::styled("  ◆ ", Style::new().fg(color)),
            Span::styled(
                format!(
                    "{} · {} · {}d",
                    entry.info.name, entry.info.method, entry.info.embed_dim
                ),
                Style::new().fg(color).bold(),
            ),
            Span::styled(
                format!("  ·  Showing {} components", total_components),
                theme::dim_style(),
            ),
        ]),
        Line::from(vec![Span::styled(
            "  Scree plot: explained variance ratio per principal component",
            theme::dim_style(),
        )]),
    ];
    let paragraph = Paragraph::new(Text::from(lines));
    frame.render_widget(paragraph, area);
}

fn draw_full_spectrum(
    frame: &mut Frame,
    area: Rect,
    app: &App,
    spectrum: &crate::analysis::VarianceSpectrum,
) {
    let color = if app.selected_model < app.models.len() {
        theme::model_color(app.selected_model)
    } else {
        theme::BLUE
    };

    let bar_area_width = area.width.saturating_sub(34) as f32;
    let max_ratio = spectrum
        .ratios
        .iter()
        .cloned()
        .fold(0.0_f32, f32::max)
        .max(0.001);

    let mut lines: Vec<Line> = Vec::new();
    let visible_count = (area.height as usize).saturating_sub(1);
    let offset = app.spectrum_scroll as usize;

    for (i, (&ratio, &cum)) in spectrum
        .ratios
        .iter()
        .zip(spectrum.cumulative.iter())
        .enumerate()
        .skip(offset)
        .take(visible_count)
    {
        let bar_len = ((ratio / max_ratio) * bar_area_width) as usize;
        let bar = "█".repeat(bar_len);
        let empty = (bar_area_width as usize).saturating_sub(bar_len);
        let empty_bar = "░".repeat(empty);

        let intensity = 1.0 - (i as f32 / spectrum.ratios.len() as f32) * 0.6;
        let bar_color = dim_color(color, intensity);

        // Cumulative progress indicator
        let cum_bar_width: usize = 10;
        let cum_filled = (cum * cum_bar_width as f32).round() as usize;
        let cum_empty = cum_bar_width.saturating_sub(cum_filled);
        let cum_bar = format!("{}{}", "▓".repeat(cum_filled), "·".repeat(cum_empty));

        let threshold =
            if cum >= 0.90 && (i == 0 || spectrum.cumulative[i.saturating_sub(1)] < 0.90) {
                " ◀ 90%"
            } else if cum >= 0.99 && (i == 0 || spectrum.cumulative[i.saturating_sub(1)] < 0.99) {
                " ◀ 99%"
            } else {
                ""
            };

        let threshold_color = if threshold.contains("90") {
            theme::YELLOW
        } else {
            theme::GREEN
        };

        lines.push(Line::from(vec![
            Span::styled(format!("  PC{:02} ", i + 1), theme::dim_style()),
            Span::styled(bar, Style::new().fg(bar_color)),
            Span::styled(empty_bar, Style::new().fg(theme::BG_PANEL)),
            Span::styled(
                format!(" {:5.1}% ", ratio * 100.0),
                Style::new().fg(bar_color),
            ),
            Span::styled(cum_bar, Style::new().fg(theme::CYAN)),
            Span::styled(
                threshold.to_string(),
                Style::new().fg(threshold_color).bold(),
            ),
        ]));
    }

    let paragraph = Paragraph::new(Text::from(lines));
    frame.render_widget(paragraph, area);
}

fn draw_spectrum_summary(
    frame: &mut Frame,
    area: Rect,
    spectrum: &crate::analysis::VarianceSpectrum,
) {
    let block = Block::bordered()
        .title(" Interpretation ")
        .title_style(theme::heading_style())
        .border_type(BorderType::Rounded)
        .border_style(Style::new().fg(theme::FG_DIM));
    let inner = block.inner(area);
    frame.render_widget(block, area);

    let concentration = spectrum.top10_concentration;
    let (assessment, assessment_color) = if concentration > 0.80 {
        (
            "Highly concentrated — the representation is dominated by a few dimensions. \
             This may indicate dimensional collapse or a narrow feature space.",
            theme::RED,
        )
    } else if concentration > 0.60 {
        (
            "Moderately concentrated — a reasonable balance between informative top \
             components and distributed lower components. Typical for well-trained SSL models.",
            theme::YELLOW,
        )
    } else {
        (
            "Well-distributed — variance is spread across many dimensions, indicating \
             a rich, high-rank representation space. Common in JEPA-family models.",
            theme::GREEN,
        )
    };

    let lines = vec![
        Line::from(vec![
            Span::styled("  Components for 90%: ", theme::dim_style()),
            Span::styled(
                format!("{}", spectrum.components_90pct),
                theme::accent_style(),
            ),
            Span::styled("  ·  Components for 99%: ", theme::dim_style()),
            Span::styled(
                format!("{}", spectrum.components_99pct),
                theme::accent_style(),
            ),
            Span::styled("  ·  Top-10: ", theme::dim_style()),
            Span::styled(
                format!("{:.1}%", concentration * 100.0),
                theme::accent_style(),
            ),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled("  ◆ ", Style::new().fg(assessment_color)),
            Span::styled(assessment.to_string(), Style::new().fg(assessment_color)),
        ]),
    ];

    let paragraph = Paragraph::new(Text::from(lines));
    frame.render_widget(paragraph, inner);
}
