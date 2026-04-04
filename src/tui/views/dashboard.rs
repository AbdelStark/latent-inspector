//! Dashboard view — model registry overview and high-level metrics.

use crate::tui::app::App;
use crate::tui::theme::{self, mini_bar, truncate};
use ratatui::prelude::*;
use ratatui::widgets::*;

pub fn draw(frame: &mut Frame, area: Rect, app: &mut App) {
    let chunks = Layout::vertical([
        Constraint::Min(9),  // Model registry
        Constraint::Fill(1), // Bottom panels
    ])
    .split(area);

    draw_model_registry(frame, chunks[0], app);
    draw_bottom_panels(frame, chunks[1], app);
}

fn draw_model_registry(frame: &mut Frame, area: Rect, app: &mut App) {
    let header = Row::new(vec![
        Cell::from(""),
        Cell::from("Model").style(theme::heading_style()),
        Cell::from("Method").style(theme::heading_style()),
        Cell::from("Arch").style(theme::heading_style()),
        Cell::from("Dim").style(theme::heading_style()),
        Cell::from("Patches").style(theme::heading_style()),
        Cell::from("Params").style(theme::heading_style()),
        Cell::from("Rank").style(theme::heading_style()),
        Cell::from("Entropy").style(theme::heading_style()),
    ])
    .height(1)
    .bottom_margin(0);

    let rows: Vec<Row> = app
        .models
        .iter()
        .enumerate()
        .map(|(i, entry)| {
            let color = theme::model_color(i);
            let indicator = if i == app.selected_model { "▸" } else { " " };

            let metrics = app.metrics.iter().find(|m| m.model_name == entry.info.name);

            let rank_cell = match metrics {
                Some(m) => {
                    let ratio = m.effective_rank as f32 / m.embed_dim as f32;
                    let bar = mini_bar(ratio, 8);
                    format!("{} {}/{}", bar, m.effective_rank, m.embed_dim)
                }
                None => "—".into(),
            };

            let entropy_cell = match metrics {
                Some(m) => format!("{:.2}", m.patch_entropy),
                None => "—".into(),
            };

            Row::new(vec![
                Cell::from(indicator).style(Style::new().fg(color)),
                Cell::from(entry.info.name.as_str()).style(Style::new().fg(color).bold()),
                Cell::from(entry.info.method.to_string()).style(Style::new().fg(theme::FG)),
                Cell::from(entry.info.architecture.as_str()).style(Style::new().fg(theme::FG_DIM)),
                Cell::from(format!("{}", entry.info.embed_dim)).style(Style::new().fg(theme::FG)),
                Cell::from(format!("{}", entry.validation.tensor.patch_count))
                    .style(Style::new().fg(theme::FG)),
                Cell::from(format!("{}M", entry.info.params_m)).style(Style::new().fg(theme::FG)),
                Cell::from(rank_cell).style(Style::new().fg(theme::FG)),
                Cell::from(entropy_cell).style(Style::new().fg(theme::FG)),
            ])
        })
        .collect();

    let widths = [
        Constraint::Length(2),
        Constraint::Length(18),
        Constraint::Length(8),
        Constraint::Length(11),
        Constraint::Length(6),
        Constraint::Length(9),
        Constraint::Length(8),
        Constraint::Length(20),
        Constraint::Length(9),
    ];

    let table = Table::new(rows, widths)
        .header(header)
        .block(
            Block::bordered()
                .title(" Model Registry ")
                .title_style(theme::title_style())
                .border_type(BorderType::Rounded)
                .border_style(Style::new().fg(theme::FG_DIM)),
        )
        .row_highlight_style(theme::highlight_style())
        .highlight_symbol("▸ ");

    frame.render_stateful_widget(table, area, &mut app.model_table_state);
}

fn draw_bottom_panels(frame: &mut Frame, area: Rect, app: &App) {
    if app.image_thumbnail.is_some() {
        // Three-column layout: image | model detail | arch comparison
        let chunks = Layout::horizontal([
            Constraint::Percentage(35),
            Constraint::Percentage(35),
            Constraint::Percentage(30),
        ])
        .split(area);
        draw_image_preview(frame, chunks[0], app);
        draw_selected_model_detail(frame, chunks[1], app);
        draw_architecture_comparison(frame, chunks[2], app);
    } else {
        // Two-column layout: arch comparison | model detail
        let chunks = Layout::horizontal([Constraint::Percentage(50), Constraint::Percentage(50)])
            .split(area);
        draw_architecture_comparison(frame, chunks[0], app);
        draw_selected_model_detail(frame, chunks[1], app);
    }
}

fn draw_image_preview(frame: &mut Frame, area: Rect, app: &App) {
    let path_name = app
        .image_path
        .as_ref()
        .and_then(|p| p.file_name())
        .and_then(|n| n.to_str())
        .unwrap_or("image");

    let block = Block::bordered()
        .title(format!(" {} ", path_name))
        .title_style(theme::title_style())
        .border_type(BorderType::Rounded)
        .border_style(Style::new().fg(theme::CYAN));
    let inner = block.inner(area);
    frame.render_widget(block, area);

    if inner.width < 4 || inner.height < 3 {
        return;
    }

    if let Some(ref thumb) = app.image_thumbnail {
        // Reserve last 2 lines for info
        let img_height = inner.height.saturating_sub(2);
        let preview_lines = crate::tui::render_image_preview(thumb, inner.width, img_height);

        let mut all_lines = preview_lines;

        // Image info line
        let dims = format!("{}x{}", thumb.width(), thumb.height());
        all_lines.push(Line::from(""));
        all_lines.push(Line::from(vec![
            Span::styled(format!(" {} ", dims), Style::new().fg(theme::FG_DIM)),
            Span::styled("o", theme::key_style()),
            Span::styled(" change", Style::new().fg(theme::FG_DIM)),
        ]));

        let paragraph = Paragraph::new(Text::from(all_lines));
        frame.render_widget(paragraph, inner);
    }
}

fn draw_architecture_comparison(frame: &mut Frame, area: Rect, app: &App) {
    let block = Block::bordered()
        .title(" Architecture ")
        .title_style(theme::title_style())
        .border_type(BorderType::Rounded)
        .border_style(Style::new().fg(theme::FG_DIM));

    let inner = block.inner(area);
    frame.render_widget(block, area);

    if inner.height < 3 {
        return;
    }

    let max_dim = app
        .models
        .iter()
        .map(|e| e.info.embed_dim)
        .max()
        .unwrap_or(1) as f32;

    let bar_area_width = inner.width.saturating_sub(22) as f32;

    let mut lines: Vec<Line> = Vec::new();

    lines.push(Line::from(vec![Span::styled(
        "  Embed Dim",
        theme::heading_style(),
    )]));
    for (i, entry) in app.models.iter().enumerate() {
        let ratio = entry.info.embed_dim as f32 / max_dim;
        let bar_len = (ratio * bar_area_width) as usize;
        let bar = "█".repeat(bar_len);
        let color = theme::model_color(i);
        let name = truncate(&entry.info.name, 10);
        lines.push(Line::from(vec![
            Span::styled(format!("  {:<10} ", name), Style::new().fg(color)),
            Span::styled(bar, Style::new().fg(color)),
            Span::styled(format!(" {}", entry.info.embed_dim), theme::dim_style()),
        ]));
    }

    lines.push(Line::from(""));

    let max_params = app
        .models
        .iter()
        .map(|e| e.info.params_m)
        .max()
        .unwrap_or(1) as f32;
    lines.push(Line::from(vec![Span::styled(
        "  Params (M)",
        theme::heading_style(),
    )]));
    for (i, entry) in app.models.iter().enumerate() {
        let ratio = entry.info.params_m as f32 / max_params;
        let bar_len = (ratio * bar_area_width) as usize;
        let bar = "▓".repeat(bar_len);
        let color = theme::model_color(i);
        let name = truncate(&entry.info.name, 10);
        lines.push(Line::from(vec![
            Span::styled(format!("  {:<10} ", name), Style::new().fg(color)),
            Span::styled(bar, Style::new().fg(color)),
            Span::styled(format!(" {}M", entry.info.params_m), theme::dim_style()),
        ]));
    }

    let paragraph = Paragraph::new(Text::from(lines)).style(Style::new().bg(theme::BG_DARK));
    frame.render_widget(paragraph, inner);
}

fn draw_selected_model_detail(frame: &mut Frame, area: Rect, app: &App) {
    let block = Block::bordered()
        .title(" Selected Model ")
        .title_style(theme::title_style())
        .border_type(BorderType::Rounded)
        .border_style(Style::new().fg(theme::FG_DIM));
    let inner = block.inner(area);
    frame.render_widget(block, area);

    if app.models.is_empty() || inner.height < 3 {
        return;
    }

    let entry = &app.models[app.selected_model];
    let color = theme::model_color(app.selected_model);

    let mut lines = vec![
        Line::from(vec![
            Span::styled("  ◆ ", Style::new().fg(color)),
            Span::styled(&entry.info.name, Style::new().fg(color).bold()),
            Span::styled(
                format!("  ({})", entry.info.architecture),
                theme::dim_style(),
            ),
        ]),
        Line::from(""),
        detail_line("  Method", &entry.info.method.to_string()),
        detail_line(
            "  Patch Size",
            &format!("{}x{} px", entry.info.patch_size, entry.info.patch_size),
        ),
        detail_line(
            "  Input Size",
            &format!("{}x{} px", entry.info.input_size, entry.info.input_size),
        ),
        detail_line("  Layers", &entry.info.num_layers.to_string()),
        detail_line("  Heads", &entry.info.num_heads.to_string()),
    ];

    if let Some(m) = app.selected_metrics() {
        lines.push(Line::from(""));
        lines.push(Line::from(vec![Span::styled(
            "  -- Metrics --",
            theme::heading_style(),
        )]));
        lines.push(metric_line(
            "  Eff. Rank",
            &format!("{}/{}", m.effective_rank, m.embed_dim),
            m.effective_rank as f32 / m.embed_dim as f32,
            true,
        ));
        lines.push(metric_line(
            "  Dead Dims",
            &format!("{}", m.dead_dimensions),
            m.dead_dimensions as f32 / m.embed_dim as f32,
            false,
        ));
        lines.push(metric_line(
            "  Entropy",
            &format!("{:.2}", m.patch_entropy),
            (m.patch_entropy / 3.0).min(1.0),
            true,
        ));
        lines.push(metric_line(
            "  Isotropy",
            &format!("{:.3}", m.patch_isotropy),
            m.patch_isotropy,
            true,
        ));
        lines.push(metric_line(
            "  Sp. coherence",
            &m.spatial_coherence
                .map(|v| format!("{:.3}", v))
                .unwrap_or_else(|| "N/A".into()),
            m.spatial_coherence.unwrap_or(0.0),
            true,
        ));
    }

    let paragraph = Paragraph::new(Text::from(lines)).style(Style::new().bg(theme::BG_DARK));
    frame.render_widget(paragraph, inner);
}

// ── Helpers ─────────────────────────────────────────────────────────────────

fn detail_line(label: &str, value: &str) -> Line<'static> {
    Line::from(vec![
        Span::styled(format!("{:<16}", label), theme::dim_style()),
        Span::styled(value.to_string(), theme::value_style()),
    ])
}

fn metric_line(label: &str, value: &str, ratio: f32, good_high: bool) -> Line<'static> {
    let color = theme::quality_color(ratio, good_high);
    let bar = mini_bar(ratio, 8);
    Line::from(vec![
        Span::styled(format!("{:<16}", label), theme::dim_style()),
        Span::styled(format!("{} ", bar), Style::new().fg(color)),
        Span::styled(value.to_string(), Style::new().fg(color)),
    ])
}
