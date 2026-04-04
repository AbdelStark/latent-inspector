//! Compare view — side-by-side model comparison with similarity matrices.

use crate::tui::app::App;
use crate::tui::theme::{self, mini_bar, truncate};
use ratatui::prelude::*;
use ratatui::widgets::*;

pub fn draw(frame: &mut Frame, area: Rect, app: &App) {
    let outer = Block::bordered()
        .title(" Cross-Model Comparison ")
        .title_style(theme::title_style())
        .border_type(BorderType::Rounded)
        .border_style(Style::new().fg(theme::FG_DIM));
    let inner = outer.inner(area);
    frame.render_widget(outer, area);

    if app.metrics.is_empty() {
        let msg = Paragraph::new(vec![
            Line::from(""),
            Line::from(Span::styled(
                "  No analysis data available for comparison.",
                theme::dim_style(),
            )),
        ]);
        frame.render_widget(msg, inner);
        return;
    }

    let chunks = Layout::vertical([
        Constraint::Min(9),  // Metrics table
        Constraint::Fill(1), // Bottom panels
    ])
    .split(inner);

    draw_metrics_table(frame, chunks[0], app);
    draw_similarity_panels(frame, chunks[1], app);
}

fn draw_metrics_table(frame: &mut Frame, area: Rect, app: &App) {
    let block = Block::bordered()
        .title(" Per-Model Metrics ")
        .title_style(theme::heading_style())
        .border_type(BorderType::Rounded)
        .border_style(Style::new().fg(theme::FG_DIM));
    let table_inner = block.inner(area);
    frame.render_widget(block, area);

    if table_inner.width < 20 {
        return;
    }

    let metric_labels = [
        "Repr. rank",
        "Dead dimensions",
        "Patch entropy",
        "CLS L2 norm",
        "Patch norm μ±σ",
        "Top-10 var%",
        "Components@90%",
        "Patch isotropy",
        "Patch uniformity",
        "Spatial coherence",
    ];

    let model_names: Vec<&str> = app.metrics.iter().map(|m| m.model_name.as_str()).collect();

    // Build header
    let mut header_cells = vec![Cell::from("Metric").style(theme::heading_style())];
    for (i, name) in model_names.iter().enumerate() {
        header_cells.push(
            Cell::from(truncate(name, 16)).style(Style::new().fg(theme::model_color(i)).bold()),
        );
    }
    let header = Row::new(header_cells).height(1);

    // Build rows
    let mut rows: Vec<Row> = Vec::new();
    for (row_idx, &label) in metric_labels.iter().enumerate() {
        let mut cells = vec![Cell::from(label).style(theme::dim_style())];
        for (i, m) in app.metrics.iter().enumerate() {
            let value = match row_idx {
                0 => format!("{}/{}", m.effective_rank, m.embed_dim),
                1 => format!("{}", m.dead_dimensions),
                2 => format!("{:.2}", m.patch_entropy),
                3 => m
                    .cls_l2_norm
                    .map(|v| format!("{:.1}", v))
                    .unwrap_or_else(|| "N/A".into()),
                4 => format!("{:.2} ± {:.2}", m.patch_norm_mean, m.patch_norm_std),
                5 => format!("{:.1}%", m.top10_variance_pct),
                6 => format!("{}", m.components_90pct),
                7 => format!("{:.3}", m.patch_isotropy),
                8 => format!("{:.2}", m.patch_uniformity),
                9 => m
                    .spatial_coherence
                    .map(|v| format!("{:.3}", v))
                    .unwrap_or_else(|| "N/A".into()),
                _ => "—".into(),
            };
            cells.push(Cell::from(value).style(Style::new().fg(theme::model_color(i))));
        }
        rows.push(Row::new(cells));
    }

    let col_width = 18;
    let mut widths = vec![Constraint::Length(18)];
    for _ in &model_names {
        widths.push(Constraint::Length(col_width));
    }

    let table = Table::new(rows, widths).header(header);
    frame.render_widget(table, table_inner);
}

fn draw_similarity_panels(frame: &mut Frame, area: Rect, app: &App) {
    let chunks =
        Layout::horizontal([Constraint::Percentage(50), Constraint::Percentage(50)]).split(area);

    draw_cls_matrix(frame, chunks[0], app);
    draw_cross_metrics(frame, chunks[1], app);
}

fn draw_cls_matrix(frame: &mut Frame, area: Rect, app: &App) {
    let block = Block::bordered()
        .title(" CLS Cosine Similarity ")
        .title_style(theme::heading_style())
        .border_type(BorderType::Rounded)
        .border_style(Style::new().fg(theme::FG_DIM));
    let inner = block.inner(area);
    frame.render_widget(block, area);

    let names: Vec<&str> = app.metrics.iter().map(|m| m.model_name.as_str()).collect();

    let mut lines: Vec<Line> = Vec::new();

    // Header row
    let mut header_spans = vec![Span::styled(format!("{:<14}", ""), theme::dim_style())];
    for (i, name) in names.iter().enumerate() {
        header_spans.push(Span::styled(
            format!("{:>8}", truncate(name, 7)),
            Style::new().fg(theme::model_color(i)).bold(),
        ));
    }
    lines.push(Line::from(header_spans));

    // Matrix rows
    for (i, name_a) in names.iter().enumerate() {
        let mut row_spans = vec![Span::styled(
            format!("  {:<12}", truncate(name_a, 11)),
            Style::new().fg(theme::model_color(i)),
        )];

        for (j, name_b) in names.iter().enumerate() {
            if i == j {
                row_spans.push(Span::styled(
                    format!("{:>8}", "1.000"),
                    Style::new().fg(theme::FG_DIM),
                ));
            } else {
                let val = find_cls_sim(&app.comparisons, name_a, name_b);
                match val {
                    Some(v) => {
                        let color = sim_color(v);
                        let block_char = sim_block(v);
                        row_spans.push(Span::styled(
                            format!("{} {:>5.3}", block_char, v),
                            Style::new().fg(color),
                        ));
                    }
                    None => {
                        row_spans.push(Span::styled(
                            format!("{:>8}", "—"),
                            Style::new().fg(theme::FG_DIM),
                        ));
                    }
                }
            }
        }

        lines.push(Line::from(row_spans));
    }

    lines.push(Line::from(""));
    lines.push(Line::from(vec![
        Span::styled("  Legend: ", theme::dim_style()),
        Span::styled("░", Style::new().fg(theme::FG_DIM)),
        Span::styled(" <0.3  ", theme::dim_style()),
        Span::styled("▒", Style::new().fg(theme::YELLOW)),
        Span::styled(" 0.3–0.6  ", theme::dim_style()),
        Span::styled("▓", Style::new().fg(theme::ORANGE)),
        Span::styled(" 0.6–0.8  ", theme::dim_style()),
        Span::styled("█", Style::new().fg(theme::GREEN)),
        Span::styled(" >0.8", theme::dim_style()),
    ]));

    let paragraph = Paragraph::new(Text::from(lines));
    frame.render_widget(paragraph, inner);
}

fn draw_cross_metrics(frame: &mut Frame, area: Rect, app: &App) {
    let block = Block::bordered()
        .title(" Cross-Model Metrics ")
        .title_style(theme::heading_style())
        .border_type(BorderType::Rounded)
        .border_style(Style::new().fg(theme::FG_DIM));
    let inner = block.inner(area);
    frame.render_widget(block, area);

    let mut lines: Vec<Line> = Vec::new();

    // CKA section
    lines.push(Line::from(vec![Span::styled(
        "  Linear CKA",
        theme::heading_style(),
    )]));
    for cmp in &app.comparisons {
        let bar = mini_bar(cmp.linear_cka, 12);
        let color = sim_color(cmp.linear_cka);
        lines.push(Line::from(vec![
            Span::styled(
                format!(
                    "  {} × {} ",
                    truncate(&cmp.model_a, 7),
                    truncate(&cmp.model_b, 7)
                ),
                theme::dim_style(),
            ),
            Span::styled(bar, Style::new().fg(color)),
            Span::styled(format!(" {:.3}", cmp.linear_cka), Style::new().fg(color)),
        ]));
    }

    lines.push(Line::from(""));

    // k-NN overlap section
    lines.push(Line::from(vec![Span::styled(
        "  k-NN Overlap (k=10)",
        theme::heading_style(),
    )]));
    for cmp in &app.comparisons {
        let bar = mini_bar(cmp.knn_overlap_k10, 12);
        let color = sim_color(cmp.knn_overlap_k10);
        lines.push(Line::from(vec![
            Span::styled(
                format!(
                    "  {} × {} ",
                    truncate(&cmp.model_a, 7),
                    truncate(&cmp.model_b, 7)
                ),
                theme::dim_style(),
            ),
            Span::styled(bar, Style::new().fg(color)),
            Span::styled(
                format!(" {:.0}%", cmp.knn_overlap_k10 * 100.0),
                Style::new().fg(color),
            ),
        ]));
    }

    let paragraph = Paragraph::new(Text::from(lines)).scroll((app.compare_scroll, 0));
    frame.render_widget(paragraph, inner);
}

// ── Helpers ─────────────────────────────────────────────────────────────────

fn find_cls_sim(
    comparisons: &[crate::analysis::ComparisonMetrics],
    a: &str,
    b: &str,
) -> Option<f32> {
    comparisons
        .iter()
        .find(|c| (c.model_a == a && c.model_b == b) || (c.model_a == b && c.model_b == a))
        .and_then(|c| c.cls_cosine_sim)
}

fn sim_color(v: f32) -> Color {
    if v > 0.8 {
        theme::GREEN
    } else if v > 0.6 {
        theme::ORANGE
    } else if v > 0.3 {
        theme::YELLOW
    } else {
        theme::FG_DIM
    }
}

fn sim_block(v: f32) -> &'static str {
    if v > 0.8 {
        "█"
    } else if v > 0.6 {
        "▓"
    } else if v > 0.3 {
        "▒"
    } else {
        "░"
    }
}
