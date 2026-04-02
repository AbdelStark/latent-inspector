//! File browser popup overlay for selecting images.

use crate::tui::app::FileBrowser;
use crate::tui::theme;
use ratatui::prelude::*;
use ratatui::widgets::*;

/// Draw the file browser as a centered popup overlay.
pub fn draw_overlay(frame: &mut Frame, browser: &FileBrowser) {
    let area = frame.area();

    // Centered popup: 70% width, 80% height
    let popup_w = (area.width as f32 * 0.70).min(90.0) as u16;
    let popup_h = (area.height as f32 * 0.80).min(40.0) as u16;
    let popup_x = (area.width.saturating_sub(popup_w)) / 2;
    let popup_y = (area.height.saturating_sub(popup_h)) / 2;
    let popup_area = Rect::new(popup_x, popup_y, popup_w, popup_h);

    // Dim background
    let dim = Block::new().style(Style::new().bg(Color::Rgb(10, 10, 15)));
    frame.render_widget(dim, area);

    // Popup container
    let dir_display = browser.current_dir.to_string_lossy();
    let title = format!(" Open Image -- {} ", truncate_front(&dir_display, 50));
    let block = Block::bordered()
        .title(title)
        .title_style(theme::title_style())
        .border_type(BorderType::Rounded)
        .border_style(Style::new().fg(theme::BLUE))
        .style(Style::new().bg(theme::BG_DARK));
    let inner = block.inner(popup_area);
    frame.render_widget(block, popup_area);

    let chunks = Layout::vertical([
        Constraint::Fill(1),   // File list
        Constraint::Length(3), // Input + help
    ])
    .split(inner);

    draw_file_list(frame, chunks[0], browser);
    draw_browser_footer(frame, chunks[1], browser);
}

fn draw_file_list(frame: &mut Frame, area: Rect, browser: &FileBrowser) {
    let visible = area.height as usize;
    let scroll = if browser.selected >= visible {
        browser.selected - visible + 1
    } else {
        0
    };

    let mut lines: Vec<Line> = Vec::new();

    for (i, entry) in browser
        .entries
        .iter()
        .enumerate()
        .skip(scroll)
        .take(visible)
    {
        let is_selected = i == browser.selected;
        let icon = if entry.is_dir { "  ▸ " } else { "  ◆ " };

        let (name_style, icon_color) = if is_selected {
            (
                Style::new()
                    .fg(theme::FG_BRIGHT)
                    .bg(theme::BG_SELECTION)
                    .bold(),
                if entry.is_dir {
                    theme::BLUE
                } else {
                    theme::GREEN
                },
            )
        } else {
            (
                Style::new().fg(if entry.is_dir {
                    theme::FG
                } else {
                    theme::GREEN
                }),
                if entry.is_dir {
                    theme::FG_DIM
                } else {
                    theme::GREEN
                },
            )
        };

        let size_str = if entry.is_dir {
            String::new()
        } else {
            format_size(entry.size)
        };

        let available = area.width.saturating_sub(18) as usize;
        let name_display = truncate_str(&entry.name, available);

        let mut spans = vec![
            Span::styled(
                icon,
                Style::new().fg(icon_color).bg(if is_selected {
                    theme::BG_SELECTION
                } else {
                    theme::BG_DARK
                }),
            ),
            Span::styled(
                format!("{:<width$}", name_display, width = available),
                name_style,
            ),
        ];

        if !size_str.is_empty() {
            spans.push(Span::styled(
                format!("{:>10}", size_str),
                Style::new().fg(theme::FG_DIM).bg(if is_selected {
                    theme::BG_SELECTION
                } else {
                    theme::BG_DARK
                }),
            ));
        }

        lines.push(Line::from(spans));
    }

    if browser.entries.is_empty() {
        lines.push(Line::from(Span::styled(
            "    (no image files in this directory)",
            theme::dim_style(),
        )));
    }

    let paragraph = Paragraph::new(Text::from(lines));
    frame.render_widget(paragraph, area);
}

fn draw_browser_footer(frame: &mut Frame, area: Rect, browser: &FileBrowser) {
    let chunks = Layout::vertical([
        Constraint::Length(1), // Separator
        Constraint::Length(1), // Path input or info
        Constraint::Length(1), // Shortcuts
    ])
    .split(area);

    // Separator
    let sep = Paragraph::new(Line::from(Span::styled(
        format!("  {}", "─".repeat(area.width.saturating_sub(4) as usize)),
        Style::new().fg(theme::FG_DIM),
    )));
    frame.render_widget(sep, chunks[0]);

    // Path input or selected file info
    if browser.input_active {
        let input_line = Line::from(vec![
            Span::styled("  Path: ", theme::dim_style()),
            Span::styled(&browser.input_buffer, Style::new().fg(theme::FG_BRIGHT)),
            Span::styled("_", Style::new().fg(theme::BLUE)),
        ]);
        frame.render_widget(Paragraph::new(input_line), chunks[1]);
    } else if let Some(entry) = browser.selected_entry() {
        let info = if entry.is_dir {
            format!("  Directory: {}", entry.path.display())
        } else {
            format!(
                "  File: {} ({})",
                entry.path.display(),
                format_size(entry.size)
            )
        };
        let info_line = Line::from(Span::styled(
            truncate_str(&info, area.width.saturating_sub(2) as usize),
            theme::dim_style(),
        ));
        frame.render_widget(Paragraph::new(info_line), chunks[1]);
    }

    // Shortcuts
    let shortcuts = Line::from(vec![
        Span::styled("  ", Style::new()),
        Span::styled("j/k", theme::key_style()),
        Span::styled(" navigate  ", Style::new().fg(theme::FG_DIM)),
        Span::styled("enter", theme::key_style()),
        Span::styled(" open  ", Style::new().fg(theme::FG_DIM)),
        Span::styled("bksp", theme::key_style()),
        Span::styled(" up  ", Style::new().fg(theme::FG_DIM)),
        Span::styled("/", theme::key_style()),
        Span::styled(" path  ", Style::new().fg(theme::FG_DIM)),
        Span::styled("esc", theme::key_style()),
        Span::styled(" close", Style::new().fg(theme::FG_DIM)),
    ]);
    frame.render_widget(Paragraph::new(shortcuts), chunks[2]);
}

// ── Helpers ─────────────────────────────────────────────────────────────────

fn format_size(bytes: u64) -> String {
    if bytes >= 1_000_000 {
        format!("{:.1} MB", bytes as f64 / 1_000_000.0)
    } else if bytes >= 1_000 {
        format!("{:.0} KB", bytes as f64 / 1_000.0)
    } else {
        format!("{} B", bytes)
    }
}

fn truncate_str(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}…", &s[..max.saturating_sub(1)])
    }
}

fn truncate_front(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("…{}", &s[s.len() - max + 1..])
    }
}
