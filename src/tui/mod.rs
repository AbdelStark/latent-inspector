//! Interactive terminal UI for the latent-inspector.
//!
//! Uses ratatui + crossterm for a rich, educational visualisation of
//! SSL model representations and analysis metrics.

pub mod app;
pub mod theme;
pub mod views;

use app::{App, Tab};
use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use ratatui::prelude::*;
use ratatui::widgets::*;
use std::io;
use std::time::Duration;

/// Launch the TUI event loop.
pub fn run(mut app: App) -> io::Result<()> {
    crossterm::terminal::enable_raw_mode()?;
    let mut stdout = io::stdout();
    crossterm::execute!(stdout, crossterm::terminal::EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;
    terminal.clear()?;

    let result = event_loop(&mut terminal, &mut app);

    crossterm::terminal::disable_raw_mode()?;
    crossterm::execute!(
        terminal.backend_mut(),
        crossterm::terminal::LeaveAlternateScreen
    )?;
    terminal.show_cursor()?;

    result
}

fn event_loop(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    app: &mut App,
) -> io::Result<()> {
    while app.running {
        terminal.draw(|frame| draw(frame, app))?;

        if event::poll(Duration::from_millis(50))? {
            if let Event::Key(key) = event::read()? {
                if key.kind != KeyEventKind::Press {
                    continue;
                }
                handle_key(app, key.code, key.modifiers);
            }
        }
    }
    Ok(())
}

fn handle_key(app: &mut App, code: KeyCode, modifiers: KeyModifiers) {
    // File browser takes priority when active
    if app.file_browser.active {
        handle_file_browser_key(app, code);
        return;
    }

    // Global keys
    match code {
        KeyCode::Char('q') | KeyCode::Esc => {
            app.running = false;
            return;
        }
        KeyCode::Char('c') if modifiers.contains(KeyModifiers::CONTROL) => {
            app.running = false;
            return;
        }
        KeyCode::Tab => {
            app.tab = app.tab.next();
            return;
        }
        KeyCode::BackTab => {
            app.tab = app.tab.prev();
            return;
        }
        KeyCode::Right => {
            app.tab = app.tab.next();
            return;
        }
        KeyCode::Left => {
            app.tab = app.tab.prev();
            return;
        }
        KeyCode::Char('1') => {
            app.tab = Tab::Dashboard;
            return;
        }
        KeyCode::Char('2') => {
            app.tab = Tab::Inspector;
            return;
        }
        KeyCode::Char('3') => {
            app.tab = Tab::Compare;
            return;
        }
        KeyCode::Char('4') => {
            app.tab = Tab::Spectrum;
            return;
        }
        KeyCode::Char('5') => {
            app.tab = Tab::Help;
            return;
        }
        KeyCode::Char('?') => {
            app.tab = Tab::Help;
            return;
        }
        KeyCode::Char('o') => {
            // Open file browser
            app.file_browser.active = true;
            app.file_browser.refresh();
            return;
        }
        _ => {}
    }

    // Per-tab keys
    match app.tab {
        Tab::Dashboard => match code {
            KeyCode::Down | KeyCode::Char('j') => app.select_next_model(),
            KeyCode::Up | KeyCode::Char('k') => app.select_prev_model(),
            KeyCode::Enter => app.tab = Tab::Inspector,
            _ => {}
        },
        Tab::Inspector => match code {
            KeyCode::Down | KeyCode::Char('j') => {
                app.inspector_scroll = app.inspector_scroll.saturating_add(1);
            }
            KeyCode::Up | KeyCode::Char('k') => {
                app.inspector_scroll = app.inspector_scroll.saturating_sub(1);
            }
            KeyCode::Char('m') => app.select_next_model(),
            KeyCode::Char('g') => app.inspector_scroll = 0,
            _ => {}
        },
        Tab::Compare => match code {
            KeyCode::Down | KeyCode::Char('j') => {
                app.compare_scroll = app.compare_scroll.saturating_add(1);
            }
            KeyCode::Up | KeyCode::Char('k') => {
                app.compare_scroll = app.compare_scroll.saturating_sub(1);
            }
            _ => {}
        },
        Tab::Spectrum => match code {
            KeyCode::Down | KeyCode::Char('j') => {
                app.spectrum_scroll = app.spectrum_scroll.saturating_add(1);
            }
            KeyCode::Up | KeyCode::Char('k') => {
                app.spectrum_scroll = app.spectrum_scroll.saturating_sub(1);
            }
            KeyCode::Char('m') => app.select_next_model(),
            KeyCode::Char('g') => app.spectrum_scroll = 0,
            _ => {}
        },
        Tab::Help => match code {
            KeyCode::Down | KeyCode::Char('j') => {
                app.help_scroll = app.help_scroll.saturating_add(1);
            }
            KeyCode::Up | KeyCode::Char('k') => {
                app.help_scroll = app.help_scroll.saturating_sub(1);
            }
            KeyCode::Char('g') => app.help_scroll = 0,
            KeyCode::Char('G') => app.help_scroll = 200,
            _ => {}
        },
    }
}

fn handle_file_browser_key(app: &mut App, code: KeyCode) {
    // Text input mode
    if app.file_browser.input_active {
        match code {
            KeyCode::Char(c) => app.file_browser.input_buffer.push(c),
            KeyCode::Backspace => {
                app.file_browser.input_buffer.pop();
            }
            KeyCode::Enter => {
                let path = std::path::PathBuf::from(&app.file_browser.input_buffer);
                if path.is_dir() {
                    app.file_browser.navigate_to(path);
                } else if path.is_file() {
                    app.load_image(&path);
                    app.file_browser.active = false;
                }
                app.file_browser.input_active = false;
            }
            KeyCode::Esc => {
                app.file_browser.input_active = false;
            }
            _ => {}
        }
        return;
    }

    // Navigation mode
    match code {
        KeyCode::Char('j') | KeyCode::Down => app.file_browser.select_down(),
        KeyCode::Char('k') | KeyCode::Up => app.file_browser.select_up(),
        KeyCode::Enter => {
            if let Some(entry) = app.file_browser.selected_entry() {
                let path = entry.path.clone();
                let is_dir = entry.is_dir;
                let is_image = entry.is_image;
                if is_dir {
                    app.file_browser.navigate_to(path);
                } else if is_image {
                    app.load_image(&path);
                    app.file_browser.active = false;
                }
            }
        }
        KeyCode::Backspace | KeyCode::Char('h') => {
            app.file_browser.go_up();
        }
        KeyCode::Esc | KeyCode::Char('q') => {
            app.file_browser.active = false;
        }
        KeyCode::Char('/') => {
            app.file_browser.toggle_input();
        }
        _ => {}
    }
}

// ── Rendering ───────────────────────────────────────────────────────────────

fn draw(frame: &mut Frame, app: &mut App) {
    let area = frame.area();

    // Background
    frame.render_widget(Block::new().style(Style::new().bg(theme::BG_DARK)), area);

    let layout = Layout::vertical([
        Constraint::Length(3), // Header + tabs
        Constraint::Fill(1),   // Content
        Constraint::Length(1), // Status bar
    ])
    .split(area);

    draw_header(frame, layout[0], app);

    match app.tab {
        Tab::Dashboard => views::dashboard::draw(frame, layout[1], app),
        Tab::Inspector => views::inspector::draw(frame, layout[1], app),
        Tab::Compare => views::compare::draw(frame, layout[1], app),
        Tab::Spectrum => views::spectrum::draw(frame, layout[1], app),
        Tab::Help => views::help::draw(frame, layout[1], app),
    }

    draw_status_bar(frame, layout[2], app);

    // File browser overlay (on top of everything)
    if app.file_browser.active {
        views::filebrowser::draw_overlay(frame, &app.file_browser);
    }
}

fn draw_header(frame: &mut Frame, area: Rect, app: &App) {
    let block = Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(Style::new().fg(theme::FG_DIM))
        .style(Style::new().bg(theme::BG_DARK));
    let inner = block.inner(area);
    frame.render_widget(block, area);

    let chunks = Layout::horizontal([
        Constraint::Length(22), // Title
        Constraint::Fill(1),    // Tabs
        Constraint::Length(8),  // Version
    ])
    .split(inner);

    // Title
    let title = Paragraph::new(Line::from(vec![
        Span::styled(" ◈ ", Style::new().fg(theme::BLUE)),
        Span::styled("LATENT", Style::new().fg(theme::FG_BRIGHT).bold()),
        Span::styled(" INSPECTOR", Style::new().fg(theme::BLUE).bold()),
    ]));
    frame.render_widget(title, chunks[0]);

    // Tabs
    let tab_titles: Vec<Line> = Tab::ALL
        .iter()
        .map(|tab| {
            let is_active = app.tab == *tab;
            if is_active {
                Line::from(vec![
                    Span::styled("◉ ", Style::new().fg(theme::BLUE)),
                    Span::styled(tab.label(), Style::new().fg(theme::FG_BRIGHT).bold()),
                ])
            } else {
                Line::from(vec![
                    Span::styled("○ ", Style::new().fg(theme::FG_DIM)),
                    Span::styled(tab.label(), Style::new().fg(theme::FG_DIM)),
                ])
            }
        })
        .collect();

    let tabs = Tabs::new(tab_titles)
        .divider(Span::styled(" │ ", Style::new().fg(theme::FG_DIM)))
        .select(app.tab.index())
        .padding(" ", " ");
    frame.render_widget(tabs, chunks[1]);

    // Version
    let version = Paragraph::new(
        Line::from(Span::styled("v0.1.0", theme::dim_style())).alignment(Alignment::Right),
    );
    frame.render_widget(version, chunks[2]);
}

fn draw_status_bar(frame: &mut Frame, area: Rect, app: &App) {
    let tab_keys = match app.tab {
        Tab::Dashboard => "↑↓ select  enter inspect  ",
        Tab::Inspector => "↑↓ scroll  m model  ",
        Tab::Compare => "↑↓ scroll  ",
        Tab::Spectrum => "↑↓ scroll  m model  g top  ",
        Tab::Help => "↑↓ scroll  g top  G bottom  ",
    };

    let mode_indicator = if app.demo_mode { " [DEMO]" } else { "" };

    let image_indicator = app
        .image_path
        .as_ref()
        .and_then(|p| p.file_name())
        .and_then(|n| n.to_str())
        .map(|n| format!("  [{n}]"))
        .unwrap_or_default();

    let line = Line::from(vec![
        Span::styled(" ", Style::new().bg(theme::BG_PANEL)),
        Span::styled("q", theme::key_style()),
        Span::styled(
            " quit  ",
            Style::new().fg(theme::FG_DIM).bg(theme::BG_PANEL),
        ),
        Span::styled("←→", theme::key_style()),
        Span::styled(" tab  ", Style::new().fg(theme::FG_DIM).bg(theme::BG_PANEL)),
        Span::styled("o", theme::key_style()),
        Span::styled(
            " open  ",
            Style::new().fg(theme::FG_DIM).bg(theme::BG_PANEL),
        ),
        Span::styled(tab_keys, Style::new().fg(theme::FG_DIM).bg(theme::BG_PANEL)),
        Span::styled("?", theme::key_style()),
        Span::styled(" help", Style::new().fg(theme::FG_DIM).bg(theme::BG_PANEL)),
        Span::styled(
            image_indicator,
            Style::new().fg(theme::CYAN).bg(theme::BG_PANEL),
        ),
        Span::styled(
            mode_indicator.to_string(),
            Style::new().fg(theme::YELLOW).bg(theme::BG_PANEL).bold(),
        ),
    ]);

    let paragraph = Paragraph::new(line).style(Style::new().bg(theme::BG_PANEL));
    frame.render_widget(paragraph, area);
}

// ── Image preview rendering ─────────────────────────────────────────────────

/// Render an image thumbnail to styled half-block lines for terminal display.
/// Each terminal cell shows 2 vertical pixels using the `▀` character with
/// fg = top pixel, bg = bottom pixel.
pub fn render_image_preview(img: &image::RgbImage, width: u16, height: u16) -> Vec<Line<'static>> {
    let pixel_h = height as u32 * 2;
    let target_w = width as u32;

    // Calculate aspect-preserving size
    let img_w = img.width() as f32;
    let img_h = img.height() as f32;
    let scale = (target_w as f32 / img_w).min(pixel_h as f32 / img_h);
    let render_w = ((img_w * scale) as u32).max(1);
    let render_h = ((img_h * scale) as u32).max(1);

    let resized = image::imageops::resize(
        img,
        render_w,
        render_h,
        image::imageops::FilterType::Triangle,
    );

    let x_pad = target_w.saturating_sub(render_w) / 2;
    let y_pad = pixel_h.saturating_sub(render_h) / 2;

    let bg = Color::Rgb(26, 27, 38); // theme::BG_DARK

    let mut lines = Vec::new();
    for y_cell in 0..height as u32 {
        let mut spans = Vec::new();
        for x in 0..target_w {
            let img_x = x.wrapping_sub(x_pad);
            let top_y = (y_cell * 2).wrapping_sub(y_pad);
            let bot_y = (y_cell * 2 + 1).wrapping_sub(y_pad);

            let top_color = if img_x < render_w && top_y < render_h {
                let p = resized.get_pixel(img_x, top_y);
                Color::Rgb(p[0], p[1], p[2])
            } else {
                bg
            };

            let bot_color = if img_x < render_w && bot_y < render_h {
                let p = resized.get_pixel(img_x, bot_y);
                Color::Rgb(p[0], p[1], p[2])
            } else {
                bg
            };

            spans.push(Span::styled("▀", Style::new().fg(top_color).bg(bot_color)));
        }
        lines.push(Line::from(spans));
    }
    lines
}
