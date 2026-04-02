//! Color palette and style helpers for the TUI.
//!
//! Inspired by Tokyo Night with ML-research aesthetics.

use ratatui::style::{Color, Modifier, Style};

// ── Base palette ────────────────────────────────────────────────────────────

pub const BG_DARK: Color = Color::Rgb(26, 27, 38);
pub const BG_PANEL: Color = Color::Rgb(36, 40, 59);
pub const BG_HIGHLIGHT: Color = Color::Rgb(41, 46, 66);
pub const BG_SELECTION: Color = Color::Rgb(52, 59, 88);

pub const FG: Color = Color::Rgb(169, 177, 214);
pub const FG_DIM: Color = Color::Rgb(86, 95, 137);
pub const FG_BRIGHT: Color = Color::Rgb(199, 208, 245);

// ── Accents ─────────────────────────────────────────────────────────────────

pub const BLUE: Color = Color::Rgb(122, 162, 247);
pub const PURPLE: Color = Color::Rgb(187, 154, 247);
pub const CYAN: Color = Color::Rgb(125, 207, 255);
pub const GREEN: Color = Color::Rgb(158, 206, 106);
pub const YELLOW: Color = Color::Rgb(224, 175, 104);
pub const ORANGE: Color = Color::Rgb(255, 158, 100);
pub const RED: Color = Color::Rgb(247, 118, 142);
pub const PINK: Color = Color::Rgb(255, 117, 165);
pub const TEAL: Color = Color::Rgb(115, 218, 202);

// ── Per-model accent colors ─────────────────────────────────────────────────

pub const MODEL_COLORS: &[Color] = &[
    Color::Rgb(122, 162, 247), // DINOv2 — blue
    Color::Rgb(158, 206, 106), // MAE    — green
    Color::Rgb(255, 158, 100), // CLIP   — orange
    Color::Rgb(187, 154, 247), // I-JEPA — purple
    Color::Rgb(125, 207, 255), // SigLIP — cyan
];

pub fn model_color(index: usize) -> Color {
    MODEL_COLORS[index % MODEL_COLORS.len()]
}

// ── Quality gradient (metric health) ────────────────────────────────────────

/// Returns a color representing quality.  When `good_high` is true a ratio
/// near 1.0 is green; when false a ratio near 0.0 is green.
pub fn quality_color(ratio: f32, good_high: bool) -> Color {
    let r = if good_high { ratio } else { 1.0 - ratio };
    let r = r.clamp(0.0, 1.0);
    if r > 0.7 {
        GREEN
    } else if r > 0.4 {
        YELLOW
    } else {
        RED
    }
}

/// Returns a heat-map color for values in `[0, 1]`.
pub fn heat_color(v: f32) -> Color {
    let v = v.clamp(0.0, 1.0);
    if v < 0.25 {
        Color::Rgb(36, 40, 59) // very cool
    } else if v < 0.5 {
        Color::Rgb(86, 95, 137) // cool
    } else if v < 0.75 {
        YELLOW
    } else {
        ORANGE
    }
}

// ── Composite styles ────────────────────────────────────────────────────────

pub fn title_style() -> Style {
    Style::new().fg(FG_BRIGHT).add_modifier(Modifier::BOLD)
}

pub fn heading_style() -> Style {
    Style::new().fg(BLUE).add_modifier(Modifier::BOLD)
}

pub fn dim_style() -> Style {
    Style::new().fg(FG_DIM)
}

pub fn highlight_style() -> Style {
    Style::new().bg(BG_SELECTION).fg(FG_BRIGHT)
}

pub fn key_style() -> Style {
    Style::new().fg(YELLOW).add_modifier(Modifier::BOLD)
}

pub fn value_style() -> Style {
    Style::new().fg(FG)
}

pub fn good_style() -> Style {
    Style::new().fg(GREEN)
}

pub fn warn_style() -> Style {
    Style::new().fg(YELLOW)
}

pub fn bad_style() -> Style {
    Style::new().fg(RED)
}

pub fn accent_style() -> Style {
    Style::new().fg(BLUE)
}

pub fn panel_block_style() -> Style {
    Style::new().fg(FG_DIM).bg(BG_DARK)
}
