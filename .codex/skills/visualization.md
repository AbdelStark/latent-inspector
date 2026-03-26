---
name: visualization
description: Output rendering — terminal (ratatui + Unicode + ANSI), PNG export (attention overlays, PCA RGB, heatmaps), JSON structured output, and HTML interactive reports. Activate when working on any output rendering, display formatting, or export functionality.
prerequisites: ratatui, crossterm, image crate, serde_json
---

# Visualization

<purpose>
Covers all output rendering: terminal display with Unicode blocks and ANSI colors, PNG image export with attention map overlays, JSON metric export for scripting, and self-contained HTML interactive reports. Each format is a separate renderer implementing a common trait.
</purpose>

<context>
— 4 output formats: terminal (default), PNG, JSON, HTML
— Terminal: ratatui for layout, crossterm for terminal control, Unicode ▓░▒ for attention
— PNG: image crate for writing, color-mapped heatmaps, PCA → RGB channels
— JSON: serde_json, all numeric metrics as structured output
— HTML: self-contained single file, inline CSS/JS, no external deps
— Renderer trait: all formats implement same interface for consistent behavior
</context>

<procedure>
### Implement a renderer
1. Define trait:
   ```rust
   pub trait Renderer {
       fn render_inspect(&self, output: &InspectResult, dest: &OutputDest) -> Result<()>;
       fn render_compare(&self, output: &CompareResult, dest: &OutputDest) -> Result<()>;
   }
   ```
2. Implement for each format: TerminalRenderer, PngRenderer, JsonRenderer, HtmlRenderer
3. OutputDest: stdout for terminal, directory path for file-based formats

### Terminal attention maps
1. Map attention weights [N] to Unicode density: ░ (low) → ▒ (med) → ▓ (high)
2. Arrange in spatial grid matching image patch layout (e.g., 16×16 for 256 patches)
3. Use ANSI colors for intensity: blue (low) → yellow (mid) → red (high)
4. Check terminal width to auto-scale or truncate

### PCA → RGB visualization
1. Take top 3 PCA components from patch features
2. Map to RGB channels: PC1 → R, PC2 → G, PC3 → B
3. Normalize each channel to [0, 255]
4. Reshape to spatial grid → save as PNG
5. Regions with similar features appear as similar colors

### JSON output structure
```json
{
  "image": "path/to/image.jpg",
  "models": {
    "dinov2": {
      "rank": 487,
      "total_dims": 1024,
      "top10_variance_pct": 23.4,
      "gini": 0.72,
      "patch_entropy": 6.82
    }
  },
  "cross_model": {
    "cls_cosine": { "dinov2_ijepa": 0.721 },
    "cka": { "dinov2_ijepa": 0.834 }
  }
}
```

### HTML report
1. Generate self-contained HTML with inline CSS and JS
2. Embed images as base64 data URIs
3. Use CSS grid for side-by-side model comparison
4. Add hover-to-compare: mouseover switches between model overlays
5. Include metrics tables with sortable columns
</procedure>

<patterns>
<do>
  — Check terminal capabilities before using Unicode/colors: fallback to ASCII on dumb terminals.
  — Use color gradients from established colormaps (viridis, inferno) for heatmaps.
  — Keep JSON output flat and scriptable — `jq` should easily extract any metric.
  — Make HTML reports work offline — no CDN links, everything inlined.
  — Use ratatui's `Table` widget for metric comparison tables.
</do>
<dont>
  — Don't hardcode terminal width — query with `crossterm::terminal::size()`.
  — Don't use raw ANSI escape codes — use crossterm's API for portability.
  — Don't generate multi-megabyte HTML — compress base64 images, use JPEG for photos.
  — Don't mix rendering logic with analysis — renderers receive computed results only.
</dont>
</patterns>

<examples>
Example: Mapping attention to Unicode blocks
```rust
fn attention_to_char(weight: f32, max_weight: f32) -> char {
    let normalized = weight / max_weight;
    match normalized {
        x if x > 0.75 => '█',
        x if x > 0.50 => '▓',
        x if x > 0.25 => '▒',
        _ => '░',
    }
}
```
</examples>

<troubleshooting>
| Symptom | Cause | Fix |
|---------|-------|-----|
| Terminal output garbled | Terminal doesn't support Unicode | Detect with `$TERM`, fall back to ASCII `#=-. ` |
| Colors not showing | Piped to file / no color support | Check `atty::is(Stream::Stdout)`, disable colors if not TTY |
| PNG output blank | All-zero PCA components | Verify PCA ran correctly, check for dead features |
| HTML too large | Uncompressed base64 images | Use JPEG encoding before base64, reduce resolution |
</troubleshooting>

<references>
— src/viz/terminal.rs: Terminal renderer
— src/viz/png.rs: PNG export
— src/viz/json.rs: JSON output
— src/viz/html.rs: HTML report
— SPECIFICATION.md: Output format specifications
</references>
