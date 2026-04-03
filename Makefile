# latent-inspector Makefile
# ────────────────────────────────────────────────────────────────

CARGO       := cargo
FEATURES    := --features onnx-inference
RELEASE     := --release
BIN         := target/release/latent-inspector
TUI_MODELS  := dinov2-vit-l14,ijepa-vit-h14
SAMPLE_IMG  := docs/assets/img/samples/elephant_sample_image.jpg
CACHE_DIR   := $(HOME)/.cache/latent-inspector

.PHONY: build build-release build-stub check clippy fmt test clean \
        tui tui-demo inspect compare models download-models help

# ── Build targets ────────────────────────────────────────────────

build:                ## Build debug with real ONNX inference
	$(CARGO) build $(FEATURES)

build-release:        ## Build release with real ONNX inference
	$(CARGO) build $(FEATURES) $(RELEASE)

build-stub:           ## Build without ONNX (stub mode, fast)
	$(CARGO) build

# ── Quality ──────────────────────────────────────────────────────

check:                ## Type-check with ONNX feature
	$(CARGO) check $(FEATURES)

clippy:               ## Lint with clippy (warnings = errors)
	$(CARGO) clippy $(FEATURES) -- -D warnings

fmt:                  ## Format code
	$(CARGO) fmt

test:                 ## Run all tests
	$(CARGO) test $(FEATURES)

# ── Run targets ──────────────────────────────────────────────────

tui: build-release    ## Launch TUI with real ONNX inference on sample image
	$(BIN) tui $(SAMPLE_IMG) -m $(TUI_MODELS)

tui-demo: build-release ## Launch TUI in demo mode with the release binary
	$(BIN) tui

inspect: build-release ## Inspect sample image with DINOv2
	$(BIN) inspect $(SAMPLE_IMG) --model dinov2-vit-l14

compare: build-release ## Compare models on sample image
	$(BIN) compare $(SAMPLE_IMG) --models dinov2-vit-l14

models: build-release ## List registered models and their status
	$(BIN) models

validate: build-release ## Validate DINOv2 preprocessing and tensor contracts
	$(BIN) validate --model dinov2-vit-l14

# ── Model management ─────────────────────────────────────────────

download-models: build-release ## Pre-download the DINOv2 ONNX model (~1.1 GB)
	$(BIN) models --download dinov2-vit-l14

cache-status:         ## Show cached model files
	@echo "Cache directory: $(CACHE_DIR)"
	@ls -lh $(CACHE_DIR)/ 2>/dev/null || echo "  (empty — run 'make download-models' first)"

cache-clean:          ## Remove all cached models (will re-download on next run)
	@echo "This will delete all cached models in $(CACHE_DIR)"
	@echo "Press Ctrl+C to cancel, or Enter to continue..."
	@read _confirm
	rm -rf $(CACHE_DIR)
	@echo "Cache cleared."

# ── Convenience ──────────────────────────────────────────────────

clean:                ## Remove build artifacts
	$(CARGO) clean

all: fmt clippy test build-release ## Full CI pipeline: format, lint, test, build

# ── Help ─────────────────────────────────────────────────────────

help:                 ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
