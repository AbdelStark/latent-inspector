# latent-inspector Makefile
# ────────────────────────────────────────────────────────────────

CARGO       := cargo
RELEASE     := --release
BIN         := target/release/latent-inspector
MODELS      := dinov2-vit-l14,ijepa-vit-h14,vjepa2-vitl-fpc2-256,eupe-vit-b16
SAMPLE_IMG  := docs/assets/img/samples/elephant_sample_image.jpg
SAMPLE_DATASET := docs/assets/img/samples
CACHE_DIR   := $(HOME)/.cache/latent-inspector
COVERAGE_IGNORE := (^|/)src/tui/|(^|/)src/cli/tui.rs$$
COVERAGE_LINE_MIN := 85
COVERAGE_FUNCTION_MIN := 80

.PHONY: build build-release build-stub check clippy coverage coverage-ci fmt \
        test clean tui tui-demo inspect compare compare-dataset models download-models help

# ── Build targets ────────────────────────────────────────────────

build:                ## Build the debug binary
	$(CARGO) build

build-release:        ## Build the release binary
	$(CARGO) build $(RELEASE)

build-stub:           ## Build default binary; use LATENT_INSPECTOR_MODEL_BACKEND=stub at runtime
	$(CARGO) build

# ── Quality ──────────────────────────────────────────────────────

check:                ## Type-check the codebase
	$(CARGO) check

clippy:               ## Lint with clippy (warnings = errors)
	$(CARGO) clippy --all-targets -- -D warnings

coverage:             ## Coverage summary for the tested non-TUI surface (requires cargo-llvm-cov)
	$(CARGO) llvm-cov --workspace --ignore-filename-regex '$(COVERAGE_IGNORE)' --summary-only

coverage-ci:          ## Enforce repository coverage thresholds in CI (requires cargo-llvm-cov)
	$(CARGO) llvm-cov --workspace --ignore-filename-regex '$(COVERAGE_IGNORE)' \
		--fail-under-lines $(COVERAGE_LINE_MIN) --fail-under-functions $(COVERAGE_FUNCTION_MIN) \
		--summary-only

fmt:                  ## Format code
	$(CARGO) fmt

test:                 ## Run all tests
	$(CARGO) test

# ── Run targets ──────────────────────────────────────────────────

tui: build-release    ## Launch TUI with real ONNX inference on sample image
	$(BIN) tui $(SAMPLE_IMG) -m $(MODELS)

tui-demo: build-release ## Launch TUI in demo mode with the release binary
	$(BIN) tui

inspect: build-release ## Inspect sample image with DINOv2
	$(BIN) inspect $(SAMPLE_IMG) --model dinov2-vit-l14

compare: build-release ## Compare models on sample image and write an HTML report
	@outdir=./target/comparison-reports/$$(date +%Y%m%d-%H%M%S)/; \
	$(BIN) compare $(SAMPLE_IMG) --models $(MODELS) --format html --output "$$outdir"

compare-dataset: build-release ## Profile all configured models on the sample dataset and write HTML reports
	@outdir=./target/profile-reports/$$(date +%Y%m%d-%H%M%S)/; \
	for model in $$(printf '%s' "$(MODELS)" | tr ',' ' '); do \
		echo "Profiling $$model on $(SAMPLE_DATASET) -> $$outdir/$$model"; \
		$(BIN) profile --model "$$model" --dataset "$(SAMPLE_DATASET)" --format html --output "$$outdir/$$model"; \
	done

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
