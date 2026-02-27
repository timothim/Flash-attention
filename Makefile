# ═══════════════════════════════════════════════════════════════════════
# Flash Attention 2 — Project Makefile
# ═══════════════════════════════════════════════════════════════════════

CUDA_DIR   := cuda
TRITON_DIR := triton
BENCH_DIR  := benchmarks
BUILD_DIR  := $(CUDA_DIR)/build

.PHONY: all build install clean test test-triton bench bench-triton plots help

# ── Default target ────────────────────────────────────────────────────
all: install

# ── Build the CUDA extension via pip (editable install) ──────────────
install:
	cd $(CUDA_DIR) && pip install -e . --no-build-isolation

# ── CMake build (for standalone testing without PyTorch) ─────────────
build:
	mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && cmake .. -DBUILD_STANDALONE_TEST=ON && make -j$$(nproc)

# ── Tests ─────────────────────────────────────────────────────────────
test: install
	python -m pytest $(CUDA_DIR)/tests/ -v --tb=short

test-triton:
	python -m pytest $(TRITON_DIR)/test_triton.py -v --tb=short

test-all: test test-triton

# ── Benchmarks ────────────────────────────────────────────────────────
bench: install
	python $(BENCH_DIR)/bench_all.py

bench-triton:
	python $(TRITON_DIR)/bench_triton.py

# ── Generate plots from saved results ────────────────────────────────
plots:
	python $(BENCH_DIR)/plot_results.py

# ── Cleanup ───────────────────────────────────────────────────────────
clean:
	rm -rf $(BUILD_DIR)
	rm -rf $(CUDA_DIR)/build $(CUDA_DIR)/dist $(CUDA_DIR)/*.egg-info
	rm -rf $(CUDA_DIR)/__pycache__ $(TRITON_DIR)/__pycache__
	find . -name "*.so" -delete
	find . -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# ── Help ──────────────────────────────────────────────────────────────
help:
	@echo "Flash Attention 2 — Build & Test"
	@echo ""
	@echo "  make install      Build + install CUDA PyTorch extension (editable)"
	@echo "  make build        CMake standalone build (no PyTorch)"
	@echo "  make test         Run CUDA correctness tests (pytest)"
	@echo "  make test-triton  Run Triton correctness tests"
	@echo "  make test-all     Run all tests"
	@echo "  make bench        Run full benchmark suite"
	@echo "  make bench-triton Run Triton-only benchmarks"
	@echo "  make plots        Generate matplotlib plots from results/"
	@echo "  make clean        Remove build artifacts"
