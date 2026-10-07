# RAPS Makefile. Run `make help` to list targets.

SHELL := /bin/bash
.DEFAULT_GOAL := help

PYTHON     ?= python3
PIP        ?= pip
IMAGE_NAME ?= raps
PLATFORM   ?= linux/amd64
NPROC      ?= 8
ARGS       ?=

FMU_MODELS_REPO := git@code.ornl.gov:exadigit/fmu-models.git
FMU_MODELS_DIR  := models/fmu-models
POWER9CSM_URL   := https://code.ornl.gov/exadigit/POWER9CSM/-/archive/main/POWER9CSM-main.zip
POWER9CSM_DIR   := models/POWER9CSM

.PHONY: help all pip run test test-fast test-unit test-nodata lint \
        pre-commit docker_build docker_run fetch-fmu-models \
        fetch-example-fmus clean clean-all

help: ## Show this help
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / \
		{printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo
	@echo "Variables: PYTHON, PIP, NPROC, ARGS, IMAGE_NAME, PLATFORM"
	@echo "Example:   make run ARGS=\"run --system frontier\""

all: pip ## Alias for pip

pip: ## Install RAPS in editable mode
	$(PIP) install -e .

run: ## Run main.py (pass options with ARGS="...")
	$(PYTHON) ./main.py $(ARGS)

test: ## Run the full test suite in parallel (NPROC workers)
	$(PYTHON) -m pytest -n $(NPROC) $(ARGS)

test-fast: ## Run tests excluding long ones
	$(PYTHON) -m pytest -n $(NPROC) -m "not long" $(ARGS)

test-unit: ## Run unit tests only
	$(PYTHON) -m pytest -n $(NPROC) -m unit $(ARGS)

test-nodata: ## Run tests that need no external data
	$(PYTHON) -m pytest -n $(NPROC) -m nodata $(ARGS)

lint: ## Run flake8 on the repo
	$(PYTHON) -m flake8 $(ARGS)

pre-commit: ## Run all pre-commit hooks on every file
	pre-commit run --all-files

docker_build: ## Build the Docker image
	docker build --platform $(PLATFORM) -t $(IMAGE_NAME) .

docker_run: ## Run the Docker image interactively
	docker run --platform $(PLATFORM) -it $(IMAGE_NAME)

fetch-fmu-models: ## Clone or update the fmu-models repo
	@if [ ! -d $(FMU_MODELS_DIR) ]; then \
		git clone $(FMU_MODELS_REPO) $(FMU_MODELS_DIR); \
	else \
		git -C $(FMU_MODELS_DIR) pull; \
	fi

fetch-example-fmus: ## Download example FMUs from POWER9CSM
	@echo "Fetching 'fmus' folder from POWER9CSM..."
	@tmp=$$(mktemp -d models/.tmp.XXXXXX) && trap 'rm -rf "$$tmp"' EXIT && \
		curl -fL -o "$$tmp/POWER9CSM.zip" $(POWER9CSM_URL) && \
		unzip -q "$$tmp/POWER9CSM.zip" -d "$$tmp" && \
		mkdir -p $(POWER9CSM_DIR) && \
		rm -rf $(POWER9CSM_DIR)/fmus && \
		mv "$$tmp/POWER9CSM-main/fmus" $(POWER9CSM_DIR)/fmus
	@echo "Copied 'fmus' folder from POWER9CSM -> $(POWER9CSM_DIR)"

clean: ## Remove caches and build artifacts
	find . -type d \( -name __pycache__ -o -name .pytest_cache -o -name .ruff_cache \) \
		-not -path './.venv/*' -not -path './venv/*' -prune -exec rm -rf {} +
	rm -rf build dist *.egg-info .coverage htmlcov

clean-all: clean ## Also remove test-output/ and raps-output-* directories
	rm -rf test-output raps-output-*
