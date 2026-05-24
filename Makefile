.PHONY: clean data lint format install setup tools deploy_vis stop_vis

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PACKAGE_NAME := src

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
YAML_PARAMS_FILE := ./config/runs/env_params.yaml
VENV_PATH := ./.venv
PROFILE = default
PROJECT_NAME = AiResearchTemplate
PYTHON_VERSION = 3.12
PYTHON_INTERPRETER := $(VENV_PATH)/bin/python
DEV_TOOLS_FROM_PYPROJECT := $(shell $(PYTHON_INTERPRETER) -c "import tomllib; f=open('pyproject.toml', 'rb'); data=tomllib.load(f); print(' '.join(data.get('tool', {}).get('project-tools', {}).get('pipx_packages', [])))")
JUPYTER_PLUGINS := jupyterlab-optuna
COMPOSE_FILE := ./CI-CD/mlops_compose_stack.yaml

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Set up the full development environment
setup: install 
	@echo "✅ Setup complete. Activate the venv with: source .venv/bin/activate"

## Install requirements with uv
install: $(PYTHON_INTERPRETER) ## Install/sync project dependencies using uv
	@echo "-> Compiling and syncing dependencies with uv..."
	@uv pip compile pyproject.toml --extra dev -o requirements.txt --python $(PYTHON_INTERPRETER)
	@uv pip sync requirements.txt --python $(PYTHON_INTERPRETER)
	@uv pip install -e . --python $(PYTHON_INTERPRETER)

## Install pipx tools
tools: $(PYTHON_INTERPRETER)
	@echo "-> Installing development tools from pyproject.toml with pipx..."
	@echo "   Found tools: $(DEV_TOOLS_FROM_PYPROJECT)"
	@for tool_spec in $(DEV_TOOLS_FROM_PYPROJECT); do \
		# Extract the clean package name (e.g., 'ruff' from 'ruff==0.5.0')  \
		# The double '$$' is needed to escape the '$' for Make, so the shell sees '$tool_spec' \
		tool_name=$$(echo "$$tool_spec" | cut -d'=' -f1 | cut -d'[' -f1); \
		\
		# Check if the tool is already in the list. grep -q is silent. \
		if pipx list | grep -q -E "package\s+$$tool_name\s+"; then \
			echo "   -> '$$tool_name' is already installed. Skipping."; \
		else \
			echo "   -> Installing '$$tool_spec' with pipx..."; \
			pipx install "$$tool_spec"; \
		fi \
	done
	# --- After the loop, handle special injections ---

	@echo "-> Checking for plugins to inject..."
	@if pipx list | grep -q -E "package\s+jupyterlab\s+"; then \
		if ! pipx list --include-injected | grep -q -w "jupyterlab-optuna"; then \
			echo "   -> JupyterLab is installed, but plugin is missing. Injecting: $(JUPYTER_PLUGINS)"; \
			pipx inject jupyterlab $(JUPYTER_PLUGINS); \
		else \
			echo "   -> JupyterLab and jupyterlab-optuna plugin are already installed. Skipping."; \
		fi \
	else \
		echo "   -> JupyterLab not found in tool list. Skipping plugin injection."; \
	fi

	@if pipx list | grep -q -E "package\s+pytest\s+"; then \
		if ! pipx list --include-injected | grep -q -w "pytest-cov"; then \
			echo "   -> Pytest is installed via pipx, but coverage plugin is missing. Injecting: pytest-cov"; \
			pipx inject pytest pytest-cov; \
		else \
			echo "   -> Pytest and pytest-cov are already installed. Skipping."; \
		fi \
	else \
		echo "   -> Pytest not found in pipx tool list. Skipping plugin injection."; \
	fi

	@if pipx list | grep -q -E "package\s+mypy\s+"; then \
		for plugin in pydantic pandas-stubs types-PyYAML types-Pygments types-cffi types-colorama types-jsonschema types-protobuf types-pyasn1 types-python-dateutil types-shapely types-tqdm; do \
			if ! pipx list --include-injected | grep -q -w "$$plugin"; then \
				echo "   -> Mypy is installed via pipx, but plugin '$$plugin' is missing. Injecting: $$plugin"; \
				pipx inject mypy "$$plugin"; \
			else \
				echo "   -> Mypy and plugin '$$plugin' are already installed. Skipping."; \
			fi; \
		done; \
	else \
		echo "   -> Mypy not found in pipx tool list. Skipping plugin injection."; \
	fi

	@echo "-> Tool check complete."
#REVIEW - In pipx list, we are not seeing anything about jupyterlab-optuna. Also it is always injecting
## Make Dataset
data: requirements
	$(PYTHON_INTERPRETER) src/data/make_dataset.py data/raw data/processed

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

#################################################################################
# Development Tasks                                                                       #
#################################################################################

## Lint using mypy and flake8
lint:
	@echo "-> Running linter..."
	ruff check .
	mypy --python-executable $(PYTHON_INTERPRETER) -p $(PACKAGE_NAME)
#flake8 src
## Format using ruff
format:
	@echo "-> Running formatters..."
	ruff format .
#black .
#isort .

## Deploy visualization servers: tensorboard, optuna, mlflow
deploy_vis:
	$(SETUP_DOCKER_ENV) \
	\
	echo "-> Starting Docker Compose with services..."; \
	docker compose -f $(COMPOSE_FILE) up -d

## Stop the visualization servers
stop_vis:
	$(SETUP_DOCKER_ENV) \
	\
	echo "-> Stopping Docker Compose with services..."; \
	docker compose -f $(COMPOSE_FILE) down

# This is the new rule that creates the venv.
# It is a FILE-BASED rule, not a phony one.
$(PYTHON_INTERPRETER):
	@echo "-> Virtual environment not found. Creating..."
	@uv venv --python $(PYTHON_VERSION) $(VENV_PATH)

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################



#################################################################################
# Scripts                                                                       #
#################################################################################

define SETUP_DOCKER_ENV
	@RUN_NAME=$$(yq --raw-output '.name' $(YAML_PARAMS_FILE)); \
	STUDY_NAME=$$(yq --raw-output '.model_seizure.optuna_parameters.study_name' $(YAML_PARAMS_FILE)); \
	\
	echo "   -> Name from YAML:       '$$RUN_NAME'"; \
	echo "   -> Study Name from YAML:   '$$STUDY_NAME'"; \
	\
	export OPTUNA_LOG_DIR="$(PROJECT_DIR)/$$RUN_NAME.db"; \
	export TENSORBOARD_LOG_DIR="$(PROJECT_DIR)/reports/tensorboard/$$RUN_NAME/logs_optuna/$$STUDY_NAME"; \
	\
	echo "      OPTUNA_LOG_DIR    = $$OPTUNA_LOG_DIR"; \
	echo "      TENSORBOARD_LOG_DIR = $$TENSORBOARD_LOG_DIR"; 
endef

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')