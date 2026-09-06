import os
import tomllib
from pathlib import Path

import yaml
from invoke import Exit, task

# =============================================================================
# GLOBALS & PATHS
# =============================================================================
PROJECT_DIR = Path(__file__).parent.resolve()
YAML_PARAMS_FILE = PROJECT_DIR / "config" / "runs" / "env_params.yaml"
COMPOSE_FILE = PROJECT_DIR / "CI-CD" / "mlops_compose_stack.yaml"

VENV_PATH = PROJECT_DIR / ".venv"
PYTHON_VERSION = "3.12"
PYTHON = VENV_PATH / "bin" / "python"
PACKAGE_NAME = "src"


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def ensure_venv(c):
    """File-based rule equivalent: Create venv if it doesn't exist."""
    if not VENV_PATH.exists():
        print("-> Virtual environment not found. Creating...")
        c.run(f"uv venv --python {PYTHON_VERSION} {VENV_PATH}", echo=True, pty=True)


def setup_docker_env():
    """Reads YAML directly in Python and sets OS environment variables for Docker."""
    if not YAML_PARAMS_FILE.exists():
        print(f"⚠️  Warning: {YAML_PARAMS_FILE} not found. Using default run names.")
        run_name, study_name = "default_run", "default_study"
    else:
        with open(YAML_PARAMS_FILE, "r") as f:
            params = yaml.safe_load(f) or {}

        run_name = params.get("name", "default_run")
        study_name = params.get("model_seizure", {}).get("optuna_parameters", {}).get("study_name", "default_study")

    optuna_dir = PROJECT_DIR / f"reports/runs/{run_name}/{study_name}"
    tb_dir = PROJECT_DIR / f"reports/tensorboard/{run_name}/logs_optuna/{study_name}"

    # Export to environment so 'docker compose' inherits them
    os.environ["OPTUNA_LOG_DIR"] = str(optuna_dir)
    os.environ["TENSORBOARD_LOG_DIR"] = str(tb_dir)

    print(f"   -> Env injected: OPTUNA_LOG_DIR={optuna_dir}")
    print(f"   -> Env injected: TENSORBOARD_LOG_DIR={tb_dir}")


# =============================================================================
# ENVIRONMENT & DEPENDENCY TASKS
# =============================================================================
@task
def install(c):
    """Install/sync project dependencies using uv"""
    ensure_venv(c)
    print("-> Compiling and syncing dependencies with uv...")
    c.run(f"uv pip compile pyproject.toml --extra dev -o requirements.txt --python {PYTHON}", echo=True, pty=True)
    c.run(f"uv pip sync requirements.txt --python {PYTHON}", echo=True, pty=True)
    c.run(f"uv pip install -e . --python {PYTHON}", echo=True, pty=True)


@task
def setup(c):
    """Set up the full development environment"""
    install(c)
    print("\n✅ Setup complete. Activate the venv with: source .venv/bin/activate")


@task
def reqs_prod(c):
    """Compile production dependencies into requirements-prod.txt"""
    ensure_venv(c)
    print("-> Compiling production dependencies into requirements-prod.txt with uv...")
    c.run(f"uv pip compile pyproject.toml -o requirements-prod.txt --python {PYTHON}", echo=True, pty=True)


@task
def lock_prod(c):
    """Compile secure production lockfile with hashes"""
    ensure_venv(c)
    print("-> Compiling secure production lockfile with hashes...")
    c.run(f"uv pip compile pyproject.toml --generate-hashes -o requirements-prod.lock --python {PYTHON}", echo=True, pty=True)


@task
def uv_lock(c):
    """Generate or update the universal uv.lock file"""
    print("-> Generating universal uv.lock file...")
    c.run("uv lock", echo=True, pty=True)


@task
def uv_sync(c):
    """Install/sync dependencies from uv.lock"""
    print("-> Syncing environment with uv sync...")
    c.run("uv sync", echo=True, pty=True)


@task
def uv_sync_prod(c):
    """Install/sync ONLY production dependencies from uv.lock"""
    print("-> Syncing production environment with uv sync...")
    c.run("uv sync --no-dev --no-default-groups", echo=True, pty=True)


# =============================================================================
# TOOLS INSTALLATION
# =============================================================================
@task
def tools(c):
    """Install development tools from pyproject.toml with pipx"""
    print("-> Installing development tools from pyproject.toml with pipx...")

    # 1. Parse pyproject.toml natively in Python
    with open(PROJECT_DIR / "pyproject.toml", "rb") as f:
        data = tomllib.load(f)
    pipx_packages = data.get("tool", {}).get("project-tools", {}).get("pipx_packages", [])

    # 2. Get currently installed tools (hide the terminal output for this check)
    installed = c.run("pipx list --short", warn=True, hide=True).stdout

    # 3. Install missing tools
    for spec in pipx_packages:
        # Strip versions/extras to check existence (e.g., 'ruff==0.5.0' -> 'ruff')
        clean_name = spec.split("=")[0].split("[")[0].split(">")[0].split("<")[0]
        if clean_name in installed:
            print(f"   -> '{clean_name}' is already installed. Skipping.")
        else:
            print(f"   -> Installing '{spec}' with pipx...")
            c.run(f"pipx install '{spec}'", echo=True, pty=True)

    # 4. Inject plugins
    print("\n-> Checking for plugins to inject...")
    injected = c.run("pipx list --include-injected", warn=True, hide=True).stdout

    if "jupyterlab" in installed and "jupyterlab-optuna" not in injected:
        c.run("pipx inject jupyterlab jupyterlab-optuna", echo=True, pty=True)

    if "pytest" in installed and "pytest-cov" not in injected:
        c.run("pipx inject pytest pytest-cov", echo=True, pty=True)

    if "mypy" in installed:
        mypy_plugins = [
            "pydantic",
            "pandas-stubs",
            "types-PyYAML",
            "types-Pygments",
            "types-cffi",
            "types-colorama",
            "types-jsonschema",
            "types-protobuf",
            "types-pyasn1",
            "types-python-dateutil",
            "types-shapely",
            "types-tqdm",
        ]
        for plugin in mypy_plugins:
            if plugin not in injected:
                c.run(f"pipx inject mypy {plugin}", echo=True, pty=True)


# =============================================================================
# DEVELOPMENT TASKS (Data, Linting, Formatting)
# =============================================================================
@task
def data(c):
    """Make Dataset"""
    ensure_venv(c)
    c.run(f"{PYTHON} src/data/make_dataset.py data/raw data/processed", echo=True, pty=True)


@task
def clean(c):
    """Delete all compiled Python files and caches"""
    c.run('find . -type f -name "*.py[co]" -delete', echo=True)
    c.run('find . -type d -name "__pycache__" -delete', echo=True)


@task
def lint(c):
    """Lint using ruff, mypy, and sqlfluff"""
    print("-> Running linter...")
    ruff_res = c.run("ruff check .", echo=True, pty=True, warn=True)
    mypy_res = c.run(f"mypy --python-executable {PYTHON} -p {PACKAGE_NAME}", echo=True, pty=True, warn=True)
    sql_res = c.run(f"{PYTHON} -m sqlfluff lint src/", echo=True, pty=True, warn=True)

    if any(res.failed for res in [ruff_res, mypy_res, sql_res]):
        raise Exit("Linting failed. Please fix the errors above.", code=1)


@task
def format(c):
    """Format using ruff and sqlfluff"""
    print("-> Running formatters...")
    c.run("ruff format .", echo=True, pty=True, warn=True)
    c.run(f"{PYTHON} -m sqlfluff fix src/", echo=True, pty=True, warn=True)


# =============================================================================
# ML PIPELINE TASKS
# =============================================================================
@task
def features(c):
    """Build features from processed data"""
    ensure_venv(c)
    print("-> Building features...")
    c.run(f"{PYTHON} src/features/build_features.py", echo=True, pty=True)


@task
def train(c):
    """Train the model"""
    ensure_venv(c)
    print("-> Training model...")
    c.run(f"{PYTHON} src/models/train_model.py", echo=True, pty=True)


@task
def predict(c):
    """Generate predictions/evaluate model"""
    ensure_venv(c)
    print("-> Generating predictions...")
    c.run(f"{PYTHON} src/models/predict_model.py", echo=True, pty=True)


# =============================================================================
# TESTING & QUALITY TASKS
# =============================================================================
@task
def test(c):
    """Run all tests using pytest"""
    ensure_venv(c)
    print("-> Running tests...")
    c.run(f"{PYTHON} -m pytest tests/", echo=True, pty=True)


@task
def check(c):
    """Run all quality checks: format, lint, and test"""
    format(c)
    lint(c)
    test(c)
    print("\n✅ All quality checks passed!")


# =============================================================================
# INFRASTRUCTURE & VISUALIZATION
# =============================================================================
@task
def deploy_vis(c):
    """Deploy visualization servers: tensorboard, optuna, mlflow"""
    setup_docker_env()
    print("-> Starting Docker Compose with services...")
    c.run(f"docker compose -f {COMPOSE_FILE} up -d", echo=True, pty=True)


@task
def stop_vis(c):
    """Stop the visualization servers"""
    setup_docker_env()
    print("-> Stopping Docker Compose with services...")
    c.run(f"docker compose -f {COMPOSE_FILE} down", echo=True, pty=True)


# =============================================================================
# DOCKER EDGE IMAGE TASKS
# =============================================================================
@task
def docker_build(c, image_name="tuh-edge-inference", output_dir="build/docker_images"):
    """Build Docker images for edge inference (amd64 & arm64) and save them to output_dir"""
    output_path = PROJECT_DIR / output_dir
    output_path.mkdir(parents=True, exist_ok=True)

    print("-> Building for amd64 (Linux/Ubuntu/Debian) architecture...")
    c.run(f"docker build -t {image_name}:amd64 -f CI-CD/Dockerfile.edge .", echo=True, pty=True)
    print(f"-> Saving amd64 image to {output_path / f'{image_name}-amd64.tar'}")
    c.run(f"docker save -o {output_path / f'{image_name}-amd64.tar'} {image_name}:amd64", echo=True, pty=True)

    print("-> Building for arm64 architecture (Requires Docker Buildx)...")
    c.run(f"docker buildx build --platform linux/arm64 -t {image_name}:arm64 -f CI-CD/Dockerfile.edge --load .", echo=True, pty=True)
    print(f"-> Saving arm64 image to {output_path / f'{image_name}-arm64.tar'}")
    c.run(f"docker save -o {output_path / f'{image_name}-arm64.tar'} {image_name}:arm64", echo=True, pty=True)


@task
def docker_benchmark(c, image="tuh-edge-inference", arch="amd64", runs=10):
    """Run docker image X times for each model type (onnx, tflite) for benchmarking"""
    full_image = f"{image}:{arch}"

    print("-> Benchmarking TFLite...")
    c.run(
        f"{PYTHON} pipelines/scripts/benchmark_edge.py --image {full_image} --model_type tflite "
        f"--model_path models/production/tf_model_int8.tflite --runs {runs}",
        echo=True,
        pty=True,
    )

    print("\n-> Benchmarking ONNX...")
    c.run(
        f"{PYTHON} pipelines/scripts/benchmark_edge.py --image {full_image} --model_type onnx --model_path models/production/tf_model.onnx --runs {runs}",
        echo=True,
        pty=True,
    )
