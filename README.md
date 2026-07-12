# AI Research Copier Template

This repository contains a **Copier template** for generating standardized AI research projects. It includes scaffolding for ML models, data pipelines, Jupyter notebooks, CI/CD, and more.

## Getting Started

To create a new project based on this template, ensure you have [Copier](https://copier.readthedocs.io/) installed. We recommend using `uv` (or `pipx`):

```bash
uv tool install copier
# or
pipx install copier
```

Then, generate your project:

```bash
copier copy https://github.com/your-username/ai_research_template path/to/your/new/project
```
*(Replace the URL with your actual Git repository URL).*

You will be prompted for project details (name, author, description).

## Updating an Existing Project

If you previously generated a project from this template and want to pull in the latest changes/improvements, go to your generated project's root folder and run:

```bash
copier update
```

## Template Development

If you are modifying this template repository itself:
* The generated project files end in `.jinja` (e.g., `pyproject.toml.jinja`).
* Local linting and formatting configuration for *this* repository is kept in the untemplated `pyproject.toml`.
* To test changes, ensure you add and commit all new/modified files, and generate a dummy project locally:
  ```bash
  git add .
  git commit -m "chore: testing changes"
  copier copy --vcs-ref HEAD . ../test-project
  ```

--------
<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

[![Copier](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/copier-org/copier/master/img/badge/badge-grayscale-inverted-border-teal.json)](https://github.com/copier-org/copier)