AI Research Template
==============================

Project Organization
------------

    ├── LICENSE
    ├── Makefile              <- Makefile with commands like `make data` or `make train`
    ├── README.md             <- The top-level README for developers using this project.
    │
    ├── archive               <- Folders/files which aren't important anymore but kept for reference
    │
    ├── CI-CD                 <- Scripts for automating
    │
    ├── config                <- Keep the configuration files
    │   ├── runs              <- Json/yaml files used for runs
    │   ├── pyproject.toml    <- Project configuration file with package metadata for      
    │   │                     src folder and configuration for tools like black
    │   └── setup.cfg         <- Configuration file for flake8
    │
    ├── data
    │   ├── external          <- Data from third party sources.
    │   ├── interim           <- Intermediate data that has been transformed.
    │   ├── processed         <- The final, canonical data sets for modeling.
    │   ├── raw               <- The original, immutable data dump.
    │   └── temporary         <- Temporrary files/folders which are generated during a run
    │
    ├── docs                  <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models                <- Trained and serialized models, model predictions, or model summaries
    │   ├── baseline          <- Models which we keep for determining the baseline F1-score
    │   ├── experimenting     <- Models on which we didn't finish tuning their hyperparameters or describing their structure
    │   └── production       <- Models which are ready for production and have a strong point 
    │                         like highest F1-score, lowest latency, lowest size, etc
    │
    ├── notebooks             <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │   ├── exploratory       <- explore the dataset
    │   └── reports           <- For producing reports 
    │
    ├── pipelines             <- Scripts for end-to-end data preparing or model training 
    │                         (Pipelines it might be removed because it overlaps with CI-CD and src/data/make_dataset.py)
    ├── references            <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports               <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   │
    │   ├── figures           <- Generated graphics and figures to be used in reporting
    │   ├── runs              <- Documents and json/yaml files of the run
    │   ├── statistics        <- General statistics about the dataset or model
    │   └── tensorboard       <- Tensoboard logs for detailed history of model training.
    │
    ├── requirements.txt      <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py              <- makes project pip installable (pip install -e .) so src can be imported
    │
    ├── src                   <- Source code for use in this project.
    │   ├── __init__.py       <- Makes src a Python module
    │   │
    │   ├── data              <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features          <- Scripts to turn raw data into features for modeling
    │   │   ├── feature_engineering
    │   │   ├── feature_selection
    │   │   ├── preprocessing
    │   │   └── build_features.py (it might be added to one of the subfolders)
    │   │
    │   ├── models            <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── production    <- Models in production
    │   │   ├── experimenting <- Models in testing
    │   │   ├── baseline      <- Models used as baseline to compare models
    │   │   ├── custom_blocks <- Custom blocks and custom layers          
    │   │   ├── monitoring    <- Code for monitoring and tracking the models
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   ├── tests             <- To test data validity, model etc
    │   │
    │   ├── utils             <- Utility files, it can include the logging code
    │   │
    │   ├── visualization     <- Scripts to create exploratory and results oriented visualizations
    │   │   └── visualize.py
    │   │
    │   └── uncategorised     <- To be decided
    │       ├── dvc.yaml
    │       ├── lint.sh       <- To check syntatical correctness (Most likely would move it in the test folder)
    │       └──  metadatafile.yaml  <- To keep track of the model information, it helps with CI/CD and pipeline automation
    │
    └── tox.ini               <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
