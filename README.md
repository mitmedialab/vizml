# VizML: Training Data, Feature Extraction, and Model Training

This repository provides access to the **Plotly dataset-visualization pairs**, **feature extraction scripts**, and **model training ssripts** used in the VizML paper.

<img src="docs/assets/flow.png" width="500" />

# Data Description
We provide **subsets** of the Plotly corpus with 10K and 100K pairs, the **full corpus** with 1,066,443 pairs(205G), and **features** extracted from an aggressively deduplicated set of 119,815 pairs (19G). More information about the corpus schema, the extracted features, and the design choices are provided in the paper.

# Dependencies

This repository uses python 3.7.3 and depends on the packages listed in `requirements.txt`. Create the virtual environment with `virtualenv -p python3 venv`, enter the virtual environment using `source venv/bin/activate`, and install dependencies with `pip install -r requirements.txt`.

# How do I use this repository?

## Accessing Data
To download and unzip the Plotly dataset-visualization pairs or features, run `./retrieve_data.sh`. Comment lines to specify which subsets or features you want to use. Then create a symlink for to access the data `ln -s data/[ plotly_plots_with_full_data_with_all_fields_and_header_{ 1k, 100k, full }.tsv data/plot_data.tsv`.

## Preparing Data
Within the `data_cleaning` directory:
- To remove charts without all data: `python remove_charts_without_all_data.py`
- To remove duplicate charts: `python remove_duplicate_charts.py`

## Extracting and Characterizing Features
Within the `feature_extraction` directory, run `python extract.py`. Then use `notebooks/Plotly Performance.ipynb` to characterize features (_e.g._ distribution of number of columns per dataset)

## Baseline Model Training
Use `notebooks/Descriptive Statistics.ipynb` to train the random forest, K-nearest neighbors, naive Bayes, and Logistic regression baseline models. Use `notebooks/Model Feature Importances.ipynb` to extract feature importances from the random forest baseline model.

## Neural Network Training
Within the `neural_network` directory, run `python agg.py [LOAD|TRAIN|EVAL]` to load features, train models, then evaluate a particular model.

## Benchmarking
Use `notebooks/Benchmarking.ipynb` to evaluate serialized models against the crowdsourced consensus ground truth.

# What's in this repository
```
retrieve_data.sh: Shell script to download and unzip dataset - visualization pairs and features from Amazon S3 storage
requirements.txt: Python dependencies
data/: Placeholder directory for raw data
features/: Placeholder directory for extracted features
results/: Placeholder directory for intermediate results and figures
models/: Placeholder directory for trained models
feature_extraction/
    └───features/
        └───aggregate_single_field_features.py: Functions to aggregate single - column features
        └───aggregation_helpers.py: Helper functions used in aggregate_single_field_features.py
        └───dateparser.py: Functions to detect and mark dates
        └───helpers.py: Helper functions used in all feature extraction scripts
        └───single_field_features.py: Functions to extract single - column features
        └───transform.py: Functions to transform single - column features
        └───type_detection.py: Functions used to detect data types
    └───outcomes/
        └───chart_outcomes.py: Functions to extract design choices of visualizations
        └───field_encoding_outcomes.py: Functions to extract design choices of encodings
    └───extract.py: Top -level entry point to extract features and outcomes
    └───general_helpers.py: Helpers used in top -level extraction function
helpers/
    └───analysis.py: Helpers functions when training baseline models
    └───processing.py: Helper functions when processing data
    └───util.py: Misc helper functions
neural_network/
    └───agg.py: Top-level entry point to load features and train neural network
    └───evaluate.py: Functions to evaluate trained neural network
    └───nets.py: Class definitions for neural network
    └───paper_ground_truth.py: Script to evaluate best network against benchmarking ground truth
    └───paper_tasks.py: Script to evaluate best network for Plotly test set
    └───save_field.py: Script to prepare training, validation, and testing splits
    └───train.py: Helper functions for model training
    └───train_field.py: Script to train network
    └───util.py: Helper functions
notebooks/
    └───Descriptive Statistics.ipynb: Notebook to generate visualizations of number of charts per user, number of rows per dataset, and number of columns per dataset
    └───Plotly Performance.ipynb: Notebook to train baseline models and assess performance on a hold-out setfrom the Plotly corpus
    └───Model Feature Importances.ipynb: Notebook to extract feature importances from trained models
    └───Benchmarking.ipynb: Notebook to generate predictions of trained models on benchmarking datasets, bootstrap crowdsourced consensus, and compare predictions
preprocessing/: Scripts to preprocess features before ML modeling
    └───deduplication.py: Helper functions to deduplicate charts
    └───impute.py: Helper function to impute missing values
    └───preprocess.py: Helper functions to prepare features for learning
docs/: Landing page and miscellaneous material for documentation
```
