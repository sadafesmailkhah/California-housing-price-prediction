# California Housing Price Prediction

This project is a machine learning regression project for predicting
median house prices using the California Housing dataset. It covers
the main steps of an ML workflow, from exploring and preparing the
data to training, evaluating, and tuning different models.

## Overview

This project was built for learning and practice. Several models
were tested to see which one could perform best and stay stable
across different data splits.

The final test RMSE was affected by a known limitation in the test
data (capped house values around $500k — more on this in the Results
section below), which made the raw test score look worse than it
actually is. When that limitation is accounted for, the RMSE comes
out to around $40,000, which is a reasonably good result.

### Dataset

- California housing data — ~20,000 rows, features include location,
  income, housing age, room counts, and proximity to the ocean
- Target: `median_house_value`

### 1. Exploratory Data Analysis

- Plotted the raw feature distributions (before any preprocessing) to
  get a general sense of the data
- Checked correlations between features to understand direct and
  indirect relationships, as well as their relationship with the
  target. Found that longitude/latitude are strongly correlated
  (expected, since they're geographic coordinates), and median income
  has a strong direct correlation with house value
- Created an income category column to use with
  `StratifiedShuffleSplit`, so the train/test split preserves the
  same income distribution in both sets — this column was dropped
  after splitting
- Plotted scatter plots for the features with the strongest
  correlations to get a clearer visual understanding before moving on
  to preprocessing

### 2. Feature Engineering

Used `skew()` to identify skewed numeric features, then confirmed
visually with histograms. Out of 8 numeric features, 5 were
right-skewed and 3 were multimodal.

- **Right-skewed features** (`total_rooms`, `total_bedrooms`,
  `population`, `households`, `median_income`): transformed using
  `PowerTransformer` to bring them closer to a normal distribution
  (also standardizes the scale)
- **`housing_median_age`** (multimodal, but a single independent
  feature): built a custom transformer using RBF kernel similarity —
  picks reference point(s) near the distribution's peaks and computes
  each sample's similarity to them, with configurable gamma and
  reference values
- **`longitude`/`latitude`** (multimodal, but only meaningful
  together, not separately): built a separate custom transformer
  combining KMeans clustering with RBF similarity, since these two
  features needed to be considered jointly rather than independently

Also created additional combined features (feature engineering) from
existing columns to give the model more useful signal — e.g. ratios
like rooms per household.

### 3. Preprocessing Pipeline

Each feature (or group of features) needs a different sequence of
transformations, so each one was wrapped in its own `Pipeline` (e.g.
impute → transform → scale). All these pipelines were then combined
using `ColumnTransformer`, which applies each one to its respective
columns and concatenates the results into the final feature set.

Categorical data (`ocean_proximity`) was handled with a
`SimpleImputer` + `OneHotEncoder` pipeline, following the same pattern.

### 4. Model Comparison

Tested 9 models total: 7 linear models (including 2 robust regressors
— `HuberRegressor` and `RANSACRegressor`) and 2 tree-based models.

Used `cross_val_score` with 5 folds to measure RMSE for each model.
Results:

- `SGDRegressor` and `RANSACRegressor` performed poorly and were
  unstable (high mean and std)
- `LinearRegression`, `Ridge`, and `Lasso` all performed similarly
  well and stayed stable — interestingly, `Ridge`/`Lasso` didn't show
  much improvement over plain `LinearRegression`
- `ElasticNet` also underperformed
- `RandomForestRegressor` had both the best performance and the best
  stability among all models tested

Based on this, `RandomForestRegressor` was selected for further tuning.

### 5. Hyperparameter Tuning

With default hyperparameters, RandomForest already performed well,
but tuning was needed to get the best possible result. Also, before
fitting the final model, the target column itself needed some
cleanup — not just the training features.

Plotted the target (`median_house_value`) distribution and found it
was right-skewed, with the tail end representing an artificial cap:
any house originally worth more than ~$500,000 was clipped down to
exactly that value in this dataset. Ignoring that tail, the
distribution is right-skewed (somewhat log-normal), which can hurt
model performance since it's not learning from a well-behaved
distribution.

To fix the skew, applied `log1p` to the target instead of a regular
logarithm, so it's also defined at zero. Since the target is
transformed, predictions come out in log-scale too, so the inverse
(`expm1`) is needed to convert them back to actual dollar values —
handled with `TransformedTargetRegressor`.

The full pipeline (preprocessing + model + target transform) was then
passed into `RandomizedSearchCV` to search for good hyperparameters.
`GridSearchCV` would try every possible combination and guarantee the
best one within the grid, but it's much more computationally
expensive — so `RandomizedSearchCV` was used instead to get a strong
result in reasonable time.

**Best parameters found:**
- `n_estimators`: 500
- `max_depth`: 30
- `max_features`: 'sqrt'
- `min_samples_split`: 2
- `n_clusters` (geo similarity): 15
- `gamma` (age similarity): 0.05

Note: `n_estimators` and `n_clusters` both landed on the edge of their
search range, suggesting that expanding the range further might yield
even better results — a reasonable next step for improvement.

### 6. Results

- After removing the **787 capped samples** from the training data
  and tuning the model, the best cross-validation RMSE was
  **$40,482.06**
- When evaluating the model on the original test set, the RMSE
  increased to **$49,501.43**. The test set still contained **205
  samples** with the capped value of $500,000
- Since the difference between the CV and test scores was quite
  large, checked whether these capped values were affecting the result
- Filtered out the 205 capped samples from the test set just for this
  analysis. The test RMSE dropped to **$40,055.43**, very close to
  the cross-validation result
- This suggests the **$500,000 cap was the main reason for the
  difference** between the CV and test scores, not an actual problem
  with the model

## Project Structure

```
├── src/housing/          # Data loading and reusable code
│   ├── data.py
│   └── __init__.py
├── notebooks/             # EDA and model experimentation
│   └── 01_eda.ipynb
├── models/                # Saved model files
├── datasets/              # Downloaded dataset files
├── pyproject.toml
├── .gitignore
└── README.md
```

## Setup & Usage

```bash
git clone https://github.com/sadafesmailkhah/California-housing-price-prediction.git
cd California-housing-price-prediction

python -m venv .venv
.venv\Scripts\Activate.ps1

pip install -e .
```

Then open `notebooks/01_eda.ipynb` and run the cells (or run
`python src/housing/data.py` to just fetch the dataset).

## Tech Stack

Python, pandas, NumPy, scikit-learn, matplotlib, seaborn

## Possible Improvements

- Try more advanced models, especially boosting-based models
- Expand the hyperparameter search ranges where the best parameters
  reached the upper end of the search range
- Experiment with different approaches for the geographic features
- Explore other ways of handling the $500,000 cap instead of
  filtering the capped samples
- Add more models and experiments to make the model comparison more
  comprehensive
- Improve the training and prediction workflow as the project grows
