# Housing Price Fairness Analysis
**Cook County Property Tax Assessment Model**

## Overview

Property taxes in Cook County, Illinois are calculated from assessed home values. When a model systematically overestimates the value of lower-priced homes and underestimates the value of higher-priced homes, the result is **regressive taxation**: lower-income households pay a disproportionately larger share of their income in taxes.

This project builds a linear regression model to predict home sale prices from the Cook County Assessor's Office (CCAO) dataset, then analyzes how prediction errors are distributed across the property value spectrum to surface and quantify patterns of unfair assessment.

## Dataset

- **Source:** Cook County Assessor's Office (CCAO), Illinois
- **Training set:** 204,792 housing records with 62 features
- **Test set:** 55,311 records
- **Target variable:** Log Sale Price (natural log of sale price in dollars)

Key features used: building square footage, land square footage, lot size, age decade, number of apartments, garage indicator, proximity to O'Hare airport noise, central air, floodplain status, road proximity, repair condition, number of bathrooms, neighborhood, geographic coordinates (latitude/longitude), and sale timing (year, quarter, month).

## Methods

### Feature Engineering

All feature engineering is encapsulated in a single pipeline function (`feature_engine_final`) to ensure identical transformations are applied to training and test data.

**Outlier removal:** IQR-based filtering on building square footage, land square footage, and lot size to remove extreme values before modeling.

**Log transformations:** Sale price, building square footage, land square footage, and lot size are log-transformed to reduce skew and improve linearity.

**Neighborhood target encoding:** Neighborhood codes are encoded using smoothed target encoding (Bayesian shrinkage toward the global mean with a smoothing factor of 50) to capture neighborhood-level price signals without overfitting to small neighborhoods.

**One-hot encoding:** Categorical variables such as repair condition, central air, garage indicator, and bathroom count are one-hot encoded with the first category dropped to avoid multicollinearity.

**Price clipping:** Log sale prices are clipped to the \[10, 14\] range (approximately $22,000 to $1,200,000) to remove extreme outliers. The bottom 2% and top 2% of prices are also trimmed.

### Model

Linear regression (ordinary least squares) via `sklearn.linear_model.LinearRegression`. The model is evaluated using Root Mean Squared Error (RMSE) on a held-out validation set and through k-fold cross-validation.

### Fairness Analysis

After fitting the model, predictions are analyzed across the full range of property values to detect systematic bias:

**RMSE by price interval:** RMSE is computed for overlapping windows of log sale price (step 0.5) to show how prediction error varies by property value.

**Overestimation rate by price interval:** The proportion of homes where the model overestimates the assessed value is computed for each price window. An overestimate means the model predicts a higher value than the actual sale price, which would result in the homeowner paying more in property taxes than warranted.

**Key finding:** The model shows a pattern consistent with regressive assessment. Lower-priced homes are overestimated at a higher rate, while higher-priced homes are underestimated at a higher rate. This mirrors the historically documented bias in Cook County's assessment system, where lower-income homeowners were systematically overtaxed.

### Custom Error Metric: MAPE

To address RMSE's limitation of being dominated by errors on expensive properties, the project implements Mean Absolute Percentage Error (MAPE):

```
MAPE(θ, X, y) = mean(|y - Xθ| / |y|)
```

MAPE normalizes each error by the true value, putting errors across different price levels on equal footing. This is a more relevant metric for fairness analysis because a $10,000 error on a $50,000 home is proportionally much more significant than the same error on a $500,000 home.

The MAPE-optimized model is fit using `scipy.optimize.minimize` with the BFGS method, treating the custom loss as the optimization target.

## Repository Structure

```
.
├── projA2.ipynb              # Main analysis notebook
├── feature_func.py           # Feature engineering helpers from Part A1
├── codebook.txt              # Variable descriptions
└── images/                   # Supporting images
```

## Key Results

- The residual plot confirms that lower-valued properties are overestimated at a higher rate than higher-valued properties, consistent with a regressive assessment pattern.
- MAPE analysis shows percentage errors are substantially higher for lower-priced homes, further confirming that standard RMSE understates the burden on lower-income homeowners.
- The custom error metric and neighborhood target encoding improve both predictive accuracy and fairness of the assessment model relative to a naive baseline.

## Context

This project was completed as part of the UC Berkeley Data 100 curriculum and uses real data from the Cook County Assessor's Office. The CCAO has publicly acknowledged historical inequities in its assessment methodology and has undertaken reform efforts. This analysis reproduces the fairness evaluation framework used by the CCAO's own data science team to audit property tax models for regressivity.

## Tools and Libraries

- Python 3
- pandas, NumPy
- scikit-learn (LinearRegression, KFold, OneHotEncoder)
- matplotlib, seaborn
- scipy.optimize (BFGS for custom loss minimization)
