"""
NumPy House Price Regression

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - impute_nan_with_mean
def impute_nan_with_mean(X):
    res = X.copy()
    nanmean = np.nanmean(res, axis=0)
    nanmean=np.where(np.isnan(nanmean),0,nanmean)
    
    res=np.where(
        np.isnan(res),
        nanmean,
        res
    )
    return res

# Step 2 - compute_iqr_bounds
def compute_iqr_bounds(X, k=1.5):
    N, F = X.shape
    # 计算每一列的q1和q3
    q1 = np.percentile(X, 25, axis=0)
    q3 = np.percentile(X, 75, axis=0)
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    return lower, upper

# Step 3 - clip_columns
def clip_columns(X, lower, upper):
    res = np.clip(X, lower, upper)
    return res

# Step 4 - make_ratio_feature
def make_ratio_feature(numerator, denominator, eps=1e-8):
    safe_denominator = denominator + eps
    return numerator / safe_denominator

# Step 5 - append_column
def append_column(X, col):
    res = np.column_stack((X, col))
    return res

# Step 6 - one_hot_encode
def one_hot_encode(labels):
    categories, inverse = np.unique(labels, return_inverse=True)
    rows = np.arange(len(labels))
    matrix = np.zeros((len(labels), len(categories)), dtype=float)
    matrix[rows,inverse] = 1.0
    return matrix

# Step 7 - fit_standardizer
def fit_standardizer(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std == 0] = 1.0
    return (mean, std)

# Step 8 - apply_standardizer
def apply_standardizer(X, mean, std):
    X_standardized = (X - mean) / std
    return X_standardized

# Step 9 - add_bias_column
def add_bias_column(X):
    bias = np.ones(len(X))
    bias = bias.reshape(-1, 1)
    return np.concatenate((bias, X), axis=1)

# Step 10 - make_shuffled_indices
def make_shuffled_indices(n_samples, seed):
    rng=np.random.default_rng(seed)
    return rng.permutation(n_samples)

# Step 11 - partition_indices (not yet solved)
# TODO: implement

# Step 12 - subset_xy (not yet solved)
# TODO: implement

# Step 13 - ols_fit (not yet solved)
# TODO: implement

# Step 14 - ols_predict (not yet solved)
# TODO: implement

# Step 15 - mean_absolute_error (not yet solved)
# TODO: implement

# Step 16 - root_mean_squared_error (not yet solved)
# TODO: implement

# Step 17 - r_squared (not yet solved)
# TODO: implement

# Step 18 - residual_summary (not yet solved)
# TODO: implement

# Step 19 - prepare_cleaned_features (not yet solved)
# TODO: implement

# Step 20 - assemble_feature_matrix (not yet solved)
# TODO: implement

# Step 21 - make_train_val_test (not yet solved)
# TODO: implement

# Step 22 - standardize_and_add_bias (not yet solved)
# TODO: implement

# Step 23 - evaluate_predictions (not yet solved)
# TODO: implement

# Step 24 - house_price_pipeline (not yet solved)
# TODO: implement

