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
    rng=np.random.RandomState(seed)
    return rng.permutation(n_samples)

# Step 11 - partition_indices
def partition_indices(indices, train_ratio, val_ratio):
    N = len(indices)
    train_nums = int(N * train_ratio)
    val_nums = int(N * val_ratio)
    train_idx = indices[:train_nums]
    val_idx = indices[train_nums : train_nums + val_nums]
    test_idx = indices[train_nums + val_nums :]
    return train_idx, val_idx, test_idx

# Step 12 - subset_xy
# 第 12 步：根据索引提取特征和标签子集
def subset_xy(X, y, indices):
    X_sub=X[indices]
    y_sub=y[indices]
    return X_sub,y_sub

# Step 13 - ols_fit
def ols_fit(X, y):
    theta = np.linalg.lstsq(X, y, rcond=None)[0]
    return theta

# Step 14 - ols_predict
def ols_predict(X, theta):
    y_pred = X @ theta
    return y_pred

# Step 15 - mean_absolute_error
def mean_absolute_error(y_true, y_pred):
    error = np.mean(np.abs(y_pred - y_true))
    value = float(error)
    return value

# Step 16 - root_mean_squared_error
def root_mean_squared_error(y_true, y_pred):
    error = np.sqrt(np.mean((y_true - y_pred) ** 2))
    value = float(error)
    return value

# Step 17 - r_squared
def r_squared(y_true, y_pred):
    SS_res = float(np.sum((y_true - y_pred) ** 2))
    y_true_mean = float(np.mean(y_true))
    SS_tot = float(np.sum((y_true - y_true_mean) ** 2))
    if SS_tot == 0:
        r_2 = 0.0
    else:
        r_2 = 1 - SS_res / SS_tot
    return r_2

# Step 18 - residual_summary
def residual_summary(y_true, y_pred):
    dict={}
    error = y_true - y_pred
    mean = float(np.mean(error))
    dict["mean"]=mean
    std = float(np.std(error))
    dict["std"]=std
    median_abs = float(np.median(np.abs(error)))
    dict["median_abs"]=median_abs
    return dict

# Step 19 - prepare_cleaned_features
def prepare_cleaned_features(X, iqr_k=1.5):
    X_clean = impute_nan_with_mean(X)
    lower, upper = compute_iqr_bounds(X_clean, iqr_k)
    res = clip_columns(X_clean, lower, upper)
    return res

# Step 20 - assemble_feature_matrix
import numpy as np
def assemble_feature_matrix(X_num, ratio_num_idx, ratio_den_idx, cat_labels=None):
    ratio = make_ratio_feature(X_num[:, ratio_num_idx], X_num[:, ratio_den_idx])
    new_X = append_column(X_num, ratio)
    if cat_labels is not None:
        matrix = one_hot_encode(cat_labels)
        new_X = np.hstack((new_X, matrix))
    return new_X

# Step 21 - make_train_val_test
def make_train_val_test(X, y, train_ratio, val_ratio, seed):
    result = {}

    indices = make_shuffled_indices(len(y), seed)

    train_idx, val_idx, test_idx = partition_indices(
        indices, train_ratio, val_ratio
    )

    X_sub, y_sub = subset_xy(X, y, train_idx)
    result["X_train"] = X_sub
    result["y_train"] = y_sub

    X_sub, y_sub = subset_xy(X, y, val_idx)
    result["X_val"] = X_sub
    result["y_val"] = y_sub

    X_sub, y_sub = subset_xy(X, y, test_idx)
    result["X_test"] = X_sub
    result["y_test"] = y_sub

    return result

# Step 22 - standardize_and_add_bias
def standardize_and_add_bias(splits):
    std_splits = splits.copy()
    mean, std = fit_standardizer(splits["X_train"])
    X_train = apply_standardizer(splits["X_train"], mean, std)
    X_train = add_bias_column(X_train)
    X_val = apply_standardizer(splits["X_val"], mean, std)
    X_val = add_bias_column(X_val)
    X_test = apply_standardizer(splits["X_test"], mean, std)
    X_test = add_bias_column(X_test)
    std_splits["X_train"] = X_train
    std_splits["X_val"] = X_val
    std_splits["X_test"] = X_test
    return std_splits, mean, std

# Step 23 - evaluate_predictions
def evaluate_predictions(y_true, y_pred):
    res = {}
    res["mae"] = mean_absolute_error(y_true, y_pred)
    res["rmse"] = root_mean_squared_error(y_true, y_pred)
    res["r2"] = r_squared(y_true, y_pred)
    res["residual_summary"] = residual_summary(y_true, y_pred)
    return res

# Step 24 - house_price_pipeline
def house_price_pipeline(
    X,
    y,
    ratio_num_idx,
    ratio_den_idx,
    cat_labels=None,
    train_ratio=0.7,
    val_ratio=0.15,
    seed=42,
    iqr_k=1.5,
):
    X_num = prepare_cleaned_features(X, iqr_k)
    feature_matrix = assemble_feature_matrix(
        X_num, ratio_num_idx, ratio_den_idx, cat_labels
    )
    splits = make_train_val_test(feature_matrix, y, train_ratio, val_ratio, seed)
    std_splits, mean, std = standardize_and_add_bias(splits)
    theta = ols_fit(std_splits["X_train"], std_splits["y_train"])

    y_val_pred = ols_predict(std_splits["X_val"], theta)
    y_test_pred = ols_predict(std_splits["X_test"], theta)

    val_metrics = evaluate_predictions(std_splits["y_val"], y_val_pred)
    test_metrics = evaluate_predictions(std_splits["y_test"], y_test_pred)

    return {
        "theta":theta,
        "y_test":std_splits["y_test"],
        "y_test_pred":y_test_pred,
        "test_metrics":test_metrics,
        "val_metrics":val_metrics
    }

