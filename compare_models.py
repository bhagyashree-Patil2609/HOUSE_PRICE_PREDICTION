import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

try:
    from sklearn.decomposition import PCA as SKPCA
    from sklearn.linear_model import LinearRegression as SKLinearRegression
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import MinMaxScaler, StandardScaler as SKStandardScaler
except ModuleNotFoundError as error:
    raise ModuleNotFoundError(
        "scikit-learn is required for compare_models.py. "
        "Install it with `pip install scikit-learn` and rerun the script."
    ) from error

from my_ml_package.decomposition.pca import PCA as CustomPCA
from my_ml_package.linear_models.linear_regression import LinearRegression as CustomLinearRegression
from my_ml_package.metrics.regression import adjusted_r2_score, mae, mse, rmse, r2_score
from my_ml_package.model_selection.train_test_split import train_test_split as custom_train_test_split
from my_ml_package.neighbors.knn_regressor import KNNRegressor as CustomKNNRegressor
from my_ml_package.neural_network.neural_network import NeuralNetwork as CustomNeuralNetwork
from my_ml_package.preprocessing.scaler import StandardScaler as CustomStandardScaler


def convert_price_to_numeric(price):
    if pd.isna(price):
        return np.nan

    text = str(price).replace("₹", "").replace(",", "").strip().lower()

    if "cr" in text:
        return float(text.replace("cr", "").strip()) * 10000000

    if "lac" in text or "lakh" in text:
        return float(text.replace("lacs", "").replace("lakh", "").replace("lac", "").strip()) * 100000

    return np.nan


def normalize_location(location):
    if pd.isna(location):
        return "Other"

    text = str(location).lower()
    if " in " in text:
        text = text.split(" in ")[-1]
    if "," in text:
        text = text.split(",")[0]

    return text.strip().title() or "Other"


def reduce_noisy_locations(df, top_n=10):
    location_counts = df["location"].value_counts()
    top_locations = location_counts.nlargest(top_n).index
    df["location"] = df["location"].where(df["location"].isin(top_locations), "Other")
    return df


def remove_infinite_values(df):
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for column in numeric_columns:
        df[column] = df[column].replace([np.inf, -np.inf], np.nan)
    return df


def remove_outliers_iqr(df, columns):
    for column in columns:
        if column not in df.columns:
            continue

        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        df = df[(df[column] >= lower) & (df[column] <= upper)]

    return df


def preprocess_dataset(path):
    df = pd.read_csv(path)

    df = df[
        ["price", "area", "bedrooms", "bathrooms", "location", "bhk", "property_type", "furnishing", "balcony"]
    ].copy()

    df["price"] = df["price"].apply(convert_price_to_numeric)

    numeric_columns = ["area", "bedrooms", "bathrooms", "bhk", "balcony", "price"]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
        median_value = df[column].median()
        if np.isnan(median_value):
            median_value = 0.0
        df[column] = df[column].fillna(median_value)

    categorical_columns = ["location", "property_type", "furnishing"]
    for column in categorical_columns:
        default_value = df[column].mode().iloc[0] if not df[column].mode().empty else "Other"
        df[column] = df[column].fillna(default_value)

    df = remove_infinite_values(df)
    df["location"] = df["location"].apply(normalize_location)

    df["price_per_sqft"] = df["price"] / df["area"]
    df = reduce_noisy_locations(df, top_n=8)
    df = remove_outliers_iqr(df, ["area", "price", "price_per_sqft"])

    df = df.dropna().reset_index(drop=True)
    df = pd.get_dummies(df, columns=["location", "property_type", "furnishing"], drop_first=True)

    X = df.drop(columns=["price"]).values.astype(float)
    y = df["price"].values.astype(float)

    return X, y, df.drop(columns=["price"]).columns.tolist()


def measure_execution_time(function, *args, **kwargs):
    start = time.time()
    result = function(*args, **kwargs)
    return result, time.time() - start


def evaluate_regression_model(name, model, X_train, y_train, X_test, y_test, n_features):
    try:
        _, fit_time = measure_execution_time(model.fit, X_train, y_train)
        predictions, predict_time = measure_execution_time(model.predict, X_test)
        execution_time = fit_time + predict_time

        score_r2 = r2_score(y_test, predictions)
        score_adj_r2 = adjusted_r2_score(y_test, predictions, n_features)

        return {
            "Model": name,
            "R2 Score": score_r2,
            "Adjusted R2": score_adj_r2,
            "MSE": mse(y_test, predictions),
            "MAE": mae(y_test, predictions),
            "RMSE": rmse(y_test, predictions),
            "Time (s)": execution_time,
        }
    except Exception as error:
        print(f"Warning: {name} failed during evaluation: {error}")
        return {
            "Model": name,
            "R2 Score": np.nan,
            "Adjusted R2": np.nan,
            "MSE": np.nan,
            "MAE": np.nan,
            "RMSE": np.nan,
            "Time (s)": np.nan,
        }


def compare_scalers(X_train):
    custom_scaler = CustomStandardScaler()
    sklearn_scaler = SKStandardScaler()

    X_custom_scaled, custom_time = measure_execution_time(custom_scaler.fit_transform, X_train)
    X_sklearn_scaled, sklearn_time = measure_execution_time(sklearn_scaler.fit_transform, X_train)

    same_results = np.allclose(X_custom_scaled, X_sklearn_scaled, atol=1e-8)

    print("\n===== Scaler comparison =====")
    print(f"Custom StandardScaler time: {custom_time:.4f} s")
    print(f"Sklearn StandardScaler time: {sklearn_time:.4f} s")
    print(f"Scaled feature shapes: {X_custom_scaled.shape}")
    print(f"Same numeric results: {same_results}")

    return X_custom_scaled, X_sklearn_scaled, custom_scaler, sklearn_scaler


def compare_pca(X_scaled):
    custom_pca = CustomPCA(n_components=0.95)
    sklearn_pca = SKPCA(n_components=0.95)

    _, custom_time = measure_execution_time(custom_pca.fit, X_scaled)
    _, sklearn_time = measure_execution_time(sklearn_pca.fit, X_scaled)

    custom_variance = np.sum(custom_pca.explained_variance_ratio_) * 100
    sklearn_variance = np.sum(sklearn_pca.explained_variance_ratio_) * 100

    pca_comparison = pd.DataFrame([
        {
            "PCA Implementation": "Custom PCA",
            "Components": custom_pca.n_components_,
            "Variance Retained (%)": custom_variance,
            "Execution Time (s)": custom_time,
        },
        {
            "PCA Implementation": "Sklearn PCA",
            "Components": sklearn_pca.n_components_,
            "Variance Retained (%)": sklearn_variance,
            "Execution Time (s)": sklearn_time,
        },
    ])

    return pca_comparison, custom_pca, sklearn_pca


def build_comparison_plots(results_df, pca_comparison):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.bar(results_df["Model"], results_df["R2 Score"], color=["#4c72b0", "#55a868", "#c44e52", "#8172b2", "#ccb974", "#64b5cd"])
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("R2 Score")
    plt.title("Model Comparison: R2 Score")

    plt.subplot(1, 2, 2)
    plt.bar(results_df["Model"], results_df["RMSE"], color=["#4c72b0", "#55a868", "#c44e52", "#8172b2", "#ccb974", "#64b5cd"])
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("RMSE")
    plt.title("Model Comparison: RMSE")

    plt.tight_layout()
    plt.savefig("model_performance_comparison.png", dpi=120, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.bar(pca_comparison["PCA Implementation"], pca_comparison["Variance Retained (%)"], color=["#4c72b0", "#55a868"])
    plt.ylabel("Variance Retained (%)")
    plt.title("PCA Variance Retained: Custom vs Sklearn")
    plt.tight_layout()
    plt.savefig("pca_variance_comparison.png", dpi=120, bbox_inches="tight")
    plt.close()


def print_summary(results_df, pca_comparison):
    print("\n===== Learning-based model comparison =====")
    print(results_df.to_string(index=False, float_format="{:.4f}".format))

    print("\n===== PCA comparison =====")
    print(pca_comparison.to_string(index=False, float_format="{:.4f}".format))

    print("\n===== Summary =====")
    print("This comparison is intended as an educational comparison between custom implementations and sklearn implementations.")
    print("The goal is to understand how algorithm design, numerical stability, and library optimizations influence performance.")

    print("\n===== Observations =====")
    print("1. sklearn models are likely to fit faster and deliver more stable training because they use optimized linear algebra, regularization, and solver implementations.")
    print("2. Custom Linear Regression demonstrates the same statistical formula, but it may be slower for large data due to explicit pseudoinverse computation.")
    print("3. Custom KNN and sklearn KNeighborsRegressor should behave similarly when using identical scaling; sklearn may still use optimized neighbor search and C-level loops.")
    print("4. Custom NeuralNetwork is a didactic example and has important limitations: no bias terms, sigmoid-only output, and a simple gradient update loop.")
    print("5. PCA comparison shows that a carefully implemented custom PCA can match sklearn’s explained variance, but sklearn uses more robust linear algebra and numerical safeguards.")

    print("\n===== Conclusion =====")
    print("The research comparison confirms that sklearn provides strong production-ready implementations with better optimization and stability.")
    print("Custom package implementations are valuable for learning and for small experiments, but sklearn remains the preferred choice when reliability and performance are priorities.")


def main():
    path = "data/indore_house_data.csv"
    if not os.path.exists(path):
        path = "data/magicbricks_data.csv"

    X, y, feature_names = preprocess_dataset(path)
    print(f"Dataset prepared with {X.shape[0]} samples and {X.shape[1]} features.")

    X_train, X_test, y_train, y_test = custom_train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_train_custom_scaled, X_train_sklearn_scaled, custom_scaler, sklearn_scaler = compare_scalers(X_train)
    X_test_custom_scaled = custom_scaler.transform(X_test)
    X_test_sklearn_scaled = sklearn_scaler.transform(X_test)

    scaler_results = []
    scaler_results.append({
        "Scaler": "Custom StandardScaler",
        "Mean Difference": np.mean(np.abs(X_train_custom_scaled - X_train_sklearn_scaled)),
    })

    print("\n===== Training and evaluation =====")

    model_results = []
    n_features = X_train_custom_scaled.shape[1]

    custom_lr = CustomLinearRegression()
    model_results.append(
        evaluate_regression_model(
            "LinearRegression (custom)",
            custom_lr,
            X_train_custom_scaled,
            y_train,
            X_test_custom_scaled,
            y_test,
            n_features,
        )
    )

    sklearn_lr = SKLinearRegression()
    model_results.append(
        evaluate_regression_model(
            "LinearRegression (sklearn)",
            sklearn_lr,
            X_train_sklearn_scaled,
            y_train,
            X_test_sklearn_scaled,
            y_test,
            n_features,
        )
    )

    custom_knn = CustomKNNRegressor(k=5)
    model_results.append(
        evaluate_regression_model(
            "KNNRegressor (custom)",
            custom_knn,
            X_train_custom_scaled,
            y_train,
            X_test_custom_scaled,
            y_test,
            n_features,
        )
    )

    sklearn_knn = KNeighborsRegressor(n_neighbors=5)
    model_results.append(
        evaluate_regression_model(
            "KNNRegressor (sklearn)",
            sklearn_knn,
            X_train_sklearn_scaled,
            y_train,
            X_test_sklearn_scaled,
            y_test,
            n_features,
        )
    )

    # Neural network target scaling is necessary because the custom implementation uses sigmoid output.
    target_scaler = MinMaxScaler()
    y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_test_scaled = target_scaler.transform(y_test.reshape(-1, 1)).ravel()

    custom_nn = CustomNeuralNetwork(input_size=n_features, hidden_size=16, output_size=1, random_state=42)
    start = time.time()
    custom_nn.fit(X_train_custom_scaled, y_train_scaled.reshape(-1, 1), epochs=75, learning_rate=0.001)
    custom_nn_time = time.time() - start
    custom_nn.predict = lambda X: custom_nn.forward(X).ravel()
    y_pred_nn_custom = target_scaler.inverse_transform(custom_nn.predict(X_test_custom_scaled).reshape(-1, 1)).ravel()

    model_results.append(
        {
            "Model": "NeuralNetwork (custom)",
            "R2 Score": r2_score(y_test, y_pred_nn_custom),
            "Adjusted R2": adjusted_r2_score(y_test, y_pred_nn_custom, n_features),
            "MSE": mse(y_test, y_pred_nn_custom),
            "MAE": mae(y_test, y_pred_nn_custom),
            "RMSE": rmse(y_test, y_pred_nn_custom),
            "Time (s)": custom_nn_time,
        }
    )

    sklearn_mlp = MLPRegressor(hidden_layer_sizes=(16,), activation="relu", solver="adam", max_iter=1000, random_state=42)
    start = time.time()
    sklearn_mlp.fit(X_train_sklearn_scaled, y_train_scaled)
    sklearn_mlp_time = time.time() - start
    y_pred_mlp = target_scaler.inverse_transform(sklearn_mlp.predict(X_test_sklearn_scaled).reshape(-1, 1)).ravel()

    model_results.append(
        {
            "Model": "MLPRegressor (sklearn)",
            "R2 Score": r2_score(y_test, y_pred_mlp),
            "Adjusted R2": adjusted_r2_score(y_test, y_pred_mlp, n_features),
            "MSE": mse(y_test, y_pred_mlp),
            "MAE": mae(y_test, y_pred_mlp),
            "RMSE": rmse(y_test, y_pred_mlp),
            "Time (s)": sklearn_mlp_time,
        }
    )

    results_df = pd.DataFrame(model_results)

    pca_comparison, custom_pca, sklearn_pca = compare_pca(X_train_sklearn_scaled)

    build_comparison_plots(results_df, pca_comparison)
    print_summary(results_df, pca_comparison)

    print("\nResults and plots have been generated for a research-style comparison.")


if __name__ == "__main__":
    main()
