import numpy as np
import pandas as pd
import os

from my_ml_package.metrics.regression import (
    adjusted_r2_score,
    mae,
    mse,
    rmse,
    r2_score,
)
from my_ml_package.model_selection.train_test_split import train_test_split
from my_ml_package.neighbors.knn_regressor import KNNRegressor
from my_ml_package.preprocessing.scaler import StandardScaler


def convert_price_to_numeric(price):
    """Convert price text like '₹62 Lac' or '₹1.2 Cr' into a numeric rupee value."""
    if pd.isna(price):
        return np.nan

    text = str(price).replace("₹", "").replace(",", "").strip().lower()

    if "cr" in text:
        text = text.replace("cr", "").strip()
        return float(text) * 10000000

    if "lac" in text or "lakh" in text:
        text = text.replace("lacs", "").replace("lakh", "").replace("lac", "").strip()
        return float(text) * 100000

    return np.nan


def normalize_location(location):
    """Simplify noisy location text and keep a stable location label."""
    if pd.isna(location):
        return "Other"

    text = str(location).strip()
    if " in " in text:
        text = text.split(" in ")[-1]
    if "," in text:
        text = text.split(",")[0]

    return text.strip() or "Other"


def reduce_noisy_locations(df, top_n=5):
    """Keep only the top frequent locations and replace the rest with 'Other'."""
    location_counts = df["location"].value_counts()
    top_locations = location_counts.nlargest(top_n).index
    df["location"] = df["location"].where(df["location"].isin(top_locations), "Other")
    return df


def remove_outliers(df):
    """Remove rows with extreme numeric values using percentile filtering."""
    df = df[df["area"] > 0]
    df = df[df["price"] > 0]
    df = df[df["price_per_sqft"] > 0]

    for column in ["area", "price", "price_per_sqft"]:
        lower = df[column].quantile(0.02)
        upper = df[column].quantile(0.98)
        df = df[df[column].between(lower, upper)]

    return df


def find_better_k_values(X_train, y_train, X_test, y_test, candidates):
    """Evaluate a few k values and return the best performing k on test R2."""
    scores = {}
    for k in candidates:
        model = KNNRegressor(k=k)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        scores[k] = r2_score(y_test, y_pred)

    sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return sorted_scores


def main():
    # Load the dataset using pandas (fallback if filename changed)
    data_path = "data/indore_house_data.csv"
    if not os.path.exists(data_path):
        data_path = "data/magicbricks_data.csv"
    df = pd.read_csv(data_path)

    # Keep only the columns required for this KNN pipeline.
    df = df[["location", "property_type", "area", "bhk", "furnishing", "price"]].copy()

    # Convert string prices like '₹62 Lac' and '₹1.2 Cr' into numeric rupee values.
    df["price"] = df["price"].apply(convert_price_to_numeric)

    # Drop exact duplicate rows to avoid repeated examples in training.
    df.drop_duplicates(inplace=True)

    # Convert numeric columns and fill missing numeric values with median values.
    numeric_columns = ["area", "bhk", "price"]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
        df[column] = df[column].fillna(df[column].median())

    # Replace missing categorical values with the most common category.
    categorical_columns = ["location", "property_type", "furnishing"]
    for column in categorical_columns:
        default_value = df[column].mode().iloc[0] if not df[column].mode().empty else "Other"
        df[column] = df[column].fillna(default_value)

    # Normalize location text to reduce noisy variants.
    df["location"] = df["location"].apply(normalize_location)

    # Create price per square foot feature from price and area.
    df["price_per_sqft"] = df["price"] / df["area"]

    # Replace infinite values and drop rows with missing or invalid numeric values.
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=["price", "area", "bhk", "price_per_sqft"], inplace=True)

    # Reduce rare locations and remove extreme outliers.
    df = reduce_noisy_locations(df, top_n=5)
    df = remove_outliers(df)

    # One-hot encode categorical columns with pandas get_dummies.
    df = pd.get_dummies(df, columns=["location", "property_type", "furnishing"], drop_first=True)

    # Choose the features for the KNN model.
    feature_columns = ["area", "bhk", "price_per_sqft"]
    feature_columns += [col for col in df.columns if col.startswith("location_") or col.startswith("property_type_") or col.startswith("furnishing_")]

    X = df[feature_columns].astype(float).values
    y = np.log1p(df["price"].astype(float).values)

    # Split the dataset into training and testing sets using the custom train_test_split.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features using the custom StandardScaler implementation.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Evaluate candidate k values and suggest better options.
    candidate_ks = [3, 5, 7, 9]
    scores = find_better_k_values(X_train, y_train, X_test, y_test, candidate_ks)
    best_ks = ", ".join([f"k={k} ({score:.4f})" for k, score in scores])

    # Train the final KNN regressor with k=5 as requested.
    model = KNNRegressor(k=5)
    model.fit(X_train, y_train)
    y_pred_log = model.predict(X_test)

    # Convert the predicted log values back to rupees for readable predictions.
    y_pred = np.expm1(y_pred_log)
    y_test_actual = np.expm1(y_test)

    # Print the first few predictions alongside actual prices.
    print("First 10 predicted prices (rupees) vs actual prices:")
    for actual, predicted in list(zip(y_test_actual[:10], y_pred[:10])):
        print(f"Actual: {actual:,.0f}, Predicted: {predicted:,.0f}")

    # Print evaluation metrics using the log-transformed target values.
    print("\nModel performance on log(price):")
    print(f"R2 Score: {r2_score(y_test, y_pred_log):.4f}")

    # Adjusted R2 requires more test examples than features.
    if len(y_test) > X_test.shape[1] + 1:
        adjusted = adjusted_r2_score(y_test, y_pred_log, n_features=X_test.shape[1])
        print(f"Adjusted R2: {adjusted:.4f}")
    else:
        print("Adjusted R2: not available for this test size")

    print(f"MSE: {mse(y_test, y_pred_log):.4f}")
    print(f"MAE: {mae(y_test, y_pred_log):.4f}")
    print(f"RMSE: {rmse(y_test, y_pred_log):.4f}")

    print("\nSuggested k values to improve R2 score:")
    print(best_ks)

    print(f"\nTraining rows used: {df.shape[0]}")
    print(f"Feature count: {X.shape[1]}")


if __name__ == "__main__":
    main()
