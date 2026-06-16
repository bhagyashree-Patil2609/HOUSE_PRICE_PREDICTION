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
from my_ml_package.neural_network.neural_network import NeuralNetwork
from my_ml_package.preprocessing.scaler import StandardScaler


def convert_price_to_numeric(price):
    """Convert price strings like '₹62 Lac' and '₹1.2 Cr' into numeric rupees."""
    if pd.isna(price):
        return np.nan

    text = str(price).replace("₹", "").replace(",", "").strip().lower()

    if "cr" in text:
        return float(text.replace("cr", "").strip()) * 10000000

    if "lac" in text or "lakh" in text:
        return float(text.replace("lacs", "").replace("lakh", "").replace("lac", "").strip()) * 100000

    return np.nan


def normalize_location(location):
    """Simplify location values to reduce noisy text variations."""
    if pd.isna(location):
        return "Other"

    text = str(location).strip()
    if " in " in text:
        text = text.split(" in ")[-1]
    if "," in text:
        text = text.split(",")[0]

    return text.strip() or "Other"


def reduce_noisy_locations(df, top_n=10):
    """Keep only the most frequent locations and group the rest as 'Other'."""
    location_counts = df["location"].value_counts()
    top_locations = location_counts.nlargest(top_n).index
    df["location"] = df["location"].where(df["location"].isin(top_locations), "Other")
    return df


def remove_outliers(df):
    """Remove extreme numeric values using a simple percentile filter."""
    df = df[df["area"] > 0]
    df = df[df["price"] > 0]
    df = df[df["price_per_sqft"] > 0]

    for column in ["area", "price", "price_per_sqft"]:
        lower = df[column].quantile(0.02)
        upper = df[column].quantile(0.98)
        df = df[df[column].between(lower, upper)]

    return df


def main():
    # Load data from the CSV file (fallback if filename changed).
    data_path = "data/indore_house_data.csv"
    if not os.path.exists(data_path):
        data_path = "data/magicbricks_data.csv"
    df = pd.read_csv(data_path)

    # Keep only the requested columns for the ANN pipeline.
    df = df[["location", "property_type", "area", "bhk", "furnishing", "price"]].copy()

    # Convert the price column from strings to numeric rupee values.
    df["price"] = df["price"].apply(convert_price_to_numeric)

    # Remove exact duplicate rows to avoid repeated samples.
    df.drop_duplicates(inplace=True)

    # Convert numeric columns and fill missing values with the median.
    numeric_columns = ["area", "bhk", "price"]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
        df[column] = df[column].fillna(df[column].median())

    # Fill missing categorical values with the most common category.
    categorical_columns = ["location", "property_type", "furnishing"]
    for column in categorical_columns:
        default_value = df[column].mode().iloc[0] if not df[column].mode().empty else "Other"
        df[column] = df[column].fillna(default_value)

    # Normalize location text so similar locations are grouped.
    df["location"] = df["location"].apply(normalize_location)

    # Drop rows that still do not have valid price values.
    df = df.dropna(subset=["price"])

    # Create a new feature for price per square foot.
    df["price_per_sqft"] = df["price"] / df["area"]

    # Replace infinite values and drop any rows that are still invalid.
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.dropna(subset=["price", "area", "price_per_sqft"])

    # Reduce noisy locations by keeping only the top frequent locations.
    df = reduce_noisy_locations(df, top_n=10)

    # Remove outliers to keep the ANN training stable.
    df = remove_outliers(df)

    # Encode categorical columns using pandas one-hot encoding.
    df = pd.get_dummies(df, columns=["location", "property_type", "furnishing"], drop_first=True)

    # Choose the final set of features for the neural network.
    feature_columns = ["area", "bhk", "price_per_sqft"]
    feature_columns += [
        col
        for col in df.columns
        if col.startswith("location_") or col.startswith("property_type_") or col.startswith("furnishing_")
    ]

    X = df[feature_columns].astype(float).values

    # Apply log transform on price to make regression easier.
    y_log = np.log1p(df["price"].astype(float).values)

    # Scale the log target to the 0-1 range because the custom NeuralNetwork uses sigmoid output.
    y_min = y_log.min()
    y_max = y_log.max()
    y_scaled = (y_log - y_min) / (y_max - y_min)

    # Split the data into training and testing sets using the custom train_test_split.
    X_train, X_test, y_train, y_test = train_test_split(X, y_scaled, test_size=0.2, random_state=42)

    # Scale features using the custom StandardScaler implementation.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Configure the ANN with the appropriate sizes.
    input_size = X_train.shape[1]
    hidden_size = max(8, input_size * 2)
    output_size = 1

    model = NeuralNetwork(input_size, hidden_size, output_size, random_state=42)

    # Train the model with a learning rate and number of epochs.
    model.fit(X_train, y_train.reshape(-1, 1), epochs=1000, learning_rate=0.001)

    # Use the raw forward output for regression instead of the default rounded predict().
    y_pred_scaled = model.forward(X_test).flatten()
    y_pred_scaled = np.clip(y_pred_scaled, 0.0, 1.0)

    # Convert the scaled predictions back to log(price) and then to actual price.
    y_pred_log = y_pred_scaled * (y_max - y_min) + y_min
    y_test_log = y_test * (y_max - y_min) + y_min
    y_pred = np.expm1(y_pred_log)
    y_test_actual = np.expm1(y_test_log)

    # Print the first few predictions and actual prices.
    print("First 10 predicted prices vs actual prices:")
    for actual, predicted in list(zip(y_test_actual[:10], y_pred[:10])):
        print(f"Actual: {actual:,.0f}, Predicted: {predicted:,.0f}")

    # Print regression metrics using the custom package.
    print("\nANN regression performance on actual prices:")
    print(f"R2 Score: {r2_score(y_test_actual, y_pred):.4f}")

    if len(y_test_actual) > input_size + 1:
        print(f"Adjusted R2: {adjusted_r2_score(y_test_actual, y_pred, n_features=input_size):.4f}")
    else:
        print("Adjusted R2: not available for this test size")

    print(f"MSE: {mse(y_test_actual, y_pred):.4f}")
    print(f"MAE: {mae(y_test_actual, y_pred):.4f}")
    print(f"RMSE: {rmse(y_test_actual, y_pred):.4f}")

    


if __name__ == "__main__":
    main()
