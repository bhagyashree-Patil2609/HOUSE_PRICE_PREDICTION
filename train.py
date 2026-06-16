# Import the libraries we need for data handling and numeric operations
import pandas as pd
import numpy as np
from pathlib import Path

# Import only the custom modules from your package
from my_ml_package.linear_models.linear_regression import LinearRegression
from my_ml_package.metrics.regression import (
    adjusted_r2_score,
    mae,
    mse,
    rmse,
    r2_score,
)
from my_ml_package.model_selection.train_test_split import train_test_split
from my_ml_package.preprocessing.scaler import StandardScaler


# Convert price text like '₹62 Lac' and '₹1.20 Cr' into numeric rupee values
def convert_price_to_numeric(price):
    if pd.isna(price):
        return np.nan

    text = str(price).replace("₹", "").replace(",", "").strip()

    if "Cr" in text:
        value = float(text.replace("Cr", "").strip())
        return value * 10000000

    if "Lac" in text or "Lakh" in text:
        value = float(text.replace("Lac", "").replace("Lakh", "").strip())
        return value * 100000

    return np.nan


# Simplify location strings and keep only the core location name
def normalize_location(location):
    if pd.isna(location):
        return "Other"

    text = str(location).strip()

    if " in " in text:
        # Keep the last part after the word 'in'
        text = text.split(" in ")[-1]

    if "," in text:
        # Keep only the first comma-separated token
        text = text.split(",")[0]

    return text.strip() or "Other"


def reduce_noisy_locations(df, min_count=30):
    # Replace locations with few records by 'Other' to reduce noise
    location_counts = df["location"].value_counts()
    frequent_locations = location_counts[location_counts >= min_count].index
    df["location"] = df["location"].where(df["location"].isin(frequent_locations), "Other")
    return df


def remove_outliers(df):
    # Remove invalid or extremely small values first
    df = df[df["area"] > 0]
    df = df[df["price"] > 0]
    df = df[df["price_per_sqft"] > 0]

    # Remove the top and bottom 2% of values in each numeric column
    for column in ["area", "price", "price_per_sqft"]:
        lower = df[column].quantile(0.02)
        upper = df[column].quantile(0.98)
        df = df[df[column].between(lower, upper)]

    return df


def main():
    # Load the dataset from the CSV file
    data_path = Path("data")
    default_file = data_path / "indore_house_data.csv"
    alternate_file = data_path / "magicbricks_data.csv"

    if default_file.exists():
        csv_path = default_file
    elif alternate_file.exists():
        csv_path = alternate_file
    else:
        raise FileNotFoundError(
            f"Dataset not found. Expected either {default_file} or {alternate_file}."
        )

    df = pd.read_csv(csv_path)

    # Keep only the columns that will be useful for modeling
    df = df[[
        "location",
        "property_type",
        "area",
        "bhk",
        "bedrooms",
        "bathrooms",
        "balcony",
        "furnishing",
        "price",
    ]].copy()

    # Convert the price strings into numeric values
    df["price"] = df["price"].apply(convert_price_to_numeric)

    # Convert numeric columns and fill missing values with median values
    numeric_columns = ["area", "bhk", "bedrooms", "bathrooms", "balcony"]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
        median_value = df[column].median()
        if np.isnan(median_value):
            median_value = 0
        df[column] = df[column].fillna(median_value)

    # Fill missing categorical values with the most common category
    categorical_columns = ["location", "property_type", "furnishing"]
    for column in categorical_columns:
        default_value = df[column].mode().iloc[0] if not df[column].mode().empty else "Other"
        df[column] = df[column].fillna(default_value)

    # Normalize location text to reduce noise
    df["location"] = df["location"].apply(normalize_location)

    # Remove any rows where price could not be converted
    df = df.dropna(subset=["price"])

    # Add a new feature for price per square foot
    df["price_per_sqft"] = df["price"] / df["area"]

    # Replace infinite values and remove incomplete rows
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.dropna(subset=["price", "area", "price_per_sqft"])
    df = df.drop_duplicates()

    # Reduce rare location values and remove outliers
    df = reduce_noisy_locations(df, min_count=30)
    df = remove_outliers(df)

    # Create extra features to help the model
    safe_bhk = df["bhk"].replace(0, 1)
    df["area_per_bhk"] = df["area"] / safe_bhk
    df["bathrooms_per_bhk"] = df["bathrooms"] / safe_bhk

    # Encode categorical columns using one-hot encoding
    df = pd.get_dummies(df, columns=["location", "property_type", "furnishing"], drop_first=True)
    
    # Save the final dataset after one-hot encoding
    df.to_csv("18_features_dataset.csv", index=False)
    print("18-feature dataset saved successfully")

    # Select the final set of features for modeling
    feature_columns = [
        "area",
        "bhk",
        "bedrooms",
        "bathrooms",
        "balcony",
        "price_per_sqft",
        "area_per_bhk",
        "bathrooms_per_bhk",
    ]
    dummy_columns = [
        col
        for col in df.columns
        if col.startswith("location_")
        or col.startswith("property_type_")
        or col.startswith("furnishing_")
    ]
    feature_columns += dummy_columns

    # Prepare feature matrix X and target vector y
    X = df[feature_columns].astype(float).values
    y = np.log1p(df["price"].values)

    # Split the data into training and test sets with a fixed random seed
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale features using the custom StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train the custom linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("Intercept:", model.intercept_)

    print("Coefficients:", model.coef_)

    # Print the evaluation metrics to see model performance
    print("Data used for modeling:", df.shape[0], "rows")
    print("Feature count:", X.shape[1])
    print("Model evaluation on log-transformed price:")
    print(f"R2 Score: {r2_score(y_test, y_pred):.4f}")
    print(f"Adjusted R2: {adjusted_r2_score(y_test, y_pred, n_features=X_test.shape[1]):.4f}")
    print(f"MSE: {mse(y_test, y_pred):.4f}")
    print(f"MAE: {mae(y_test, y_pred):.4f}")
    print(f"RMSE: {rmse(y_test, y_pred):.4f}")


if __name__ == "__main__":
    main()
