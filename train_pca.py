import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from my_ml_package.decomposition.pca import PCA
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


def remove_infinite_values(df):
    """Replace infinite values with NaN and handle them."""
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for column in numeric_columns:
        df[column] = df[column].replace([np.inf, -np.inf], np.nan)
    return df


def main():
    # ========== STEP 1: LOAD DATA ==========
    # Load the Indore house price dataset from CSV file
    print("=" * 60)
    print("Loading dataset...")
    print("=" * 60)
    
    data_path = "data/indore_house_data.csv"
    if not os.path.exists(data_path):
        data_path = "data/magicbricks_data.csv"
    df = pd.read_csv(data_path)
    print(f"Original dataset shape: {df.shape}")

    # ========== STEP 2: SELECT REQUIRED COLUMNS ==========
    # Select only the columns we need for the PCA analysis
    df = df[["location", "property_type", "area", "bhk", "furnishing", "price"]].copy()
    print(f"After column selection: {df.shape}")

    # ========== STEP 3: CONVERT PRICE STRINGS TO NUMERIC ==========
    # Convert price from currency format (₹62 Lac, ₹1.2 Cr) to numeric rupees
    print("\nConverting prices to numeric format...")
    df["price"] = df["price"].apply(convert_price_to_numeric)

    # ========== STEP 4: HANDLE DUPLICATES ==========
    # Remove exact duplicate rows to avoid data leakage
    print(f"Rows before removing duplicates: {len(df)}")
    df.drop_duplicates(inplace=True)
    print(f"Rows after removing duplicates: {len(df)}")

    # ========== STEP 5: HANDLE MISSING AND INFINITE VALUES ==========
    # Convert numeric columns and handle NaN values with median imputation
    print("\nHandling missing values...")
    numeric_columns = ["area", "bhk", "price"]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
        missing_count = df[column].isna().sum()
        if missing_count > 0:
            print(f"  {column}: {missing_count} missing values -> filled with median")
            df[column] = df[column].fillna(df[column].median())

    # Handle infinite values
    df = remove_infinite_values(df)

    # Fill missing categorical values with the most common category
    categorical_columns = ["location", "property_type", "furnishing"]
    for column in categorical_columns:
        default_value = df[column].mode().iloc[0] if not df[column].mode().empty else "Other"
        df[column] = df[column].fillna(default_value)

    # Drop rows that still don't have valid price values
    df = df.dropna(subset=["price"])
    print(f"Rows after cleaning: {len(df)}")

    # ========== STEP 6: NORMALIZE LOCATION NAMES ==========
    # Simplify location names to reduce noise and group similar locations
    print("\nNormalizing location names...")
    df["location"] = df["location"].apply(normalize_location)

    # ========== STEP 7: CREATE NEW FEATURE ==========
    # Create price per square foot feature (useful ratio for real estate)
    print("Creating price_per_sqft feature...")
    df["price_per_sqft"] = df["price"] / df["area"]

    # ========== STEP 8: REDUCE NOISY LOCATIONS ==========
    # Keep only top 10 frequent locations to reduce noise
    print("Reducing noisy locations...")
    df = reduce_noisy_locations(df, top_n=10)
    print(f"Unique locations after filtering: {df['location'].nunique()}")

    # ========== STEP 9: ENCODE CATEGORICAL COLUMNS ==========
    """
    DIMENSIONALITY REDUCTION - ENCODING STEP:
    
    Before we can apply PCA, we need to convert categorical columns (location, property_type, 
    furnishing) into numeric format. We use one-hot encoding (pandas get_dummies) which creates 
    binary columns for each category.
    
    This increases the number of features (columns) temporarily, which is why dimensionality 
    reduction becomes important!
    """
    print("\n" + "=" * 60)
    print("Encoding categorical variables...")
    print("=" * 60)
    
    # Apply one-hot encoding to convert categorical columns to numeric
    df = pd.get_dummies(df, columns=["location", "property_type", "furnishing"], drop_first=True)
    print(f"Features after one-hot encoding: {df.shape[1]}")
    print(f"Feature count details: {df.shape}")

    # Separate features and display data info
    X = df.values.astype(np.float64)
    print(f"\nOriginal features shape: {X.shape}")
    print(f"  - Samples (rows): {X.shape[0]}")
    print(f"  - Features (columns): {X.shape[1]}")

    # ========== STEP 10: STANDARDIZE FEATURES ==========
    """
    IMPORTANCE OF SCALING FOR PCA:
    
    PCA finds directions of maximum variance in the data. If features have different scales 
    (e.g., price in millions vs area in hundreds), features with larger scale will dominate 
    the principal components. Standardization ensures all features contribute equally.
    
    StandardScaler transforms each feature to have:
      - Mean = 0
      - Standard Deviation = 1
    """
    print("\n" + "=" * 60)
    print("Standardizing features (scaling)...")
    print("=" * 60)
    
    # Initialize and apply StandardScaler from our custom ML package
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"Scaled features shape: {X_scaled.shape}")
    print(f"  - Scaled mean (should be ~0): {np.mean(X_scaled, axis=0)[:3]}")
    print(f"  - Scaled std (should be ~1): {np.std(X_scaled, axis=0)[:3]}")

    # ========== STEP 11: APPLY PCA FOR DIMENSIONALITY REDUCTION ==========
    """
    DIMENSIONALITY REDUCTION WITH PCA:
    
    We currently have {X.shape[1]} features after one-hot encoding. PCA (Principal Component Analysis) 
    finds new directions (principal components) where the data has the most variance. By reducing 
    to 2 components, we:
    
    1. REDUCE DIMENSIONALITY: Compress {X.shape[1]} features → 2 components
    2. REMOVE NOISE: Keeps only the most important variance patterns
    3. IMPROVE SPEED: Fewer features = faster model training
    4. REDUCE MULTICOLLINEARITY: Principal components are uncorrelated by design
    5. ENABLE VISUALIZATION: 2D data can be plotted and visualized
    
    HOW PCA WORKS:
    - Computes covariance matrix to understand feature relationships
    - Finds eigenvectors (principal component directions)
    - Ranks them by eigenvalues (importance/variance explained)
    - Projects data onto top principal components
    """
    print("\n" + "=" * 60)
    print("Applying PCA for dimensionality reduction...")
    print("=" * 60)
    
    # Initialize PCA to preserve 95% of the variance.
    # This lets PCA choose the minimum number of components required
    # instead of forcing only 2 components.
    pca = PCA(n_components=0.95)

    # Fit the PCA model and transform the data
    X_transformed = pca.fit_transform(X_scaled)

    cumulative_variance = np.sum(pca.explained_variance_ratio_[:pca.n_components_]) * 100
    print(f"Transformed features shape: {X_transformed.shape}")
    print(f"  - Number of principal components used: {pca.n_components_}")
    print(f"  - Variance retained: {cumulative_variance:.2f}%")
    print(f"  - Samples (rows): {X_transformed.shape[0]} (unchanged)")
    print(f"  - Components (columns): {X_transformed.shape[1]} (reduced from {X.shape[1]})")
    print(f"  - Dimensionality reduction: {X.shape[1]} → {X_transformed.shape[1]} components")
    print(f"\n✓ Successfully reduced {X.shape[1]} features to {X_transformed.shape[1]} components!")

    # ========== STEP 12: CALCULATE VARIANCE EXPLAINED ==========
    """
    EXPLAINED VARIANCE:
    
    The explained variance ratio tells us how much of the original data's information 
    is retained in each principal component.
    """
    print("\n" + "=" * 60)
    print("Calculating variance explained by selected components...")
    print("=" * 60)

    explained_variance = pca.explained_variance_ratio_[:min(2, pca.n_components_)]
    cumulative_variance_top = np.cumsum(explained_variance)
    total_retained = np.sum(pca.explained_variance_ratio_[:pca.n_components_]) * 100

    print(f"\nVariance explained by the first {len(explained_variance)} component(s):")
    for i, (var, cum_var) in enumerate(zip(explained_variance, cumulative_variance_top)):
        print(f"  PC{i+1}: {var*100:.2f}% (Cumulative: {cum_var*100:.2f}%)")

    if pca.n_components_ > 2:
        print(f"\nTotal variance retained in {pca.n_components_} components: {total_retained:.2f}%")
    else:
        print(f"\nTotal variance retained in {pca.n_components_} component(s): {total_retained:.2f}%")

    print(f"Variance lost: {(100 - total_retained):.2f}%")

    # ========== STEP 13: VISUALIZE PCA RESULTS ==========
    """
    VISUALIZATION:
    
    A 2D scatter plot of the first two principal components helps us understand the 
    data structure and identify patterns or clusters.
    """
    print("\n" + "=" * 60)
    print("Visualizing PCA results...")
    print("=" * 60)
    
    plt.figure(figsize=(10, 7))
    
    # Create scatter plot with color gradient based on samples
    scatter = plt.scatter(
        X_transformed[:, 0],
        X_transformed[:, 1],
        c=range(len(X_transformed)),
        cmap="viridis",
        alpha=0.6,
        s=50,
        edgecolors="k",
        linewidth=0.5
    )
    
    plt.xlabel(f"Principal Component 1 ({explained_variance[0]*100:.2f}%)", fontsize=12)
    plt.ylabel(f"Principal Component 2 ({explained_variance[1]*100:.2f}%)", fontsize=12)
    plt.title(
        f"PCA: 2D Projection of Indore House Data\n(First 2 of {pca.n_components_} selected components, reduced from {X.shape[1]} features)",
        fontsize=14,
        fontweight="bold"
    )
    
    plt.colorbar(scatter, label="Sample Index")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot instead of showing it interactively
    plt.savefig("pca_visualization.png", dpi=100, bbox_inches="tight")
    print("✓ PCA visualization saved as 'pca_visualization.png'")
    plt.close()

    # ========== STEP 14: SUMMARY ==========
    print("\n" + "=" * 60)
    print("PCA ANALYSIS COMPLETE!")
    print("=" * 60)
    print(f"\n KEY RESULTS:")
    print(f"  • Original features: {X.shape[1]}")
    print(f"  • Reduced components: {X_transformed.shape[1]}")
    print(f"  • Compression ratio: {X.shape[1]/X_transformed.shape[1]:.1f}x")
    print(f"  • Variance retained: {total_retained:.2f}%")
    print(f"\n BENEFITS OF PCA:")
    print(f"  ✓ Reduced dimensionality: {X.shape[1]} → {X_transformed.shape[1]} features")
    print(f"  ✓ Removed noise: Kept only high-variance directions")
    print(f"  ✓ Faster training: {X.shape[1]/X_transformed.shape[1]:.1f}x fewer features")
    print(f"  ✓ Reduced multicollinearity: Principal components are uncorrelated")
    print(f"  ✓ Enabled visualization: Now 2D and easy to interpret")
    print("=" * 60)


if __name__ == "__main__":
    main()
