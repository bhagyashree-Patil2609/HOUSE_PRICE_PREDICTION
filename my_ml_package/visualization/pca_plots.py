import matplotlib.pyplot as plt

def plot_pca(X_transformed):

    plt.scatter(
        X_transformed[:, 0],
        [0] * len(X_transformed)
    )

    plt.title("PCA Projection")

    plt.xlabel("Principal Component 1")

    plt.show()