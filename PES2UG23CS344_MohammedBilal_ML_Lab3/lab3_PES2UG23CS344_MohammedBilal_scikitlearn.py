import numpy as np
from sklearn.datasets import load_iris, load_wine

def get_entropy_of_dataset(data: np.ndarray) -> float:
    """
    Calculate the entropy of the dataset using the target column (last column).
    """
    if data.shape[0] == 0:
        return 0.0

    target_col = data[:, -1]
    classes, counts = np.unique(target_col, return_counts=True)
    probs = counts / counts.sum()

    entropy = -np.sum(probs * np.log2(probs, where=(probs > 0)))
    return float(entropy)


def get_avg_info_of_attribute(data: np.ndarray, attribute: int) -> float:
    """
    Calculate the average information (weighted entropy) of an attribute.
    """
    total_rows = data.shape[0]
    if total_rows == 0:
        return 0.0

    attr_values = data[:, attribute]
    unique_vals = np.unique(attr_values)

    avg_info = 0.0
    for val in unique_vals:
        subset = data[attr_values == val]
        weight = subset.shape[0] / total_rows
        subset_entropy = get_entropy_of_dataset(subset)
        avg_info += weight * subset_entropy

    return float(avg_info)


def get_information_gain(data: np.ndarray, attribute: int) -> float:
    """
    Calculate the Information Gain for an attribute.
    """
    dataset_entropy = get_entropy_of_dataset(data)
    avg_info = get_avg_info_of_attribute(data, attribute)
    info_gain = dataset_entropy - avg_info
    return round(float(info_gain), 4)


def get_selected_attribute(data: np.ndarray) -> tuple:
    """
    Select the best attribute (highest information gain).
    Returns (dict of gains, index of best attribute)
    """
    n_attributes = data.shape[1] - 1 
    info_gains = {}

    for attr in range(n_attributes):
        info_gains[attr] = get_information_gain(data, attr)

    best_attr = max(info_gains, key=info_gains.get)
    return info_gains, best_attr


def prepare_sklearn_dataset(features: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    Append labels as the last column to match our entropy/info-gain functions.
    """
    return np.column_stack((features, labels))


if __name__ == "__main__":
    iris = load_iris()
    data = prepare_sklearn_dataset(iris.data, iris.target)

    print("Entropy of full dataset:", get_entropy_of_dataset(data))
    print("Information Gain dict + best attribute:", get_selected_attribute(data))

    wine = load_wine()
    wine_data = prepare_sklearn_dataset(wine.data, wine.target)
    print("\nWine dataset best attribute:", get_selected_attribute(wine_data))
