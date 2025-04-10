import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_histogram(data, bins=10):
    flattened_data = data.flatten()
    plt.figure(figsize=(10, 6))
    sns.histplot(flattened_data, kde=False, color="blue", bins=bins, label="Histogram")
    bin_width = (flattened_data.max() - flattened_data.min()) / bins
    xticks = np.arange(
        flattened_data.min(), flattened_data.max() + bin_width, bin_width
    )
    plt.xticks(xticks, rotation=45)
    plt.title("Data Distribution")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()


def plot_KDE(data):
    flattened_data = data.flatten()
    plt.figure(figsize=(10, 6))
    sns.kdeplot(flattened_data, color="blue", label="KDE")
    plt.title("Data Distribution")
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.legend()
    plt.show()


# 示例用法
if __name__ == "__main__":
    # 创建一个示例多维数组
    example_data = np.random.randn(1000, 3)
    plot_histogram(example_data)
