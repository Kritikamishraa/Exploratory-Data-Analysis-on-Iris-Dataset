# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
df = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv")

# 1. Basic Inspection
print("First 5 Rows:\n", df.head())
print("\nDataset Info:")
df.info()

# 2. Check for missing values
print("\nMissing values:\n", df.isnull().sum())

# 3. Descriptive Statistics
print("\nSummary Stats:\n", df.describe())

# 4. Unique Values in Species
print("\nUnique Species:\n", df['species'].unique())

# 5. Class distribution
print("\nClass Distribution:\n", df['species'].value_counts())

# 6. Pairplot for Feature Relationships
sns.pairplot(df, hue="species")
plt.suptitle("Pairplot of Iris Features", y=1.02)
plt.show()

# 7. Correlation Matrix
correlation_matrix = df.corr(numeric_only=True)
print("\nCorrelation Matrix:\n", correlation_matrix)

sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# 8. GroupBy Mean
group_means = df.groupby("species").mean(numeric_only=True)
print("\nGroup-wise Mean Values:\n", group_means)

# 9. Boxplots for each feature
features = df.columns[:-1]
for feature in features:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x='species', y=feature, data=df)
    plt.title(f'Boxplot of {feature} by Species')
    plt.show()

# 10. NumPy-based Insight: Mean and Std
feature_array = df.iloc[:, :-1].values  # only numeric features
mean_values = np.mean(feature_array, axis=0)
std_dev_values = np.std(feature_array, axis=0)
print("\nMean of Features (NumPy):", mean_values)
print("Standard Deviation of Features (NumPy):", std_dev_values)
