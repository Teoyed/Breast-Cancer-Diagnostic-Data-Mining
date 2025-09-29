import pandas as pd

# UCI dataset URL
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"

# Column names from UCI
columns = ["ID", "Diagnosis",
           "radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean",
           "compactness_mean","concavity_mean","concave_points_mean","symmetry_mean","fractal_dimension_mean",
           "radius_se","texture_se","perimeter_se","area_se","smoothness_se",
           "compactness_se","concavity_se","concave_points_se","symmetry_se","fractal_dimension_se",
           "radius_worst","texture_worst","perimeter_worst","area_worst","smoothness_worst",
           "compactness_worst","concavity_worst","concave_points_worst","symmetry_worst","fractal_dimension_worst"]

# Read dataset
df = pd.read_csv(url, header=None, names=columns)

# Drop ID column (optional, since it's not useful for ML)
df = df.drop("ID", axis=1)

# Save as new CSV file
df.to_csv("breast_cancer_wisconsin_clean.csv", index=False)

print("Saved CSV with shape:", df.shape)