import pandas as pd
from sklearn.model_selection import train_test_split

# Load your dataset
data = pd.read_csv('Docs/data.csv')

# Select specific features (columns)
selected_features = ['id', 'diagnosis', 'perimeter_worst', 'area_se', 'concavity_se', 'perimeter_se', 'radius_se', 'fractal_dimension_se', 'area_worst', 'smoothness_se', 'area_mean', 'symmetry_se']

data = data[selected_features]

# Save the splits
data.to_csv('tasks/Task 1 - MRMR Implementation/mrmr_dataset.csv', index=False)

print("Dataset shape:", data.shape)