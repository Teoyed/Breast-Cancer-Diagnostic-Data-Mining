import pandas as pd
from sklearn.model_selection import train_test_split

# Load your dataset
data = pd.read_csv('Docs/data.csv')

# Select specific features (columns)
selected_features = ['id', 'diagnosis', 'perimeter_worst', 'area_se', 'concavity_se', 'perimeter_se', 'radius_se', 'fractal_dimension_se', 'area_worst', 'smoothness_se', 'area_mean', 'symmetry_se']

data = data[selected_features]

# Split the dataset (80% train, 20% test)
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42, shuffle=True)

# Save the splits
train_data.to_csv('tasks/Task 1/mrmr_train_data.csv', index=False)
test_data.to_csv('tasks/Task 1/mrmr_test_data.csv', index=False)

print("Training set shape:", train_data.shape)
print("Testing set shape:", test_data.shape)