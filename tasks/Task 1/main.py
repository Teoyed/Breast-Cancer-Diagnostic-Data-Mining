import pandas as pd
from sklearn.model_selection import train_test_split

# Load your dataset
data = pd.read_csv('Docs/data.csv')  # replace with your actual file name

# Split the dataset (80% train, 20% test)
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42, shuffle=True)

# Save the splits if you want
train_data.to_csv('tasks/Task 1/train_data.csv', index=False)
test_data.to_csv('tasks/Task 1/test_data.csv', index=False)

print("Training set shape:", train_data.shape)
print("Testing set shape:", test_data.shape)