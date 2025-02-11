import pandas as pd

# Load the dataset
file_path = "iris.csv"  # Ensure this file is in your working directory
df = pd.read_csv(file_path)

# Remove the label column if it exists
if 'Label' in df.columns or 'Species' in df.columns:
    df = df.drop(columns=['Label'], errors='ignore')
    df = df.drop(columns=['Species'], errors='ignore')

# Select only 2 features
selected_columns = df.columns[:2]  # Select the first 2 features
reduced_df = df[selected_columns].sample(n=100, random_state=42)

# Save the reduced dataset
reduced_df.to_csv("iris-reduced.csv", index=False)

print("Dataset reduced to 100 rows with 2 features (excluding label) and saved as 'iris-reduced.csv'")
