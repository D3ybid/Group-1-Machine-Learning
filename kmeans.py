import pandas as pd

# Load the dataset
file_path = "wine-clustering.csv"  # Ensure this file is in your working directory
df = pd.read_csv(file_path)

# Select only 5 columns
selected_columns = df.columns[:5]  # Select the first 5 columns
reduced_df = df[selected_columns].sample(n=100, random_state=42)

# Save the reduced dataset
reduced_df.to_csv("wine-clustering-reduced.csv", index=False)

print("Dataset reduced to 100 rows with 5 columns and saved as 'wine-clustering-reduced.csv'")
