import pandas as pd

csv = "assets/reviews_by_asin.csv"

df = pd.read_csv(csv)

df['label'] = df['rating'].apply(lambda x : 1 if x >= 3 else 0)

print(df.head()) 
df.to_csv("assets/reviews_with_labels.csv", index=False)

import pandas as pd

# Path to the saved CSV file
input_csv_file = "assets/reviews_with_labels.csv"
output_csv_file = "assets/reviews_for_train.csv"

# Load the updated CSV file
df = pd.read_csv(input_csv_file)

# Select only the 'labels' and 'text' columns
df_filtered = df[['label', 'text']]

# Drop any NaNs
df_filtered = df_filtered.dropna(subset=['text', 'label'])
print(df_filtered.isnull().sum())  # Verify that no NaN values remain
print(f"Dataset size: {len(df)}")

# Save the filtered DataFrame to a new CSV file
df_filtered.to_csv(output_csv_file, index=False)



print(f"New CSV file with 'label' and 'text' saved as '{output_csv_file}'")

