import json
import pandas as pd

# Path to your JSONL file
jsonl_file_path = "data/Electronics.jsonl"
output_csv_file = "assets/reviews_by_asin.csv"

# Load the data
data = []
with open(jsonl_file_path, 'r', encoding='utf-8') as file:
    for line in file:
        # Parse each line as a JSON object
        json_obj = json.loads(line)
        asin = json_obj.get("asin", "")
        
        # Filter for specific ASIN
        if asin == "B01G8JO5F2":
            data.append({
                "asin": asin,
                "text": json_obj.get("text", ""),  # Get the review text
                "rating": json_obj.get("rating", "")  # Get the rating
            })

# Convert the list of dictionaries to a DataFrame
df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
df.to_csv(output_csv_file, index=False)

print(f"Data successfully saved to {output_csv_file}")
