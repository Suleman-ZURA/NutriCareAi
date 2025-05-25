import pandas as pd

raw_csv_path = "nutricare_dataset.csv"
clean_csv_path = "nutricare_dataset_clean.csv"

print(f"📥 Reading raw CSV file: {raw_csv_path}")

try:
    df = pd.read_csv(
        raw_csv_path,
        quotechar='"',
        on_bad_lines='skip',  # <-- skips malformed lines safely in new pandas
        encoding='utf-8'
    )
except Exception as e:
    print(f"❌ Error reading CSV: {e}")
    exit(1)

print(f"📊 Original rows read (excluding bad lines): {len(df)}")

# Drop rows missing essential columns
df.dropna(subset=['food_entity', 'disease_entity', 'is_cause'], inplace=True)

print(f"✅ Rows after dropping missing essential columns: {len(df)}")

# Optional: keep only cause relationships
df = df[df['is_cause'] == 1]

print(f"✅ Rows after filtering to cause relationships: {len(df)}")

# Save cleaned CSV
df.to_csv(clean_csv_path, index=False, encoding='utf-8')

print(f"💾 Cleaned dataset saved to: {clean_csv_path}")
