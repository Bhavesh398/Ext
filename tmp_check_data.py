from data.data_loader import load_all_datasets
df = load_all_datasets()
print("\n=== SUMMARY ===")
print(f"Total rows: {len(df)}")
print(f"Phishing:   {df['label'].sum()}")
print(f"Safe:       {len(df) - df['label'].sum()}")
print(f"Phishing %: {df['label'].mean():.1%}")
