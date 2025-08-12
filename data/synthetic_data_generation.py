import numpy as np
import pandas as pd

# Number of synthetic samples and features
samples = 10
features = 100

# Create synthetic Sample IDs
sample_ID = [f"SP_{i+1}" for i in range(samples)]

# Create synthetic gene names
gene_names = [f"Gene_{i+1}" for i in range(features)]

# Generate random gene expression values (0 to 10)
synthetic_features = np.random.rand(samples, features) * 10

# Create features DataFrame with Sample_ID first
features_df = pd.DataFrame(synthetic_features, columns=gene_names)
features_df.insert(0, "Sample_ID", sample_ID)

# Save features dataset
features_df.to_csv("synthetic_features_data.csv", index=False)

# Create labels (0 or 1) and save
targets = np.random.randint(0, 2, size=samples)
label_df = pd.DataFrame({"Sample_ID": sample_ID, "Target": targets})
label_df.to_csv("synthetic_labels.csv", index=False)

print("Synthetic datasets (features + labels) created successfully!")
