import json
import numpy as np
import pandas as pd

# 1. Load the metadata to get the match names (the column labels)
with open("activations_meta.json", "r", encoding="utf-8") as f:
    meta = json.load(f)
match_labels = meta["labels"]

# 2. Load the binary raw numbers array
# Shape will be (15 matches, 28 layers)
raw_matrix = np.load("activations.npy")

# 3. Create a clean Pandas DataFrame
# We transpose it (.T) so layers are rows and matches are columns, matching your heatmap
layer_names = [f"Layer {i}" for i in range(28)]
df = pd.DataFrame(raw_matrix.T, index=layer_names, columns=match_labels)

# 4. (Optional) Add a row at the bottom showing the exact peak layer (l*) for each match
df.loc["Peak Layer (l*)"] = meta["l_star"]

# 5. Save it to a readable file
df.to_csv("tournament_activations_raw.csv")
# If you have openpyxl installed, you can also save to Excel:
# df.to_excel("tournament_activations_raw.xlsx")

# 6. Display the first few rows and columns in your terminal to see how it looks
print("=== RAW NUMERICAL ACTIVATION DATA (Sample) ===")
print(df.iloc[:, :4].round(2))  # Shows the first 4 matches
