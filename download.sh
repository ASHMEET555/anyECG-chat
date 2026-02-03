python - <<EOF
import os
import pandas as pd
import wfdb

BASE = "ecg_data1/MIMIC"
os.makedirs(BASE, exist_ok=True)

# Download metadata only (small)
wfdb.dl_database("mimic-iv-ecg", dl_dir="mimic_meta", records=None)

df = pd.read_csv("mimic_meta/records.csv")

# Select ~5000 ECGs
records = df["record_name"].values[:5000]

for r in records:
    wfdb.dl_database(
        "mimic-iv-ecg",
        dl_dir=BASE,
        records=[r],
        annotators=[]
    )

print("Downloaded", len(records), "ECGs")
EOF