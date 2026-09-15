import re
import pandas as pd
from huggingface_hub import hffs


BUCKET = "Invicto69/Jobs-Dataset-bucket"

CHANGES_PATH = (
    f"buckets/{BUCKET}/data/minimal/changes/*.parquet"
)


# Find all daily parquet files
files = hffs.glob(CHANGES_PATH)

dated_files = []

for file in files:
    match = re.search(r"(\d{4}-\d{2}-\d{2})\.parquet$", file)

    if match:
        dated_files.append(
            (match.group(1), file)
        )


if not dated_files:
    raise RuntimeError("No daily job files found.")


# Find latest available file
latest_date, latest_file = max(dated_files)

print(f"Latest dataset date: {latest_date}")
print(f"Reading: {latest_file}")


# Read the latest parquet file
with hffs.open(latest_file, "rb") as f:
    df = pd.read_parquet(f)


print(f"Jobs in daily file: {len(df):,}")

print("\nColumns:")
print(df.columns.tolist())


# Target roles
target_roles = [
    r"\bdata engineer\b",
    r"\banalytics engineer\b",
    r"\bbig data engineer\b",
    r"\bbi developer\b",
    r"\bbusiness intelligence developer\b",
    r"\bdata analyst\b",
]

role_pattern = "|".join(target_roles)


# Filter target job titles
jobs = df[
    df["title"]
    .fillna("")
    .str.lower()
    .str.contains(role_pattern, regex=True)
].copy()


# Remove duplicates
jobs = jobs.drop_duplicates(
    subset=["job_id"]
)


# Sort newest first
jobs["posted_at"] = pd.to_datetime(
    jobs["posted_at"],
    errors="coerce",
    utc=True
)

jobs = jobs.sort_values(
    "posted_at",
    ascending=False
)


# Save results
jobs.to_csv(
    "new_jobs.csv",
    index=False
)


print(f"\nMatching jobs found: {len(jobs):,}")
print("Saved as new_jobs.csv")


print("\nSample jobs:")

print(
    jobs[
        [
            "title",
            "country",
            "workplace_type",
            "posted_at",
            "apply_url",
        ]
    ]
    .head(10)
    .to_string(index=False)
)
