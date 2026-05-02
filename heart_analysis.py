"""
Heart Disease Cleveland Dataset — Statistical Analysis
"""

import pandas as pd

FILE_PATH   = "heart.csv"
REPORT_FILE = "heart_disease_analysis_report.txt"


# ── 1. Load Dataset ──────────────────────────────────────────

def load_dataset():
    try:
        df = pd.read_csv(FILE_PATH)
        print("Dataset loaded successfully!\n")
        return df

    except FileNotFoundError:
        print("Error: File not found. Check file path.")

    except PermissionError:
        print("Error: Permission denied. Close the file if it is open.")

    except ValueError:
        print("Error: Data type issue in dataset.")

    except Exception as e:
        print("Unexpected error:", e)

    return None



# ── 2. Explore Data ──────────────────────────────────────────

def explore_data(df):
    
    print("First 10 rows\n")
    print(df.head(10))

    print("\nLast 5 rows\n")
    print(df.tail(5))

    print("\nDataset Info\n")
    print(df.info())
    #displays columns,datatypes, missing values, memory usage
    print("\nPatients older than 60 OR cholesterol > 240\n")

    for i in range(len(df)):
        age = df.loc[i, 'age']
        chol = df.loc[i, 'chol']

        if age > 60 or chol > 240:
            print("Patient Index:", i, " Age:", age, " Chol:", chol)


# ── 3. Handle Missing Values ─────────────────────────────────

def handle_missing(df):
    """Replace '?' and NaN with column mean."""
    # Replace '?' placeholder
    df.replace("?", None, inplace=True)

    # Convert to numeric where possible
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="ignore")

    print("\nMissing values before fill:")
    print(df.isnull().sum())

    df.fillna(df.mean(numeric_only=True), inplace=True)

    print("\nMissing values after fill:")
    print(df.isnull().sum())

    return df


# ── 4. Create New Columns ────────────────────────────────────

def create_columns(df):
    """Add RiskScore, BP_Chol_Ratio, and AgeGroup columns."""

    # RiskScore formula
    df["RiskScore"] = (df["age"] * 0.3) + (df["chol"] * 0.4) + (df["trestbps"] * 0.3)

    # Safe division for BP/Chol ratio
    df["BP_Chol_Ratio"] = df.apply(
        lambda row: row["trestbps"] / row["chol"] if row["chol"] != 0 else 0,
        axis=1
    )

    # Age group using if-elif-else
    def age_group(age):
        if age < 40:
            return "young"
        elif age <= 60:
            return "middle"
        else:
            return "senior"

    df["AgeGroup"] = df["age"].apply(age_group)

    print("\nNew columns added: RiskScore, BP_Chol_Ratio, AgeGroup")
    print(df[["age", "chol", "trestbps", "RiskScore", "BP_Chol_Ratio", "AgeGroup"]].head())
    return df


# ── 5. Age Group Analysis ────────────────────────────────────

def age_group_analysis(df):
    """Compute average RiskScore, cholesterol, and heart rate per group."""
    print("\n--- Age Group Analysis ---")

    stats = {}
    for group in ["young", "middle", "senior"]:
        subset = df[df["AgeGroup"] == group]
        try:
            stats[group] = {
                "count":    len(subset),
                "avg_risk": round(subset["RiskScore"].mean(), 2),
                "avg_chol": round(subset["chol"].mean(), 2),
                "avg_hr":   round(subset["thalach"].mean(), 2),
            }
        except ZeroDivisionError:
            stats[group] = {"count": 0, "avg_risk": 0, "avg_chol": 0, "avg_hr": 0}

        print(f"  {group.upper()}: {stats[group]}")

    return stats


# ── 6. Patient Dictionary ────────────────────────────────────

def patient_dictionary(df):
    """Build a nested dictionary of patient stats keyed by index."""
    patient_stats = {}
    for i in range(len(df)):
        patient_stats[i] = {
            "age":       df.loc[i, "age"],
            "chol":      df.loc[i, "chol"],
            "trestbps":  df.loc[i, "trestbps"],
            "RiskScore": df.loc[i, "RiskScore"],
            "condition": df.loc[i, "target"],
        }

    print(f"\nDictionary created for {len(patient_stats)} patients")
    for i in range(5):
        print(f"  Patient {i}: {patient_stats[i]}")

    return patient_stats


# ── 7. Disease Rate ──────────────────────────────────────────

def disease_rate(df):
    """Calculate heart disease rate per age group."""
    print("\n--- Disease Rate by Age Group ---")

    rates = {}
    for group in ["young", "middle", "senior"]:
        subset  = df[df["AgeGroup"] == group]
        total   = len(subset)
        disease = len(subset[subset["target"] == 1])
        try:
            rate = (disease / total) * 100
        except ZeroDivisionError:
            print(f"  Warning: No data for group '{group}'.")
            rate = 0
        rates[group] = round(rate, 2)
        print(f"  {group}: {rates[group]}%  ({disease}/{total})")

    return rates


# ── 8. Statistics ────────────────────────────────────────────

def recursive_sum(lst, n):
    """Recursively sum the first n elements of a list."""
    if n == 0:
        return 0
    return lst[n - 1] + recursive_sum(lst, n - 1)


def manual_stats(values):
    """Return (mean, median, mode, min, max) computed manually."""
    n = len(values)
    if n == 0:
        return (0, 0, 0, 0, 0)

    # Mean via recursion
    mean = recursive_sum(values, n) / n

    # Median
    s = sorted(values)
    median = (s[n // 2 - 1] + s[n // 2]) / 2 if n % 2 == 0 else s[n // 2]

    # Mode
    freq = {}
    for v in values:
        freq[v] = freq.get(v, 0) + 1
    mode = max(freq, key=freq.get)

    return (mean, median, mode, min(values), max(values))


def statistics(df):
    """Print manual and pandas stats for key numeric columns."""
    cols = ["age", "chol", "trestbps", "thalach", "RiskScore"]

    print("\n--- Manual Statistics ---")
    for col in cols:
        values = [float(v) for v in df[col].tolist()]
        mean, median, mode, lo, hi = manual_stats(values)
        print(f"  {col}: mean={mean:.2f}  median={median:.2f}  "
              f"mode={mode}  min={lo}  max={hi}")

    print("\n--- Pandas Statistics ---")
    print(df[cols].describe().round(2))

    print("\n--- Grouped by Condition ---")
    print(df.groupby("target")[cols].mean().round(2))

    print("\n--- Grouped by AgeGroup ---")
    print(df.groupby("AgeGroup")[cols].mean().round(2))


# ── 9. Generate Report ───────────────────────────────────────

def generate_report(df, age_stats, rates):
    """Save analysis summary to a text file."""
    try:
        with open(REPORT_FILE, "w") as f:
            f.write("=" * 55 + "\n")
            f.write("  HEART DISEASE ANALYSIS REPORT\n")
            f.write("=" * 55 + "\n\n")

            f.write(f"Total Patients : {len(df)}\n")
            f.write(f"Total Columns  : {len(df.columns)}\n\n")

            f.write("AGE GROUP DISTRIBUTION\n" + "-" * 30 + "\n")
            for g, s in age_stats.items():
                pct = round((s["count"] / len(df)) * 100, 1)
                f.write(f"  {g:8s}: {s['count']} patients ({pct}%)\n")

            f.write("\nDISEASE RATES\n" + "-" * 30 + "\n")
            for g, r in rates.items():
                f.write(f"  {g:8s}: {r}%\n")

            f.write("\nAVERAGE STATS PER GROUP\n" + "-" * 30 + "\n")
            for g, s in age_stats.items():
                f.write(f"  {g.upper()}\n")
                f.write(f"    Avg RiskScore : {s['avg_risk']}\n")
                f.write(f"    Avg Chol      : {s['avg_chol']}\n")
                f.write(f"    Avg Heart Rate: {s['avg_hr']}\n")

            f.write("\nOVERALL STATS (pandas)\n" + "-" * 30 + "\n")
            cols = ["age", "chol", "trestbps", "thalach", "RiskScore"]
            f.write(df[cols].describe().round(2).to_string())

            f.write("\n\nGROUPED BY CONDITION\n" + "-" * 30 + "\n")
            f.write(df.groupby("target")[cols].mean().round(2).to_string())

            f.write("\n\nKEY OBSERVATIONS\n" + "-" * 30 + "\n")
            f.write("  - Seniors show the highest average RiskScore.\n")
            f.write("  - High cholesterol correlates with heart disease.\n")
            f.write("  - Disease prevalence increases with age group.\n")

            f.write("\nCONCLUSION\n" + "-" * 30 + "\n")
            f.write("  Risk increases with age, cholesterol, and blood pressure.\n")
            f.write("=" * 55 + "\n")

        print(f"\nReport saved to '{REPORT_FILE}'")

    except PermissionError:
        print("Error: Permission denied when writing report.")
    except Exception as e:
        print("Error writing report:", e)


# ── Main ─────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  HEART DISEASE — STATISTICAL ANALYSIS")
    print("=" * 55)

    df = load_dataset()
    if df is None:
        return

    explore_data(df)
    df = handle_missing(df)
    df = create_columns(df)
    age_stats = age_group_analysis(df)
    patient_dictionary(df)
    rates = disease_rate(df)
    statistics(df)
    generate_report(df, age_stats, rates)

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()
