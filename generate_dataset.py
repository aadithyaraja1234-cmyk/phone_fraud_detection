import pandas as pd
import numpy as np

np.random.seed(42)
n = 100_000

def generate_data(n, label):
    data = pd.DataFrame()
    data["number"] = np.random.randint(9000000000, 9999999999, n, dtype=np.int64)

    if label == 0:  # Not Fraud
        data["total_calls"] = np.random.randint(40, 250, n)
        data["avg_duration"] = np.random.uniform(3, 10, n)
        data["night_calls"] = np.random.randint(0, 10, n)
        data["unique_contacts"] = np.random.randint(40, 150, n)
        data["imei_changes"] = np.random.randint(0, 2, n)
        data["scam_keyword_flag"] = np.random.choice([0, 1], n, p=[0.96, 0.04])
        data["short_call_ratio"] = np.random.uniform(0.05, 0.4, n)

    elif label == 1:  # Fraud
        data["total_calls"] = np.random.randint(100, 500, n)
        data["avg_duration"] = np.random.uniform(0.3, 3.5, n)
        data["night_calls"] = np.random.randint(10, 100, n)
        data["unique_contacts"] = np.random.randint(5, 60, n)
        data["imei_changes"] = np.random.randint(0, 4, n)
        data["scam_keyword_flag"] = np.random.choice([0, 1], n, p=[0.35, 0.65])
        data["short_call_ratio"] = np.random.uniform(0.4, 0.95, n)

    return data

# Generate main data groups
normal = generate_data(60_000, 0)
fraud = generate_data(30_000, 1)

# Borderline (mixed behavior)
borderline = pd.DataFrame()
borderline["number"] = np.random.randint(9000000000, 9999999999, 10_000, dtype=np.int64)
borderline["total_calls"] = np.random.randint(60, 300, 10_000)
borderline["avg_duration"] = np.random.uniform(1.5, 6, 10_000)
borderline["night_calls"] = np.random.randint(3, 40, 10_000)
borderline["unique_contacts"] = np.random.randint(20, 100, 10_000)
borderline["imei_changes"] = np.random.randint(0, 3, 10_000)
borderline["scam_keyword_flag"] = np.random.choice([0, 1], 10_000, p=[0.75, 0.25])
borderline["short_call_ratio"] = np.random.uniform(0.25, 0.6, 10_000)

# Combine
df = pd.concat([normal, fraud, borderline], ignore_index=True)

# Fraud scoring system (balanced)
def fraud_score(row):
    score = (
        (row["short_call_ratio"] * 3.0)
        + (row["scam_keyword_flag"] * 2.5)
        + (row["night_calls"] / 30)
        + (row["imei_changes"] * 1.2)
        - (row["avg_duration"] / 4)
        - (row["unique_contacts"] / 80)
        + np.random.normal(0, 0.4)  # add randomness
    )
    return score

df["fraud_score"] = df.apply(fraud_score, axis=1)

# Normalize to [0, 1] and add mild noise
score_scaled = 1 / (1 + np.exp(-df["fraud_score"]))
score_scaled = np.clip(score_scaled + np.random.normal(0, 0.05, len(score_scaled)), 0, 1)

# Assign binary label (more balanced)
df["is_fraud"] = (score_scaled > 0.6).astype(int)

# Drop helper
df.drop(columns=["fraud_score"], inplace=True)

# Save final dataset
df.to_csv("fraud_dataset.csv", index=False)
fraud_rate = df["is_fraud"].mean() * 100
print(f"✅ Dataset generated successfully — Fraud rate: {fraud_rate:.2f}%")
print(df.head())
