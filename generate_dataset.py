import pandas as pd
import numpy as np

np.random.seed(42)
n = 100_000

# ---------------------------------------------------------------------------
# Design note (why this isn't a trivial formula-recovery problem)
#
# An earlier version of this generator computed `is_fraud` directly from a
# weighted formula over the exact same 7 columns the model is trained on
# (short_call_ratio*3.0 + scam_keyword_flag*2.5 + ...). That makes the
# "classification" task pure formula recovery: a model can hit ~96% just by
# re-deriving the label-generating function, which says nothing about
# whether it would work on real fraud signals.
#
# Here the label instead comes from a HIDDEN latent risk factor that the
# model never sees. The 7 visible features are noisy, only partially
# informative proxies of that latent factor -- exactly like real telecom
# fraud signals are partial, noisy proxies of a fraudster's true intent.
# We also apply an "evasiveness" factor (sophisticated fraud disguises
# itself better in the visible signals) and random label noise (real
# fraud labels are never perfectly clean -- they come from complaints,
# investigations, and heuristics that are themselves imperfect).
# This gives the model genuine irreducible uncertainty to contend with.
# ---------------------------------------------------------------------------

# Hidden latent fraud propensity: most numbers are low-risk, with a
# long tail of high-risk ones. Never written to the output CSV.
latent_risk = np.random.beta(2, 5, n)

# Hidden "evasiveness": how well a fraudulent operation disguises itself
# in the visible call-behavior signals. Also never written to the CSV.
evasiveness = np.random.uniform(0, 1, n)

# How strongly the latent risk actually shows up in observable behavior,
# dampened for evasive operations.
signal = latent_risk * (1 - 0.2 * evasiveness)


def noisy(base, noise_std, lo=None, hi=None):
    val = base + np.random.normal(0, noise_std, n)
    if lo is not None or hi is not None:
        val = np.clip(val, lo, hi)
    return val


total_calls = noisy(80 + 300 * signal, 28, 5, None).round().astype(int)
avg_duration = noisy(9 - 7.5 * signal, 1.2, 0.2, 15)
night_calls = noisy(5 + 70 * signal, 6, 0, None).round().astype(int)
unique_contacts = noisy(110 - 80 * signal, 14, 3, None).round().astype(int)
imei_changes = np.random.poisson(np.clip(0.3 + 3 * signal * (1 - 0.1 * evasiveness), 0, None))
scam_keyword_flag = (
    np.random.uniform(0, 1, n) < (0.04 + 0.55 * signal * (1 - 0.2 * evasiveness))
).astype(int)
short_call_ratio = noisy(0.15 + 0.7 * signal, 0.055, 0.02, 0.98)

df = pd.DataFrame(
    {
        "number": np.random.randint(9000000000, 9999999999, n, dtype=np.int64),
        "total_calls": total_calls,
        "avg_duration": avg_duration,
        "night_calls": night_calls,
        "unique_contacts": unique_contacts,
        "imei_changes": imei_changes,
        "scam_keyword_flag": scam_keyword_flag,
        "short_call_ratio": short_call_ratio,
    }
)

# Label comes from the hidden latent factor (plus its own noise), NOT from
# the visible feature columns above.
label_noise = np.random.normal(0, 0.08, n)
is_fraud = ((latent_risk + label_noise) > 0.55).astype(int)

# Realistic label noise: fraud-labeling pipelines (complaints,
# investigations, heuristics) are never perfectly clean. Flip a small
# fraction of labels to simulate that.
flip_mask = np.random.uniform(0, 1, n) < 0.03
is_fraud = np.where(flip_mask, 1 - is_fraud, is_fraud)

df["is_fraud"] = is_fraud

# Save final dataset
df.to_csv("fraud_dataset.csv", index=False)
fraud_rate = df["is_fraud"].mean() * 100
print(f"Dataset generated successfully - Fraud rate: {fraud_rate:.2f}%")
print(df.head())
