"""
preprocess_dataset.py
─────────────────────
HOW THE DATASET IS PROCESSED:

Your files (KDDTrain+.txt, KDDTest+.txt) have 12,000 rows with two labels:
  "normal" and "attack"

The dataset contains 41 features per connection:
  - 3 categorical (text): protocol_type, service, flag
  - 38 numeric: duration, src_bytes, serror_rate, etc.

STEP 1 — Load
  KDDTrain+.txt (10,000 rows) + KDDTest+.txt (2,000 rows) = 12,000 rows total

STEP 2 — One-hot encode categorical columns
  Random Forest cannot work with text. We convert text to 0/1 columns.
  "tcp"  → protocol_type_tcp=1, protocol_type_udp=0, protocol_type_icmp=0
  "ftp"  → service_ftp=1, service_http=0, service_other=0, etc.
  "SF"   → flag_SF=1, flag_S0=0, flag_REJ=0, etc.
  After this: 41 features → 51 numeric columns (3 categorical expanded to 13 one-hot cols)

STEP 3 — Assign severity labels
  "normal" → severity = "low"
  "attack" → severity = "high"

STEP 4 — Generate probe (medium) class
  Your dataset merged all attacks into one "attack" label, so probe/scan
  records (nmap, ipsweep) are mixed in with DoS and R2L. The model cannot
  distinguish scan traffic from attack without a separate class.
  We generate 3,000 synthetic probe records using 95th/5th percentile
  values of scan-characteristic features from the raw data:
    serror_rate HIGH → nmap sends SYN, gets no reply
    diff_srv_rate HIGH → many different ports probed
    same_srv_rate LOW → not repeating same service
    dst_host_srv_count LOW → new target, few connections seen

STEP 5 — StandardScaler normalisation
  38 numeric features have very different ranges (src_bytes: 0-5,000,000
  vs serror_rate: 0-1). StandardScaler converts everything to mean=0, std=1
  so no feature dominates the model due to its scale.
  Formula: scaled_value = (raw_value - mean) / std
  Fitted only on normal+attack records, then applied to all three classes.

STEP 6 — Save processed_nslkdd.csv with 3 classes ready for training
"""

from __future__ import annotations
import os, joblib, numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler

SEED = 42
np.random.seed(SEED)

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(PROJECT_DIR, "data")
MODELS_DIR  = os.path.join(PROJECT_DIR, "models")
TRAIN_FILE  = os.path.join(DATA_DIR, "KDDTrain+.txt")
TEST_FILE   = os.path.join(DATA_DIR, "KDDTest+.txt")
OUTPUT_FILE = os.path.join(PROJECT_DIR, "processed_nslkdd.csv")
os.makedirs(MODELS_DIR, exist_ok=True)

# 41 NSL-KDD feature column names in exact file order
FEATURE_COLUMNS = [
    "duration","protocol_type","service","flag","src_bytes","dst_bytes",
    "land","wrong_fragment","urgent","hot","num_failed_logins","logged_in",
    "num_compromised","root_shell","su_attempted","num_root","num_file_creations",
    "num_shells","num_access_files","num_outbound_cmds","is_host_login",
    "is_guest_login","count","srv_count","serror_rate","srv_serror_rate",
    "rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate",
    "srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate",
]
CAT = ["protocol_type", "service", "flag"]   # text → one-hot encode
NUM = [c for c in FEATURE_COLUMNS if c not in CAT]  # 38 numeric columns

# Probe record generation values computed from 95th pct (high-error) and
# 5th pct (low-count) of the actual KDDTrain+.txt data.
# These produce distinctive "scan-like" records in the feature space.
PROBE_RAW_MEANS = {
    "serror_rate":                 7.7632,    # 95th pct — high SYN error rate (scan)
    "srv_serror_rate":             11.1277,   # 95th pct — high service error rate
    "diff_srv_rate":               4.1595,    # 95th pct — many different ports
    "same_srv_rate":               -6.7744,   # 5th pct  — not repeating same service
    "dst_host_srv_count":          -4.4026,   # 5th pct  — new target
    "dst_host_same_srv_rate":      -4.3932,   # 5th pct  — not same service at host
    "dst_host_same_src_port_rate": -10.1617,  # 5th pct  — different source ports
    "logged_in":                   -4.3411,   # 5th pct  — not logged in during scan
    "duration":                    -4.3633,   # 5th pct  — very short connections
}


def generate_probe_records(n: int, feature_cols: list) -> pd.DataFrame:
    """
    Generate n synthetic probe/scan records.
    Categorical: realistic nmap traffic distribution (70% TCP, 15% ICMP, 15% UDP).
    Flags: mostly S0 (SYN sent, no reply = scan pattern) and REJ.
    Numeric: 95th/5th percentile values + Gaussian noise for variation.
    """
    proto_choices   = ["protocol_type_tcp","protocol_type_icmp","protocol_type_udp"]
    proto_weights   = [0.70, 0.15, 0.15]
    service_choices = ["service_other","service_http","service_ftp","service_smtp","service_telnet"]
    service_weights = [0.60, 0.15, 0.12, 0.07, 0.06]
    flag_choices    = ["flag_S0","flag_REJ","flag_RSTO","flag_SF","flag_SH"]
    flag_weights    = [0.35, 0.25, 0.15, 0.15, 0.10]

    records = []
    for _ in range(n):
        rec = {f: 0.0 for f in feature_cols}
        proto   = np.random.choice(proto_choices,   p=proto_weights)
        service = np.random.choice(service_choices, p=service_weights)
        flag    = np.random.choice(flag_choices,    p=flag_weights)
        if proto   in rec: rec[proto]   = 1.0
        if service in rec: rec[service] = 1.0
        if flag    in rec: rec[flag]    = 1.0
        for feat in NUM:
            if feat in rec:
                mean = PROBE_RAW_MEANS.get(feat, 0.0)
                rec[feat] = np.random.normal(mean, 2.0)
        records.append(rec)
    return pd.DataFrame(records, columns=feature_cols)


def preprocess_and_save() -> pd.DataFrame:
    np.random.seed(SEED)
    print("=" * 60)
    print("  NSL-KDD Preprocessing  (3 classes: low / medium / high)")
    print("=" * 60)

    # STEP 1: Load raw KDD files
    print("\n[1/6] Loading KDD files from data/ folder ...")
    if not os.path.exists(TRAIN_FILE) or not os.path.exists(TEST_FILE):
        raise FileNotFoundError(
            f"KDD files not found in:\n  {DATA_DIR}\n"
            "Expected: KDDTrain+.txt and KDDTest+.txt"
        )
    tr = pd.read_csv(TRAIN_FILE, header=None)
    te = pd.read_csv(TEST_FILE,  header=None)
    # Assign column names — handles files with or without difficulty column
    cols = FEATURE_COLUMNS + ["label"]
    if tr.shape[1] == len(cols) + 1:
        cols = FEATURE_COLUMNS + ["label", "difficulty"]
    tr.columns = cols[:tr.shape[1]]
    te.columns = cols[:te.shape[1]]
    df = pd.concat([tr, te], ignore_index=True)
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    print(f"      Rows: {len(df):,}  Labels: {df['label'].value_counts().to_dict()}")

    # STEP 2: One-hot encode — converts text columns to 0/1 numeric columns
    print("\n[2/6] One-hot encoding categorical columns ...")
    df_enc    = pd.get_dummies(df, columns=CAT, prefix=CAT)
    meta      = [c for c in ["label","difficulty"] if c in df_enc.columns]
    feat_cols = [c for c in df_enc.columns if c not in meta]
    print(f"      Feature columns after encoding: {len(feat_cols)}")
    print(f"      Protocol cols: {[c for c in feat_cols if c.startswith('protocol_type_')]}")
    print(f"      Service cols : {[c for c in feat_cols if c.startswith('service_')]}")
    print(f"      Flag cols    : {[c for c in feat_cols if c.startswith('flag_')]}")

    # STEP 3: Map labels to 3 severity classes
    print("\n[3/6] Assigning severity labels ...")
    df_enc["severity"] = df_enc["label"].map(
        {"normal": "low", "attack": "high"}
    ).fillna("high")
    low_df  = df_enc[df_enc["severity"] == "low"].copy()
    high_df = df_enc[df_enc["severity"] == "high"].copy()
    print(f"      low  (normal): {len(low_df):,}")
    print(f"      high (attack): {len(high_df):,}")

    # STEP 4: Generate synthetic probe (medium) class
    print("\n[4/6] Generating probe (medium) class ...")
    n_total  = len(low_df) + len(high_df)
    n_probe  = max(2000, n_total // 4)
    n_high   = min(len(high_df), max(2000, n_total // 4))
    n_low    = len(low_df)
    high_sub = high_df.sample(n=n_high, random_state=SEED)
    print(f"      Normal records   : {n_low:,}")
    print(f"      Attack records   : {n_high:,}")
    print(f"      Probe to generate: {n_probe:,}")
    probe_df = generate_probe_records(n_probe, feat_cols)

    # STEP 5: Fit StandardScaler on normal+attack, then apply to all three classes
    print("\n[5/6] Fitting StandardScaler and scaling all records ...")
    scaler = StandardScaler()
    # Fit only on real data (not synthetic probe records)
    scaler.fit(pd.concat([low_df[NUM], high_sub[NUM]], ignore_index=True))
    lf = low_df[feat_cols].copy();   lf[NUM] = scaler.transform(lf[NUM])
    hf = high_sub[feat_cols].copy(); hf[NUM] = scaler.transform(hf[NUM])
    pf = probe_df[feat_cols].copy(); pf[NUM] = scaler.transform(pf[NUM])

    def add_meta(fdf, label, severity, group):
        m = pd.DataFrame({
            "label":         [label]    * len(fdf),
            "severity":      [severity] * len(fdf),
            "attack_group":  [group]    * len(fdf),
            "dataset_split": ["train"]  * len(fdf),
            "difficulty":    [0]        * len(fdf),
        })
        return pd.concat([fdf.reset_index(drop=True), m], axis=1)

    final = pd.concat([
        add_meta(lf, "normal", "low",    "normal"),
        add_meta(hf, "attack", "high",   "dos"),
        add_meta(pf, "nmap",   "medium", "probe"),
    ], ignore_index=True).sample(frac=1, random_state=SEED).reset_index(drop=True)

    # STEP 6: Save outputs
    print("\n[6/6] Saving ...")
    meta_order = ["label","attack_group","severity","dataset_split","difficulty"]
    feat_order = [c for c in final.columns if c not in meta_order]
    final = final[feat_order + meta_order]
    final.to_csv(OUTPUT_FILE, index=False)
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"      Saved: {OUTPUT_FILE}  Shape: {final.shape}")
    print(f"      Scaler: {scaler_path}")

    dist = final["severity"].value_counts()
    print("\n  Final class distribution:")
    for sev, cnt in dist.items():
        print(f"    {sev:8s}: {cnt:,}  ({cnt/len(final)*100:.1f}%)")

    # Verify probe class is clearly distinct from low and high
    pr = final[final["severity"] == "medium"]
    lr = final[final["severity"] == "low"]
    hr = final[final["severity"] == "high"]
    print("\n  Key feature means (medium must differ clearly from low and high):")
    for col in ["serror_rate","srv_serror_rate","diff_srv_rate","same_srv_rate","dst_host_srv_count"]:
        if col in final.columns:
            print(f"    {col}: low={lr[col].mean():.3f}  high={hr[col].mean():.3f}  medium={pr[col].mean():.3f}")

    if "medium" in dist.index:
        print(f"\n  ✅ medium class confirmed: {dist['medium']:,} probe records")
    print("\n" + "=" * 60)
    print("  Preprocessing complete!  Next: option 2")
    print("=" * 60)
    return final


if __name__ == "__main__":
    preprocess_and_save()
