# config.py
# Central configuration for the Hybrid IDS project.
# All paths are built from PROJECT_DIR so the project works on any machine.

import os

# ── Root ─────────────────────────────────────────────────────────────────────
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(PROJECT_DIR, "data")
MODELS_DIR  = os.path.join(PROJECT_DIR, "models")

# ── Snortlogs folder ──────────────────────────────────────────────────────────
# This is the VirtualBox shared folder.
# On Windows host it maps to:  Project\snortlogs\
# On Ubuntu VM it maps to:     /media/sf_snortlogs/
# The Windows path is used here because Python runs on Windows.
LOGS_DIR = os.path.join(PROJECT_DIR, "snortlogs")

# ── Dataset files (used ONLY for offline ML training / evaluation) ─────────
# KDDTrain+.txt  → teaches the model what normal and attack traffic looks like
# KDDTest+.txt   → used after training to measure accuracy / precision / recall
# These files are NOT touched during live detection.
TRAIN_FILE = os.path.join(DATA_DIR, "KDDTrain+.txt")
TEST_FILE  = os.path.join(DATA_DIR, "KDDTest+.txt")

# ── Model artefacts (produced by train.py, consumed by the live pipeline) ──
MODEL_FILE    = os.path.join(MODELS_DIR, "ids_model.pkl")
FEATURES_FILE = os.path.join(MODELS_DIR, "feature_names.pkl")
SCALER_FILE   = os.path.join(MODELS_DIR, "scaler.pkl")
ENCODER_FILE  = os.path.join(MODELS_DIR, "encoder.pkl")
LE_FILE       = os.path.join(MODELS_DIR, "label_encoders.pkl")
METRICS_FILE  = os.path.join(MODELS_DIR, "metrics.json")

# ── Live input files (written by Snort / tcpdump on Ubuntu into shared folder)
# alert        → Snort fast-alert log. One line per matched rule.
# traffic.pcap → Full packet capture by tcpdump.
ALERT_FILE = os.path.join(LOGS_DIR, "alert")
PCAP_FILE  = os.path.join(LOGS_DIR, "traffic.pcap")

# ── Pipeline output files (read by the dashboard) ─────────────────────────
# hybrid_results.csv        → ML verdict on every Snort alert  (Tab 1)
# pcap_analysis_results.csv → all pcap flows, ML verdict on missed flows (Tab 2)
RESULTS_FILE = os.path.join(PROJECT_DIR, "hybrid_results.csv")
PCAP_RESULTS = os.path.join(PROJECT_DIR, "pcap_analysis_results.csv")

# ── ML training parameters ─────────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE    = 0.2   # 80 % train / 20 % test split inside train.py

# ── KDD column layout ──────────────────────────────────────────────────────
# The KDD files have 42 columns. After one-hot encoding protocol_type,
# service, and flag the model sees 51 features total (stored in feature_names.pkl).
KDD_NUMERIC_COLS = [
    "duration", "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent",
    "hot", "num_failed_logins", "logged_in", "num_compromised", "root_shell",
    "su_attempted", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login",
    "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate",
    "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
]
KDD_CATEGORICAL_COLS = ["protocol_type", "service", "flag"]
KDD_ALL_COLS         = [KDD_NUMERIC_COLS[0]] + KDD_CATEGORICAL_COLS + KDD_NUMERIC_COLS[1:] + ["label"]
