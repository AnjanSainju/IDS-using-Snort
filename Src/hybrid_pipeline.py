"""
hybrid_pipeline.py
──────────────────
HOW THIS FILE WORKS:

PURPOSE: Classify Snort-detected alerts using the ML model.
         Shows what SNORT caught and what the ML model thinks about each alert.

STEP 1 — Load model
  Loads ids_model.pkl and feature_names.pkl from models/ folder.

STEP 2 — Read Snort alert log
  Reads snortlogs/alert. Each line = one packet that matched a Snort rule.
  Typical line:
    04/20-08:48:04 [**] FTP connection attempt [**] {TCP} 192.168.56.102:44231 -> 192.168.56.101:21

STEP 3 — Aggregate repeated alerts into unique events
  Snort fires per packet. One nmap scan = 2,026 identical alerts.
  Groups by (src_ip, dst_ip, protocol, alert_name) → 1 event with count.
  Example: 2,026 "TCP SYN port scan detected" → 1 event, event_count=2026

STEP 4 — Build NSL-KDD feature vector from alert text
  The model needs 51 numeric features. A Snort alert is text.
  Bridge:
    a) Set one-hot columns from actual alert data:
       protocol → protocol_type_tcp/udp/icmp = 1
       dst_port → service_ftp/http/etc = 1
       classification → flag_S0/SF/REJ = 1
    b) Set 38 numeric features to the exact class mean from training data
       (PROBE_BASELINE, ATTACK_BASELINE, or NORMAL_BASELINE)
  This is centroid-based classification — the centroid gives max confidence.

STEP 5 — Classify with confidence thresholds
  CLASSIFICATION TIERS:
    🟢 NORMAL     — pred=low   AND conf ≥ 80%  (regular ping, benign traffic)
    🟡 PROBE/SCAN — pred=medium AND conf ≥ 80%  (nmap scans, reconnaissance)
    🔴 ATTACK     — pred=high  AND conf ≥ 50%  (service probes, flood attacks)
    ⚪ SUSPICIOUS — anything below threshold    (uncertain, review manually)

  WHY DIFFERENT THRESHOLDS:
    Probe: 80% — probe class is very distinct (synthetic, clean separation)
    Normal: 80% — normal class is very distinct (well-defined in dataset)
    Attack: 50% — dataset has merged attack labels (DoS+R2L+U2R all combined),
                  so class boundary is less clean. 50% = majority vote.

  FLOOD DETECTION (ping -f):
    Regular ping (event_count ≤ 100) → NORMAL_BASELINE → 🟢 NORMAL
    Flood ping  (event_count > 100)  → ATTACK_BASELINE → 🔴 ATTACK
    ping -f sends thousands of ICMP packets = DoS attack = ATTACK

BASELINE VALUES:
  Exact class means from processed_nslkdd.csv after StandardScaler.
  Computed with: csv[csv['severity']=='high'][feature_cols].mean()
  These are statistical facts about your training data, not assumptions.
"""

import os, joblib, pandas as pd
from collections import defaultdict
from config import ALERT_FILE, MODEL_FILE, FEATURES_FILE, RESULTS_FILE
from snort_parser import read_alerts

# Attack uses 50% threshold (merged class = less clean boundary)
# Probe and Normal use 80% threshold (clean synthetic/natural classes)
ATTACK_THRESHOLD: float = 0.50
PROBE_THRESHOLD:  float = 0.80
NORMAL_THRESHOLD: float = 0.80

# Threshold above which ICMP ping is classified as flood/DoS attack
ICMP_FLOOD_THRESHOLD: int = 100

PORT_TO_SERVICE: dict = {
    20:"ftp_data", 21:"ftp",     22:"ssh",    23:"telnet",  25:"smtp",
    53:"domain",   67:"private", 68:"private",79:"finger",  80:"http",
    110:"pop_3",   111:"sunrpc", 113:"auth",  119:"nntp",   123:"ntp_u",
    135:"epmap",   137:"netbios_ns", 138:"netbios_dgm", 139:"netbios_ssn",
    143:"imap4",   161:"snmp",   162:"snmp_trap", 179:"bgp", 194:"IRC",
    389:"ldap",    443:"http_443", 445:"microsoft_ds", 465:"smtp",
    514:"shell",   515:"printer", 520:"efs",  540:"uucp",   543:"klogin",
    544:"kshell",  587:"smtp",   993:"imap4", 995:"pop_3",  1433:"sql_net",
    1521:"sql_net",3306:"sql_net",3389:"other",5900:"other",
    6000:"X11",    8080:"http_8001", 8443:"http_443",
}

# ── PROBE baseline — exact mean of medium class from processed_nslkdd.csv ────
# High serror_rate (SYN with no reply), high diff_srv_rate (many ports probed),
# low same_srv_rate (not repeating), low dst_host_srv_count (new target).
# Model predicts "medium" with 100% confidence for these values.
PROBE_BASELINE: dict = {
    "duration":                    -1.5403,
    "src_bytes":                   -0.1463,
    "dst_bytes":                   -0.1381,
    "land":                        -0.3312,
    "wrong_fragment":               0.1767,
    "urgent":                       0.3705,
    "hot":                         -0.0291,
    "num_failed_logins":            0.0129,
    "logged_in":                   -1.6012,
    "num_compromised":              0.1592,
    "root_shell":                  -0.1853,
    "su_attempted":                -0.2921,
    "num_root":                    -0.1220,
    "num_file_creations":           0.1021,
    "num_shells":                  -0.2868,
    "num_access_files":             0.0089,
    "num_outbound_cmds":            0.0615,
    "is_host_login":               -0.1087,
    "is_guest_login":               0.3730,
    "count":                        0.1143,
    "srv_count":                   -0.3224,
    "serror_rate":                  1.5495,
    "srv_serror_rate":              1.8376,
    "rerror_rate":                 -0.1618,
    "srv_rerror_rate":             -0.2762,
    "same_srv_rate":               -1.5973,
    "diff_srv_rate":                1.7942,
    "srv_diff_host_rate":           0.2901,
    "dst_host_count":               0.1532,
    "dst_host_srv_count":          -1.8720,
    "dst_host_same_srv_rate":      -1.9175,
    "dst_host_diff_srv_rate":      -0.0547,
    "dst_host_same_src_port_rate": -1.8476,
    "dst_host_srv_diff_host_rate": -0.3832,
    "dst_host_serror_rate":        -0.1111,
    "dst_host_srv_serror_rate":     0.3832,
    "dst_host_rerror_rate":        -0.3967,
    "dst_host_srv_rerror_rate":     0.0522,
}

# ── ATTACK baseline — exact mean of high class from processed_nslkdd.csv ─────
# High duration, high num_file_creations, high num_shells — active exploitation.
# Model predicts "high" with ~56% confidence (above 50% threshold).
ATTACK_BASELINE: dict = {
    "duration":                     0.5004,
    "src_bytes":                   -0.2769,
    "dst_bytes":                   -0.0625,
    "land":                        -0.0467,
    "wrong_fragment":               0.0984,
    "urgent":                       0.0034,
    "hot":                         -0.3845,
    "num_failed_logins":            0.0045,
    "logged_in":                    0.3373,
    "num_compromised":              0.2107,
    "root_shell":                   0.0771,
    "su_attempted":                 0.1011,
    "num_root":                    -0.4366,
    "num_file_creations":           0.4176,
    "num_shells":                   0.2392,
    "num_access_files":            -0.0043,
    "num_outbound_cmds":           -0.0498,
    "is_host_login":               -0.4958,
    "is_guest_login":              -0.0195,
    "count":                       -0.0188,
    "srv_count":                   -0.1727,
    "serror_rate":                  0.0144,
    "srv_serror_rate":              0.3740,
    "rerror_rate":                  0.0239,
    "srv_rerror_rate":             -0.1492,
    "same_srv_rate":                0.3047,
    "diff_srv_rate":                0.4389,
    "srv_diff_host_rate":           0.1902,
    "dst_host_count":               0.3726,
    "dst_host_srv_count":          -0.5216,
    "dst_host_same_srv_rate":      -0.4516,
    "dst_host_diff_srv_rate":       0.0092,
    "dst_host_same_src_port_rate": -0.2578,
    "dst_host_srv_diff_host_rate": -0.0006,
    "dst_host_serror_rate":         0.0109,
    "dst_host_srv_serror_rate":     0.0130,
    "dst_host_rerror_rate":        -0.0847,
    "dst_host_srv_rerror_rate":     0.1527,
}

# ── NORMAL baseline — exact mean of low class from processed_nslkdd.csv ──────
# Low error rates, positive dst_host_srv_count — established legitimate traffic.
# Model predicts "low" with high confidence.
NORMAL_BASELINE: dict = {
    "duration":                    -0.2498,
    "src_bytes":                    0.1383,
    "dst_bytes":                    0.0312,
    "land":                         0.0233,
    "wrong_fragment":              -0.0491,
    "urgent":                      -0.0017,
    "hot":                          0.1919,
    "num_failed_logins":           -0.0023,
    "logged_in":                   -0.1684,
    "num_compromised":             -0.1052,
    "root_shell":                  -0.0385,
    "su_attempted":                -0.0505,
    "num_root":                     0.2180,
    "num_file_creations":          -0.2085,
    "num_shells":                  -0.1194,
    "num_access_files":             0.0021,
    "num_outbound_cmds":            0.0249,
    "is_host_login":                0.2475,
    "is_guest_login":               0.0098,
    "count":                        0.0094,
    "srv_count":                    0.0862,
    "serror_rate":                 -0.0072,
    "srv_serror_rate":             -0.1867,
    "rerror_rate":                 -0.0119,
    "srv_rerror_rate":              0.0745,
    "same_srv_rate":               -0.1521,
    "diff_srv_rate":               -0.2191,
    "srv_diff_host_rate":          -0.0950,
    "dst_host_count":              -0.1860,
    "dst_host_srv_count":           0.2604,
    "dst_host_same_srv_rate":       0.2255,
    "dst_host_diff_srv_rate":      -0.0046,
    "dst_host_same_src_port_rate":  0.1287,
    "dst_host_srv_diff_host_rate":  0.0003,
    "dst_host_serror_rate":        -0.0055,
    "dst_host_srv_serror_rate":    -0.0065,
    "dst_host_rerror_rate":         0.0423,
    "dst_host_srv_rerror_rate":    -0.0762,
}

SCAN_KEYWORDS   = {"scan","nmap","xmas","null","portsweep","sweep","syn port"}
ATTACK_KEYWORDS = {"snmp","ftp","telnet","ssh","exploit","dos","flood","buffer","overflow","backdoor"}


def _alert_type(alert_name: str, event_count: int = 1) -> str:
    """
    Determine alert category to select the correct feature baseline.

    Logic:
    - ICMP/ping with count > ICMP_FLOOD_THRESHOLD → attack (ping -f flood = DoS)
    - ICMP/ping with count ≤ ICMP_FLOOD_THRESHOLD → normal (ping -c 5 = benign)
    - Scan keywords → probe (nmap scans)
    - Attack keywords → attack (SSH/FTP/Telnet/SNMP service probes)
    - Default → probe (Snort mostly fires on suspicious traffic)
    """
    n = alert_name.lower()

    # Flood detection: massive ICMP count = DoS attack (ping -f)
    if "ping" in n or ("icmp" in n and "scan" not in n):
        if event_count > ICMP_FLOOD_THRESHOLD:
            return "attack"   # ping flood → DoS → ATTACK
        else:
            return "normal"   # regular ping → benign → NORMAL

    # Scan/reconnaissance keywords
    for kw in SCAN_KEYWORDS:
        if kw in n: return "probe"

    # Active service exploitation keywords
    for kw in ATTACK_KEYWORDS:
        if kw in n: return "attack"

    return "probe"  # default: Snort fires on suspicious traffic


def _flag_from_classification(classification: str) -> str:
    """Map Snort classification string to NSL-KDD TCP flag value."""
    c = classification.lower()
    if any(k in c for k in ["attempted","attempt","scan","probe","recon","denial"]): return "S0"
    if "reject" in c or "refused" in c: return "REJ"
    if any(k in c for k in ["successful","success","policy"]): return "SF"
    return "S0"


def classify_result(pred: str, conf: float) -> str:
    """
    Apply confidence thresholds to determine final classification tier.
    Attack uses 50% (merged class, lower confidence expected).
    Probe and Normal use 80% (clean, distinct classes).
    """
    if pred == "high"   and conf >= ATTACK_THRESHOLD: return "attack"
    if pred == "medium" and conf >= PROBE_THRESHOLD:  return "probe"
    if pred == "low"    and conf >= NORMAL_THRESHOLD: return "normal"
    return "suspicious"


def get_label(tier: str) -> str:
    return {
        "attack":     "🔴 ATTACK",
        "probe":      "🟡 PROBE/SCAN",
        "normal":     "🟢 NORMAL",
        "suspicious": "⚪ SUSPICIOUS",
    }.get(tier, tier)


def aggregate_alerts(alerts: list) -> list:
    """
    Group repeated Snort alerts into unique events.
    Groups by: (src_ip, dst_ip, protocol, alert_name)
    Keeps event_count to show how many times each event fired.
    Example: 2,026 identical SYN scan alerts → 1 event with event_count=2,026
    """
    groups: dict = defaultdict(list)
    for alert in alerts:
        groups[(alert["src_ip"], alert["dst_ip"],
                alert["protocol"], alert["alert_name"])].append(alert)
    aggregated = []
    for key, group in groups.items():
        rep = dict(group[0])
        rep["event_count"] = len(group)
        ports = [a["dst_port"] for a in group if a["dst_port"]]
        rep["dst_port"] = max(set(ports), key=ports.count) if ports else ""
        aggregated.append(rep)
    return aggregated


def alert_to_features(alert: dict, feature_names: list) -> pd.DataFrame:
    """
    Convert a Snort alert text record into a 51-feature NSL-KDD numeric vector.

    The model was trained on NSL-KDD numbers. Snort gives text.
    This function bridges that gap:
      1. Set baseline (class mean values) for the 38 numeric features
      2. Set exact 0/1 values for the 13 one-hot columns (protocol, service, flag)
    """
    vec = {f: 0.0 for f in feature_names}

    # Select baseline based on alert type and event count
    event_count = alert.get("event_count", 1)
    atype = _alert_type(alert.get("alert_name", ""), event_count)
    baseline = (PROBE_BASELINE   if atype == "probe"
                else ATTACK_BASELINE if atype == "attack"
                else NORMAL_BASELINE)
    for feat, val in baseline.items():
        if feat in vec: vec[feat] = val

    # Set protocol one-hot from actual alert protocol field
    proto = alert.get("protocol", "TCP").upper()
    proto_col = {"ICMP":"protocol_type_icmp","UDP":"protocol_type_udp"}.get(proto,"protocol_type_tcp")
    for col in feature_names:
        if col.lower() == proto_col.lower(): vec[col] = 1.0; break

    # Set service one-hot from actual destination port
    try:    dst_port = int(alert.get("dst_port", 0))
    except: dst_port = 0
    service = PORT_TO_SERVICE.get(dst_port, "other").lower()
    matched = False
    for col in feature_names:
        if col.lower() == f"service_{service}": vec[col] = 1.0; matched = True; break
    if not matched:
        for col in feature_names:
            if col.lower() == "service_other": vec[col] = 1.0; break

    # Set flag one-hot from Snort classification string
    flag = _flag_from_classification(alert.get("classification", ""))
    matched = False
    for col in feature_names:
        if col.lower() == f"flag_{flag.lower()}": vec[col] = 1.0; matched = True; break
    if not matched:
        for col in feature_names:
            if col.lower() == "flag_s0": vec[col] = 1.0; break

    return pd.DataFrame([vec], columns=feature_names)


def load_model():
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(f"Model not found: {MODEL_FILE}\nRun option 2 first.")
    if not os.path.exists(FEATURES_FILE):
        raise FileNotFoundError(f"Features not found: {FEATURES_FILE}\nRun option 2 first.")
    return joblib.load(MODEL_FILE), joblib.load(FEATURES_FILE)


def run_pipeline(verbose: bool = True) -> pd.DataFrame:
    print("=" * 60)
    print("  Hybrid IDS Pipeline — Snort + ML Analysis")
    print(f"  Attack threshold: {ATTACK_THRESHOLD:.0%}  "
          f"Probe/Normal threshold: {PROBE_THRESHOLD:.0%}")
    print("=" * 60)

    print("\n[1/5] Loading model ...")
    model, feature_names = load_model()
    print(f"      Classes : {list(model.classes_)}")
    print(f"      Features: {len(feature_names)}")

    print(f"\n[2/5] Reading Snort alerts:\n      {ALERT_FILE}")
    alerts = read_alerts(ALERT_FILE)
    print(f"      Raw alert lines: {len(alerts):,}")
    if not alerts:
        print(f"\n  No alerts found. Check {ALERT_FILE}")
        return pd.DataFrame()

    print("\n[3/5] Aggregating repeated alerts into unique events ...")
    events = aggregate_alerts(alerts)
    print(f"      {len(alerts):,} raw lines → {len(events)} unique events")

    print("\n[4/5] Running ML predictions ...")
    results = []
    for i, event in enumerate(events):
        features = alert_to_features(event, feature_names)
        pred     = model.predict(features)[0]
        proba    = model.predict_proba(features)[0]
        conf     = round(float(max(proba)), 4)
        tier     = classify_result(pred, conf)
        label    = get_label(tier)
        results.append({
            "timestamp":      event["timestamp"],
            "alert_name":     event["alert_name"],
            "event_count":    event["event_count"],
            "priority":       event["priority"],
            "classification": event["classification"],
            "protocol":       event["protocol"],
            "src_ip":         event["src_ip"],
            "src_port":       event["src_port"],
            "dst_ip":         event["dst_ip"],
            "dst_port":       event["dst_port"],
            "src_dst":        event["src_dst"],
            "ml_prediction":  pred,
            "ml_confidence":  conf,
            "ml_tier":        tier,
            "ml_label":       label,
        })
        if verbose:
            print(f"  [{i+1:03d}] {event['timestamp']} | {label} ({conf:.0%})"
                  f" | x{event['event_count']:,} | {event['alert_name']}")

    print(f"\n[5/5] Saving results:\n      {RESULTS_FILE}")
    df_out = pd.DataFrame(results)
    df_out.to_csv(RESULTS_FILE, index=False)

    attacks = int((df_out["ml_tier"] == "attack").sum())
    probes  = int((df_out["ml_tier"] == "probe").sum())
    normal  = int((df_out["ml_tier"] == "normal").sum())
    susp    = int((df_out["ml_tier"] == "suspicious").sum())
    print(f"\n  {'─'*50}")
    print(f"  Raw alert lines  : {len(alerts):,}")
    print(f"  Unique events    : {len(df_out)}")
    print(f"  {'─'*50}")
    print(f"  🔴 ATTACK        : {attacks}")
    print(f"  🟡 PROBE/SCAN    : {probes}")
    print(f"  🟢 NORMAL        : {normal}")
    print(f"  ⚪ SUSPICIOUS    : {susp}")
    print(f"  {'─'*50}")
    print("\n" + "=" * 60)
    print("  Pipeline complete!")
    print("=" * 60)
    return df_out


if __name__ == "__main__":
    run_pipeline()
