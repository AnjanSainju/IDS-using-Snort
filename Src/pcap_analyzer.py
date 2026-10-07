"""
pcap_analyzer.py
────────────────
HOW THIS FILE WORKS:

PURPOSE: Analyze ALL network traffic from traffic.pcap.
         Compare what Snort detected vs what Snort missed.
         Run ML on both Snort-detected and Snort-missed flows.
         Show the detection gap between signature-only and hybrid approach.

COMPARISON LOGIC:
  1. Parse traffic.pcap → extract all packets
  2. Group packets into flows: (src_ip, dst_ip, protocol, dst_port, flag)
  3. Load Snort alert log → build set of (src_ip, dst_ip, dst_port, protocol) tuples
  4. For each flow:
     - If flow signature is in Snort set → snort_caught=True
     - If flow signature NOT in Snort set → snort_caught=False
  5. Run ML on ALL flows (both caught and missed)
  6. Aggregate similar flows into unique events for cleaner output

CLASSIFICATION TIERS (same as hybrid_pipeline.py):
  🔴 ATTACK     — pred=high   AND conf ≥ 50%
  🟡 PROBE/SCAN — pred=medium AND conf ≥ 80%
  🟢 NORMAL     — pred=low    AND conf ≥ 80%
  ⚪ SUSPICIOUS — below threshold (uncertain)

FEATURE EXTRACTION FROM PACKETS:
  Unlike hybrid_pipeline.py which estimates features from alert metadata,
  this file computes REAL features from actual packet measurements:
    serror_rate = SYN-only packets / total packets (real S0 flag count)
    diff_srv_rate = unique ports / total flows (real port diversity)
    src_bytes = actual payload bytes measured
    duration = actual timestamp difference
  These real features make pcap_analyzer more accurate than hybrid_pipeline.
"""

from __future__ import annotations

import math
import os
import re
from collections import Counter, defaultdict
from datetime import datetime

import joblib
import pandas as pd

from config import ALERT_FILE, FEATURES_FILE, MODEL_FILE, PCAP_FILE, PCAP_RESULTS

# Consistent with hybrid_pipeline.py thresholds
ATTACK_THRESHOLD  = 0.50
PROBE_THRESHOLD   = 0.80
NORMAL_THRESHOLD  = 0.80

PORT_TO_SERVICE = {
    20:"ftp_data", 21:"ftp",    22:"ssh",    23:"telnet", 25:"smtp",
    53:"domain",   80:"http",   110:"pop_3", 111:"sunrpc",123:"ntp_u",
    135:"epmap",   137:"netbios_ns", 138:"netbios_dgm", 139:"netbios_ssn",
    143:"imap4",   161:"snmp",  162:"snmp_trap", 179:"bgp",
    389:"ldap",    443:"http_443", 445:"microsoft_ds", 465:"smtp",
    514:"shell",   515:"printer", 520:"efs", 543:"klogin", 544:"kshell",
    587:"smtp",    993:"imap4", 995:"pop_3", 1433:"sql_net", 1521:"sql_net",
    3306:"sql_net",3389:"other",5900:"other", 6000:"X11",
    8080:"http_8001", 8443:"http_443",
}


def safe_int(value, default=0):
    try:
        return int(str(value))
    except Exception:
        return default


def classify_flow(prediction: str, confidence: float) -> str:
    """
    Apply thresholds to produce classification tier.
    Consistent with hybrid_pipeline.py classification logic.
    """
    pred = str(prediction).lower()
    if pred == "high"   and confidence >= ATTACK_THRESHOLD: return "attack"
    if pred == "medium" and confidence >= PROBE_THRESHOLD:  return "probe"
    if pred == "low"    and confidence >= NORMAL_THRESHOLD: return "normal"
    return "suspicious"


def get_label(tier: str) -> str:
    labels = {
        "attack":     "🔴 ATTACK",
        "probe":      "🟡 PROBE/SCAN",
        "normal":     "🟢 NORMAL",
        "suspicious": "⚪ SUSPICIOUS",
    }
    return labels.get(tier, str(tier).upper())


def tcp_flags_to_nslkdd(flags_int: int) -> str:
    """Convert TCP flags bitmask to NSL-KDD flag label."""
    syn = flags_int & 0x02
    rst = flags_int & 0x04
    fin = flags_int & 0x01
    ack = flags_int & 0x10
    if syn and not ack and not fin: return "S0"    # SYN only = scan
    if syn and ack:                 return "SF"    # SYN+ACK = completed
    if rst and ack:                 return "RSTO"  # RST+ACK = reset
    if rst and not ack:             return "REJ"   # RST only = rejected
    if fin and ack:                 return "SF"
    if syn and fin:                 return "SH"
    return "OTH"


def parse_pcap(path: str) -> list[dict]:
    """Parse traffic.pcap with Scapy, extract per-packet metadata."""
    try:
        from scapy.all import ICMP, IP, TCP, UDP, rdpcap
    except Exception:
        print("ERROR: scapy is not installed. Run: pip install scapy")
        return []
    try:
        packets = rdpcap(path)
    except Exception as exc:
        print(f"ERROR reading pcap: {exc}")
        return []

    rows = []
    for pkt in packets:
        try:
            if not pkt.haslayer(IP): continue
            ts     = float(pkt.time)
            src_ip = pkt[IP].src
            dst_ip = pkt[IP].dst
            if pkt.haslayer(TCP):
                proto    = "TCP"
                src_port = int(pkt[TCP].sport)
                dst_port = int(pkt[TCP].dport)
                flag     = tcp_flags_to_nslkdd(int(pkt[TCP].flags))
                payload  = len(bytes(pkt[TCP].payload))
            elif pkt.haslayer(UDP):
                proto    = "UDP"
                src_port = int(pkt[UDP].sport)
                dst_port = int(pkt[UDP].dport)
                flag     = "SF"
                payload  = len(bytes(pkt[UDP].payload))
            elif pkt.haslayer(ICMP):
                proto    = "ICMP"
                src_port = 0
                dst_port = 0
                flag     = "SF"
                payload  = len(bytes(pkt[ICMP].payload))
            else:
                continue
            rows.append({
                "timestamp": datetime.fromtimestamp(ts).strftime("%m/%d-%H:%M:%S.%f")[:20],
                "ts_raw":    ts,
                "src_ip":    src_ip,
                "dst_ip":    dst_ip,
                "src_port":  src_port,
                "dst_port":  dst_port,
                "protocol":  proto,
                "flag":      flag,
                "payload":   payload,
            })
        except Exception:
            continue
    return rows


def build_flow_rows(packets: list[dict]) -> list[dict]:
    """
    Group packets into flows by (src_ip, dst_ip, protocol, dst_port, flag).
    Each flow represents a distinct connection type from one host to another.
    """
    grouped = defaultdict(list)
    for pkt in packets:
        key = (pkt["src_ip"], pkt["dst_ip"], pkt["protocol"], pkt["dst_port"], pkt["flag"])
        grouped[key].append(pkt)

    flow_rows = []
    for (src_ip, dst_ip, protocol, dst_port, flag), pkt_group in grouped.items():
        timestamps = sorted(p["ts_raw"] for p in pkt_group)
        duration   = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0.0
        src_port_mode = Counter(p["src_port"] for p in pkt_group).most_common(1)[0][0]
        flow_rows.append({
            "timestamp":  pkt_group[0]["timestamp"],
            "src_ip":     src_ip,
            "dst_ip":     dst_ip,
            "src_port":   src_port_mode,
            "dst_port":   dst_port,
            "protocol":   protocol,
            "service":    PORT_TO_SERVICE.get(dst_port, "other"),
            "flag":       flag,
            "pkt_count":  len(pkt_group),
            "src_bytes":  sum(p["payload"] for p in pkt_group),
            "duration_s": duration,
            "s0_count":   sum(1 for p in pkt_group if p["flag"] == "S0"),
        })
    return flow_rows


def load_snort_signatures(path: str) -> set[tuple]:
    """
    Parse Snort alert log to build set of detected flow signatures.
    Each signature: (src_ip, dst_ip, dst_port, protocol)
    Used to match against PCAP flows to determine what Snort caught.
    """
    if not os.path.exists(path):
        return set()
    signatures = set()
    ip_pat    = re.compile(r"(\d+\.\d+\.\d+\.\d+)(?::(\d+))?\s*->\s*(\d+\.\d+\.\d+\.\d+)(?::(\d+))?")
    proto_pat = re.compile(r"\{(\w+)\}")
    with open(path, "r", errors="replace") as fh:
        for line in fh:
            ip_m    = ip_pat.search(line)
            proto_m = proto_pat.search(line)
            if ip_m and proto_m:
                signatures.add((
                    ip_m.group(1),
                    ip_m.group(3),
                    safe_int(ip_m.group(4), 0),
                    proto_m.group(1).upper(),
                ))
    return signatures


def one_hot_assign(vec: dict, feature_names: list[str], target: str) -> None:
    """Set a one-hot column to 1.0 by case-insensitive match."""
    target_lower = target.lower()
    for col in feature_names:
        if col.lower() == target_lower:
            vec[col] = 1.0
            return


def flow_to_features(flow: dict, pair_stats: dict, feature_names: list[str]) -> pd.DataFrame:
    """
    Convert a real PCAP flow into a 51-feature NSL-KDD vector.
    Uses actual packet measurements — more accurate than hybrid_pipeline's estimates.

    Real features computed:
      serror_rate: actual fraction of S0 (SYN-only) packets in this flow
      diff_srv_rate: actual port diversity relative to all flows from same src→dst
      src_bytes: actual total payload bytes
      duration: actual time from first to last packet
    """
    vec = {name: 0.0 for name in feature_names}

    # One-hot: protocol
    proto = flow["protocol"].upper()
    if proto == "ICMP":   one_hot_assign(vec, feature_names, "protocol_type_icmp")
    elif proto == "UDP":  one_hot_assign(vec, feature_names, "protocol_type_udp")
    else:                 one_hot_assign(vec, feature_names, "protocol_type_tcp")

    # One-hot: service from dst_port
    service = flow["service"].lower()
    one_hot_assign(vec, feature_names, f"service_{service}")
    if all(col.lower() != f"service_{service}" for col in feature_names):
        one_hot_assign(vec, feature_names, "service_other")

    # One-hot: flag
    one_hot_assign(vec, feature_names, f"flag_{flow['flag'].lower()}")

    # Real numeric features from actual packet data
    pair_key = (flow["src_ip"], flow["dst_ip"], flow["protocol"])
    stats    = pair_stats.get(pair_key, {"total": 1, "unique_ports": 1, "same_service": 1})
    total    = max(1, stats["total"])
    unique_ports  = max(1, stats["unique_ports"])
    same_service  = max(1, stats["same_service"])

    if "duration"         in vec: vec["duration"]         = min(3.0, flow["duration_s"] / 2.0)
    if "src_bytes"        in vec: vec["src_bytes"]        = min(3.0, math.log1p(max(flow["src_bytes"], 0)) / 4.0)
    if "count"            in vec: vec["count"]            = min(3.0, math.log1p(total))
    if "srv_count"        in vec: vec["srv_count"]        = min(3.0, math.log1p(same_service))
    if "dst_host_count"   in vec: vec["dst_host_count"]   = min(3.0, math.log1p(total))
    if "dst_host_srv_count" in vec: vec["dst_host_srv_count"] = min(3.0, math.log1p(same_service))

    # Real rate features
    same_ratio = same_service / float(total)
    diff_ratio = unique_ports / float(total)
    if "same_srv_rate" in vec: vec["same_srv_rate"] = (same_ratio - 0.5) * 2.0
    if "diff_srv_rate" in vec: vec["diff_srv_rate"] = (diff_ratio - 0.2) * 2.0

    # Real SYN error rate from actual flag counts
    if flow["pkt_count"] > 0:
        serror = flow["s0_count"] / float(flow["pkt_count"])
        if "serror_rate"     in vec: vec["serror_rate"]     = (serror - 0.25) * 3.0
        if "srv_serror_rate" in vec: vec["srv_serror_rate"] = (serror - 0.25) * 3.0

    return pd.DataFrame([vec], columns=feature_names)


def aggregate_event_rows(rows: list[dict]) -> pd.DataFrame:
    """
    Group similar flows into unique events for cleaner output.
    Groups by (src_ip, dst_ip, protocol, service, flag, snort_caught, ml_tier).
    """
    grouped = defaultdict(list)
    for row in rows:
        key = (
            row["src_ip"], row["dst_ip"], row["protocol"],
            row["service"], row["flag"],
            row["snort_caught"], row["ml_tier"], row["ml_prediction"],
        )
        grouped[key].append(row)

    events = []
    for _, group in grouped.items():
        first    = group[0]
        dst_ports = sorted({safe_int(r["dst_port"], 0) for r in group})
        avg_conf  = [r["ml_confidence"] for r in group if r["ml_confidence"] is not None]
        mean_conf = round(sum(avg_conf) / len(avg_conf), 4) if avg_conf else None
        events.append({
            "timestamp":       min(r["timestamp"] for r in group),
            "src_ip":          first["src_ip"],
            "src_port":        first["src_port"],
            "dst_ip":          first["dst_ip"],
            "dst_port":        dst_ports[0] if dst_ports else 0,
            "protocol":        first["protocol"],
            "service":         first["service"],
            "flag":            first["flag"],
            "pkt_count":       sum(r["pkt_count"] for r in group),
            "flow_count":      len(group),
            "snort_caught":    first["snort_caught"],
            "source":          "Snort+ML" if first["snort_caught"] else "ML Only (Snort Missed)",
            "ml_prediction":   first["ml_prediction"],
            "ml_confidence":   mean_conf,
            "ml_label":        first["ml_label"],
            "ml_tier":         first["ml_tier"],
            "unique_dst_ports": len(dst_ports),
            "dst_ports_preview": ",".join(str(p) for p in dst_ports[:12]),
        })

    df = pd.DataFrame(events)
    if not df.empty:
        df = df.sort_values(["timestamp","flow_count"], ascending=[False,False]).reset_index(drop=True)
    return df


def run_pcap_analysis(verbose: bool = True) -> pd.DataFrame:
    print("=" * 68)
    print("  PCAP Analyzer — Snort Coverage vs ML Classification")
    print(f"  Attack threshold: {ATTACK_THRESHOLD:.0%}  "
          f"Probe/Normal threshold: {PROBE_THRESHOLD:.0%}")
    print("=" * 68)

    if not os.path.exists(MODEL_FILE) or not os.path.exists(FEATURES_FILE):
        print("  ERROR: model artefacts missing. Run option 2 first.")
        return pd.DataFrame()
    if not os.path.exists(PCAP_FILE):
        print(f"  ERROR: pcap file not found: {PCAP_FILE}")
        return pd.DataFrame()

    # Load model
    model         = joblib.load(MODEL_FILE)
    feature_names = joblib.load(FEATURES_FILE)
    print(f"\n[1/5] Model loaded. Classes: {list(model.classes_)}  Features: {len(feature_names)}")

    # Parse PCAP
    print(f"\n[2/5] Reading PCAP: {PCAP_FILE}")
    packets = parse_pcap(PCAP_FILE)
    if not packets:
        print("  No packets parsed.")
        return pd.DataFrame()
    print(f"      Total packets: {len(packets):,}")

    # Build flows
    print(f"\n[3/5] Building flows from {len(packets):,} packets ...")
    flow_rows = build_flow_rows(packets)
    print(f"      Unique flow signatures: {len(flow_rows):,}")

    # Load Snort signatures for comparison
    print(f"\n[4/5] Loading Snort signatures: {ALERT_FILE}")
    snort_signatures = load_snort_signatures(ALERT_FILE)
    print(f"      Snort alert signatures: {len(snort_signatures):,}")

    # Build per-pair statistics for real feature computation
    pair_group = defaultdict(list)
    for flow in flow_rows:
        pair_group[(flow["src_ip"], flow["dst_ip"], flow["protocol"])].append(flow)
    pair_stats = {
        key: {
            "total":       len(vals),
            "unique_ports": len({v["dst_port"] for v in vals}),
            "same_service": max(Counter(v["service"] for v in vals).values()),
        }
        for key, vals in pair_group.items()
    }

    # Classify ALL flows with ML, compare each against Snort signatures
    print(f"\n[5/5] Classifying {len(flow_rows):,} flows with ML ...")
    classified_rows = []
    for flow in flow_rows:
        # Compare flow against Snort alert set
        signature   = (flow["src_ip"], flow["dst_ip"], flow["dst_port"], flow["protocol"])
        snort_caught = signature in snort_signatures

        # Run ML on this flow using real packet features
        features    = flow_to_features(flow, pair_stats, feature_names)
        prediction  = model.predict(features)[0]
        probs       = model.predict_proba(features)[0]
        confidence  = round(float(max(probs)), 4)
        tier        = classify_flow(prediction, confidence)

        classified_rows.append({
            **flow,
            "snort_caught":    snort_caught,
            "ml_prediction":   prediction,
            "ml_confidence":   confidence,
            "ml_tier":         tier,
            "ml_label":        get_label(tier),
        })

    # Aggregate into unique events
    events_df = aggregate_event_rows(classified_rows)
    events_df.to_csv(PCAP_RESULTS, index=False)

    # Summary statistics
    total_events  = len(events_df)
    caught_events = int(events_df["snort_caught"].sum()) if not events_df.empty else 0
    missed_events = total_events - caught_events
    missed_only   = events_df[~events_df["snort_caught"]] if not events_df.empty else pd.DataFrame()

    ml_attack  = int((missed_only["ml_tier"] == "attack").sum())    if not missed_only.empty else 0
    ml_probe   = int((missed_only["ml_tier"] == "probe").sum())     if not missed_only.empty else 0
    ml_normal  = int((missed_only["ml_tier"] == "normal").sum())    if not missed_only.empty else 0
    ml_susp    = int((missed_only["ml_tier"] == "suspicious").sum())if not missed_only.empty else 0

    print(f"\n  {'─'*66}")
    print(f"  Raw packets parsed                  : {len(packets):,}")
    print(f"  Raw flow signatures                 : {len(flow_rows):,}")
    print(f"  Unique events (after aggregation)   : {total_events:,}")
    print(f"  {'─'*66}")
    print(f"  ✅ Snort detected events             : {caught_events:,}")
    print(f"  ⚠️  Snort missed events               : {missed_events:,}")
    print(f"  {'─'*66}")
    print(f"  Of the Snort-missed events, ML says:")
    print(f"    🔴 ATTACK     : {ml_attack:,}")
    print(f"    🟡 PROBE/SCAN : {ml_probe:,}")
    print(f"    🟢 NORMAL     : {ml_normal:,}")
    print(f"    ⚪ SUSPICIOUS : {ml_susp:,}")
    print(f"  {'─'*66}")
    print(f"  Saved: {PCAP_RESULTS}")
    print("\n" + "=" * 68)
    print("  PCAP analysis complete!")
    print("=" * 68)
    return events_df


if __name__ == "__main__":
    run_pcap_analysis(verbose=True)
