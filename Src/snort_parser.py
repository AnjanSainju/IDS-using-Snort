"""
snort_parser.py
───────────────
WHAT THIS FILE DOES
  Parses Snort fast-alert log lines into structured Python dicts.

SNORT FAST-ALERT FORMAT (one line per matched packet)
  04/12-08:49:47.959155  [**] [1:1418:11] SNMP request tcp [**]
    [Classification: Attempted Information Leak] [Priority: 2]
    {TCP} 192.168.56.102:48223 -> 192.168.56.101:161

  Fields we extract:
    timestamp      "04/12-08:49:47.959155"
    alert_name     "SNMP request tcp"
    classification "Attempted Information Leak"
    priority       2  (int; 1 = most severe, 3+ = informational)
    protocol       "TCP"
    src_ip         "192.168.56.102"
    src_port       "48223"
    dst_ip         "192.168.56.101"
    dst_port       "161"
    src_dst        "192.168.56.102:48223 -> 192.168.56.101:161"

HOW IT IS USED
  hybrid_pipeline.py calls read_alerts() to get all alerts logged so far,
  then feeds each alert to the ML model for a second-opinion severity rating.

  The alert log grows as Snort detects new packets.  Re-calling read_alerts()
  at any time returns the full current list.
"""

import re
import os
from config import ALERT_FILE


def parse_alert_line(line: str) -> dict | None:
    """Parse one Snort fast-alert line into a dict, or return None if unparseable."""
    line = line.strip()
    if not line:
        return None

    # Timestamp:  04/12-08:49:32.029758
    ts_match = re.match(r'^(\d{2}/\d{2}-\d{2}:\d{2}:\d{2}\.\d+)', line)
    if not ts_match:
        return None
    timestamp = ts_match.group(1)

    # Alert name:  text between [**] [sid:...] and the closing [**]
    name_match = re.search(r'\[\*\*\]\s*\[[\d:]+\]\s*(.*?)\s*\[\*\*\]', line)
    if not name_match:
        return None
    alert_name = name_match.group(1).strip()

    # Classification (optional)
    class_match = re.search(r'\[Classification:\s*(.*?)\]', line)
    classification = class_match.group(1).strip() if class_match else "No Classification"

    # Priority
    pri_match = re.search(r'\[Priority:\s*(\d+)\]', line)
    priority = int(pri_match.group(1)) if pri_match else 3

    # Protocol:  {TCP} / {UDP} / {ICMP}
    proto_match = re.search(r'\{(\w+)\}', line)
    protocol = proto_match.group(1).upper() if proto_match else "TCP"

    # IPs and ports:  192.168.56.102:48223 -> 192.168.56.101:22
    #             or  192.168.56.102 -> 192.168.56.101  (no ports for ICMP)
    ip_match = re.search(
        r'(\d+\.\d+\.\d+\.\d+)(?::(\d+))?\s*->\s*(\d+\.\d+\.\d+\.\d+)(?::(\d+))?',
        line,
    )
    if ip_match:
        src_ip   = ip_match.group(1)
        src_port = ip_match.group(2) or ""
        dst_ip   = ip_match.group(3)
        dst_port = ip_match.group(4) or ""
        src_dst  = (
            f"{src_ip}{':{}'.format(src_port) if src_port else ''}"
            " -> "
            f"{dst_ip}{':{}'.format(dst_port) if dst_port else ''}"
        )
    else:
        src_ip = dst_ip = src_port = dst_port = src_dst = "unknown"

    return {
        "timestamp":      timestamp,
        "alert_name":     alert_name,
        "priority":       priority,
        "classification": classification,
        "protocol":       protocol,
        "src_ip":         src_ip,
        "src_port":       src_port,
        "dst_ip":         dst_ip,
        "dst_port":       dst_port,
        "src_dst":        src_dst,
        "raw":            line,
    }


def read_alerts(filepath: str = ALERT_FILE) -> list[dict]:
    """Read and parse every alert in a Snort fast-alert log file."""
    if not os.path.exists(filepath):
        return []
    alerts = []
    with open(filepath, "r", errors="replace") as fh:
        for line in fh:
            parsed = parse_alert_line(line)
            if parsed:
                alerts.append(parsed)
    return alerts


def get_recent_alerts(since_timestamp: str | None = None) -> list[dict]:
    """Return all alerts, optionally filtered to those after since_timestamp."""
    alerts = read_alerts()
    if since_timestamp:
        return [a for a in alerts if a["timestamp"] > since_timestamp]
    return alerts


if __name__ == "__main__":
    alerts = read_alerts()
    print(f"Total alerts parsed: {len(alerts)}")
    for a in alerts[:5]:
        lbl = "🔴" if a["priority"] <= 2 else "🟡"
        print(f"  {lbl} [{a['timestamp']}] {a['alert_name']} | "
              f"{a['protocol']} | {a['src_dst']} | P:{a['priority']}")
