"""
dashboard.py  —  Hybrid IDS Dashboard
Run:  streamlit run dashboard.py

TWO TABS
  Tab 1 — Snort Alerts + ML Assessment
    Shows unique aggregated events (not raw alert lines).
    Four tiers from the 3-class calibrated model:
      🔴 ATTACK     → active exploit (conf ≥ 80%, model says high)
      🟡 PROBE/SCAN → reconnaissance, nmap (conf ≥ 80%, model says medium)
      🟢 NORMAL     → benign (conf ≥ 80%, model says low)
      ⚪ SUSPICIOUS → uncertain (conf < 80%)

  Tab 2 — PCAP Analysis — What Snort Missed
    All network flows from traffic.pcap:
      ✅ Snort Detected  → Snort had a rule
      🔴 ML: ATTACK      → Snort missed, ML says active attack
      🟡 ML: PROBE/SCAN  → Snort missed, ML says reconnaissance
      🟢 ML: Normal      → Snort missed, ML says normal
      ⚪ Suspicious      → Snort missed, ML uncertain
"""

import os
import json
import pandas as pd
import streamlit as st
from datetime import datetime

from config import METRICS_FILE, RESULTS_FILE, PCAP_RESULTS

st.set_page_config(
    page_title="Hybrid IDS Dashboard",
    page_icon="🛡️",
    layout="wide",
)

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("🛡️ Hybrid IDS")
    st.caption("Snort + Machine Learning (3-class model)")
    st.divider()

    st.subheader("⚙️ Run Actions")

    if st.button("🔄 Run Pipeline (Snort → ML)", use_container_width=True):
        with st.spinner("Reading Snort alerts and running ML predictions …"):
            from hybrid_pipeline import run_pipeline
            run_pipeline(verbose=False)
        st.success("Done! hybrid_results.csv updated.")
        st.rerun()

    if st.button("📦 Run PCAP Analysis", use_container_width=True):
        with st.spinner("Comparing pcap traffic vs Snort alerts …"):
            from pcap_analyzer import run_pcap_analysis
            run_pcap_analysis(verbose=False)
        st.success("Done! pcap_analysis_results.csv updated.")
        st.rerun()

    st.divider()

    st.subheader("📊 ML Model Performance")
    if os.path.exists(METRICS_FILE):
        with open(METRICS_FILE) as fh:
            mj = json.load(fh)

        rf = mj.get("models", {}).get("RandomForest", {})
        acc = rf.get("accuracy", 0)
        f1 = rf.get("f1_weighted", 0)

        st.metric("Accuracy", f"{acc:.2%}")
        st.metric("F1-Score", f"{f1:.2%}")
    else:
        st.warning("Model metrics not found. Run option 2 first.")

    st.divider()
    st.caption(f"Last loaded: {datetime.now().strftime('%H:%M:%S')}")




# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2 = st.tabs([
    "📡 Snort Alerts + ML Assessment",
    "📦 PCAP Analysis — What Snort Missed",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.title("📡 Snort Alerts + ML Severity Assessment")
    st.markdown(
        "Shows **aggregated unique events** — repeated alerts from the same source/type "
        "are grouped into one row with an `event_count`. "
        "The 3-class ML model classifies each event: "
        "🔴 **ATTACK**, 🟡 **PROBE/SCAN**, 🟢 **NORMAL**, or ⚪ **SUSPICIOUS**."
    )
    st.divider()

    if not os.path.exists(RESULTS_FILE):
        st.info(
            "No results yet.\n\n"
            "1. Start Snort on Ubuntu\n"
            "2. Perform attacks from Kali\n"
            "3. Click **🔄 Run Pipeline** in the sidebar"
        )
        st.stop()

    df = pd.read_csv(RESULTS_FILE)
    if df.empty:
        st.warning("hybrid_results.csv is empty. Run the pipeline after attacking.")
        st.stop()

    # Back-compat: rebuild ml_tier if old file format
    if "ml_tier" not in df.columns:
        def _rebuild(row):
            conf = row.get("ml_confidence", 0) or 0
            pred = row.get("ml_prediction", "low")
            if conf >= 0.80:
                if pred == "high":   return "attack"
                if pred == "medium": return "probe"
                return "normal"
            return "suspicious"
        df["ml_tier"] = df.apply(_rebuild, axis=1)

    total   = len(df)
    attacks = int((df["ml_tier"] == "attack").sum())
    probes  = int((df["ml_tier"] == "probe").sum())
    normal  = int((df["ml_tier"] == "normal").sum())
    susp    = int((df["ml_tier"] == "suspicious").sum())

    # Show total raw events if event_count column exists
    if "event_count" in df.columns:
        raw_total = int(df["event_count"].sum())
        st.caption(f"ℹ️ {raw_total:,} raw Snort alert lines aggregated into {total:,} unique events.")

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Unique Events",  total)
    k2.metric("🔴 ATTACK",      attacks)
    k3.metric("🟡 PROBE/SCAN",  probes)
    k4.metric("🟢 NORMAL",      normal)
    k5.metric("⚪ SUSPICIOUS",  susp)

    st.divider()

    ch1, ch2, ch3 = st.columns(3)
    with ch1:
        st.subheader("ML Verdict")
        tier_counts = df["ml_tier"].value_counts().rename(index={
            "attack":     "🔴 ATTACK",
            "probe":      "🟡 PROBE/SCAN",
            "normal":     "🟢 NORMAL",
            "suspicious": "⚪ SUSPICIOUS",
        })
        st.bar_chart(tier_counts)

    with ch2:
        st.subheader("Protocol Distribution")
        st.bar_chart(df["protocol"].value_counts())

    with ch3:
        st.subheader("Top Source IPs")
        st.bar_chart(df["src_ip"].value_counts().head(10))

    st.divider()
    st.subheader("📋 Event Detail")

    flt = st.selectbox("Filter:", [
        "All events",
        "🔴 ATTACK only",
        "🟡 PROBE/SCAN only",
        "🟢 NORMAL only",
        "⚪ SUSPICIOUS only",
    ])
    disp = df.copy()
    if flt == "🔴 ATTACK only":
        disp = disp[disp["ml_tier"] == "attack"]
    elif flt == "🟡 PROBE/SCAN only":
        disp = disp[disp["ml_tier"] == "probe"]
    elif flt == "🟢 NORMAL only":
        disp = disp[disp["ml_tier"] == "normal"]
    elif flt == "⚪ SUSPICIOUS only":
        disp = disp[disp["ml_tier"] == "suspicious"]

    show_cols = [c for c in [
        "timestamp", "ml_label", "ml_confidence", "event_count",
        "alert_name", "priority", "protocol", "src_dst", "classification",
    ] if c in disp.columns]

    st.dataframe(disp[show_cols], use_container_width=True, hide_index=True)
    st.caption(f"Showing {len(disp):,} of {total:,} unique events")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.title("📦 PCAP Analysis — What Snort Missed")
    st.markdown(
        "`traffic.pcap` captures **every packet** — not just what Snort caught. "
        "This tab shows all network flows, which Snort detected, and which it missed. "
        "The 3-class ML model classifies every flow, and highlights missed flows as "
        "🔴 **ATTACK**, 🟡 **PROBE/SCAN**, 🟢 **Normal**, or ⚪ **Suspicious**."
    )
    st.divider()

    if not os.path.exists(PCAP_RESULTS):
        st.info(
            "No PCAP analysis results yet.\n\n"
            "1. Ubuntu Terminal 1: `sudo tcpdump -i enp0s8 -w /media/sf_snortlogs/traffic.pcap`\n"
            "2. Ubuntu Terminal 2: start Snort\n"
            "3. Kali: run attacks\n"
            "4. Stop both, then click **📦 Run PCAP Analysis** in the sidebar"
        )
        st.stop()

    df2 = pd.read_csv(PCAP_RESULTS)
    if df2.empty:
        st.warning("pcap_analysis_results.csv is empty.")
        st.stop()

    # Back-compat
    if "ml_tier" not in df2.columns:
        def _rebuild2(row):
            if row.get("snort_caught", False):
                return "snort"
            conf = row.get("ml_confidence") or 0
            pred = row.get("ml_prediction", "low")
            if conf >= 0.80:
                if pred == "high":   return "attack"
                if pred == "medium": return "probe"
                return "normal"
            return "suspicious"
        df2["ml_tier"] = df2.apply(_rebuild2, axis=1)

    total_flows  = len(df2)
    snort_caught = int(df2["snort_caught"].sum())
    snort_missed = total_flows - snort_caught

    missed_df = df2[df2["snort_caught"] == False].copy()
    ml_attack = int((missed_df["ml_tier"] == "attack").sum()) if not missed_df.empty else 0
    ml_probe  = int((missed_df["ml_tier"] == "probe").sum()) if not missed_df.empty else 0
    ml_normal = int((missed_df["ml_tier"] == "normal").sum()) if not missed_df.empty else 0
    ml_susp   = int((missed_df["ml_tier"] == "suspicious").sum()) if not missed_df.empty else 0

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Total Flows",        total_flows)
    k2.metric("✅ Snort Detected",  snort_caught)
    k3.metric("⚠️ Snort Missed",    snort_missed)
    k4.metric("🔴 ML: ATTACK",      ml_attack)
    k5.metric("🟡 ML: PROBE/SCAN",  ml_probe)
    k6.metric("⚪ ML: SUSPICIOUS",  ml_susp)

    st.divider()

    v1, v2 = st.columns(2)
    with v1:
        st.subheader("Snort Coverage")
        miss_pct = snort_missed / total_flows if total_flows else 0
        st.caption(
            f"Snort detected {snort_caught:,} of {total_flows:,} flows "
            f"({snort_caught/total_flows:.0%}). "
            f"{snort_missed:,} flows ({miss_pct:.0%}) had no matching rule."
        )
        st.bar_chart(pd.Series({
            "✅ Snort Detected": snort_caught,
            "⚠️ Snort Missed":   snort_missed,
        }))

    with v2:
        st.subheader("ML Classification of Missed Flows")
        if snort_missed > 0:
            st.bar_chart(pd.Series({
                "🔴 ML: ATTACK":    ml_attack,
                "🟡 ML: PROBE":     ml_probe,
                "🟢 ML: Normal":    ml_normal,
                "⚪ Suspicious":    ml_susp,
            }))
        else:
            st.success("Snort detected all flows.")

    st.divider()

    # Threats Snort missed
    st.subheader("🚨 Threats Snort Missed — Caught by ML")
    missed_threats = df2[(df2["snort_caught"] == False) & (df2["ml_tier"].isin(["attack", "probe"]))].copy()

    if missed_threats.empty:
        st.success("No threat flows beyond what Snort already detected.")
    else:
        ml_a = int((missed_threats["ml_tier"] == "attack").sum())
        ml_p = int((missed_threats["ml_tier"] == "probe").sum())
        st.error(
            f"**{len(missed_threats):,} flows** were invisible to Snort but flagged by ML: "
            f"{ml_a} active attacks, {ml_p} probe/scan flows."
        )
        show_cols = [c for c in [
            "timestamp", "ml_label", "ml_confidence", "src_ip", "src_port",
            "dst_ip", "dst_port", "protocol", "service", "flag", "pkt_count",
        ] if c in missed_threats.columns]
        st.dataframe(
            missed_threats[show_cols].sort_values("ml_confidence", ascending=False),
            use_container_width=True, hide_index=True,
        )

    st.divider()

    # Suspicious
    st.subheader("⚪ Suspicious Flows (ML uncertain — conf < 80%)")
    susp_flows = df2[(df2["snort_caught"] == False) & (df2["ml_tier"] == "suspicious")].copy()
    if susp_flows.empty:
        st.info("No suspicious flows.")
    else:
        st.warning(
            f"**{len(susp_flows):,} flows** Snort missed and ML is uncertain about. "
            f"Worth manual review."
        )
        show_s = [c for c in [
            "timestamp", "src_ip", "dst_ip", "dst_port",
            "protocol", "service", "flag", "pkt_count", "ml_confidence",
        ] if c in susp_flows.columns]
        st.dataframe(susp_flows[show_s].head(100), use_container_width=True, hide_index=True)

    st.divider()

    st.subheader("📋 All Flows from PCAP")
    flt2 = st.selectbox("Show:", [
        "All flows",
        "✅ Snort Detected only",
        "🔴 ML: ATTACK (Snort Missed)",
        "🟡 ML: PROBE/SCAN (Snort Missed)",
        "🟢 ML: Normal (Snort Missed)",
        "⚪ Suspicious (Snort Missed)",
    ])
    df2_disp = df2.copy()
    if flt2 == "✅ Snort Detected only":
        df2_disp = df2_disp[df2_disp["snort_caught"] == True]
    elif flt2 == "🔴 ML: ATTACK (Snort Missed)":
        df2_disp = df2_disp[(df2_disp["snort_caught"] == False) & (df2_disp["ml_tier"] == "attack")]
    elif flt2 == "🟡 ML: PROBE/SCAN (Snort Missed)":
        df2_disp = df2_disp[(df2_disp["snort_caught"] == False) & (df2_disp["ml_tier"] == "probe")]
    elif flt2 == "🟢 ML: Normal (Snort Missed)":
        df2_disp = df2_disp[(df2_disp["snort_caught"] == False) & (df2_disp["ml_tier"] == "normal")]
    elif flt2 == "⚪ Suspicious (Snort Missed)":
        df2_disp = df2_disp[(df2_disp["snort_caught"] == False) & (df2_disp["ml_tier"] == "suspicious")]

    show2 = [c for c in [
        "timestamp", "ml_label", "src_ip", "src_port", "dst_ip", "dst_port",
        "protocol", "service", "flag", "pkt_count", "snort_caught", "ml_confidence",
    ] if c in df2_disp.columns]
    st.dataframe(df2_disp[show2], use_container_width=True, hide_index=True)
    st.caption(f"Showing {len(df2_disp):,} of {total_flows:,} total flows")
