"""
main.py — Single entry point for the Hybrid IDS project.

RUN ORDER (first time only):
  1 → Preprocess Dataset
  2 → Train ML Model

EVERY ATTACK SESSION:
  (Ubuntu)  sudo tcpdump -i enp0s8 -w /media/sf_snortlogs/traffic.pcap
  (Ubuntu)  sudo snort -A fast -q -c /etc/snort/snort.conf -i enp0s8 -l /media/sf_snortlogs
  (Kali)    ping -c 5 192.168.56.101
  (Kali)    nmap -sS / nmap -sX / nmap -sN / nmap -A 192.168.56.101
  (Kali)    sudo ping -f 192.168.56.101
  (Ubuntu)  Ctrl+C on both Snort and tcpdump

THEN ON WINDOWS:
  3 → Run Hybrid Pipeline
  4 → Run PCAP Analysis
  5 → Launch Dashboard  → http://localhost:8501
"""

import os
import sys


def clear():
    os.system("cls" if os.name == "nt" else "clear")


def ensure_dirs():
    for d in ["models", "snortlogs", "data"]:
        os.makedirs(d, exist_ok=True)


def menu():
    print("=" * 60)
    print("  Hybrid IDS — Snort + Machine Learning")
    print("=" * 60)
    print()
    print("  ── First-time setup (run once) ──────────────────────────")
    print("  [1] Preprocess Dataset   (KDDTrain+.txt → processed_nslkdd.csv)")
    print("  [2] Train ML Model       (processed CSV → models/ids_model.pkl)")
    print()
    print("  ── Each attack session (after Snort + tcpdump on Ubuntu) ─")
    print("  [3] Run Hybrid Pipeline  (Snort alerts → ML → hybrid_results.csv)")
    print("  [4] Run PCAP Analysis    (traffic.pcap → Snort vs ML comparison)")
    print("  [5] Launch Dashboard     (http://localhost:8501)")
    print()
    print("  [0] Exit")
    print()
    return input("  Choose option: ").strip()


def main():
    ensure_dirs()

    while True:
        clear()
        choice = menu()

        if choice == "1":
            print()
            from preprocess_dataset import preprocess_and_save
            preprocess_and_save()
            input("\nPress Enter to continue ...")

        elif choice == "2":
            print()
            import subprocess
            subprocess.run([sys.executable, "train.py"], check=False)
            input("\nPress Enter to continue ...")

        elif choice == "3":
            print()
            from config import ALERT_FILE
            if not os.path.exists(ALERT_FILE):
                print(f"  Alert file not found: {ALERT_FILE}")
                print("  Start Snort on Ubuntu and perform an attack from Kali first.")
                input("\nPress Enter to continue ...")
                continue
            if os.path.getsize(ALERT_FILE) == 0:
                print(f"  Alert file is empty: {ALERT_FILE}")
                print("  Perform an attack from Kali first, then re-run.")
                input("\nPress Enter to continue ...")
                continue
            from hybrid_pipeline import run_pipeline
            run_pipeline(verbose=True)
            input("\nPress Enter to continue ...")

        elif choice == "4":
            print()
            from config import PCAP_FILE
            if not os.path.exists(PCAP_FILE):
                print(f"  PCAP file not found: {PCAP_FILE}")
                print("  On Ubuntu, run before attacking:")
                print("  sudo tcpdump -i enp0s8 -w /media/sf_snortlogs/traffic.pcap")
                input("\nPress Enter to continue ...")
                continue
            if os.path.getsize(PCAP_FILE) == 0:
                print(f"  PCAP file is empty: {PCAP_FILE}")
                print("  Start tcpdump BEFORE performing the attack.")
                input("\nPress Enter to continue ...")
                continue
            from pcap_analyzer import run_pcap_analysis
            run_pcap_analysis(verbose=True)
            input("\nPress Enter to continue ...")

        elif choice == "5":
            print()
            print("  Opening http://localhost:8501  (Ctrl+C to stop)\n")
            import subprocess
            subprocess.run(
                [sys.executable, "-m", "streamlit", "run", "dashboard.py"],
                check=False,
            )

        elif choice == "0":
            print("\n  Goodbye!\n")
            break

        else:
            input("\n  Invalid option. Press Enter ...")


if __name__ == "__main__":
    main()
