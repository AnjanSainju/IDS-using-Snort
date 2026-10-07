# Intrusion Detection System Using Snort 

A Intrusion Detection System (IDS) that combines **Snort signature-based detection** with **Machine Learning** to identify and classify suspicious network activity. The system also performs **PCAP traffic analysis** to identify network flows that may not have been detected by Snort and provides a **Streamlit dashboard** for monitoring and visualization.

## Overview

The system combines traditional rule-based intrusion detection with machine learning-based network traffic analysis.

* **Snort** detects known threats using signature-based rules.
* **Random Forest** classifies network activity based on extracted traffic features.
* **PCAP analysis** examines captured network traffic and identifies potentially missed flows.
* **Streamlit dashboard** provides visualizations and summaries of alerts, classifications, protocols, source IPs, and missed traffic.
* The machine learning model is trained using the **NSL-KDD dataset**.

## Key Features

* Signature-based intrusion detection using Snort
* Machine learning-based network traffic classification
* Random Forest classification model
* PCAP traffic and flow analysis
* Identification of potentially missed network flows
* Snort alert processing and classification
* Network protocol distribution analysis
* Source IP activity analysis
* ML confidence scoring
* Interactive Streamlit dashboard
* End-to-end network monitoring workflow

## System Architecture


                    Network Traffic
                          |
             +------------+------------+
             |                         |
          Snort                    tcpdump
             |                         |
      Snort Alerts                  PCAP
             |                         |
             +------------+------------+
                          |
                   Feature Extraction
                          |
                    Random Forest
                     ML Model
                          |
             +------------+------------+
             |                         |
      Alert Classification       PCAP Flow Analysis
             |                         |
             +------------+------------+
                          |
                 Streamlit Dashboard
                          |
        +-----------------+-----------------+
        |                 |                 |
   Alert Analysis    ML Results      Missed Flows


## Detection Workflow

1. Network traffic is generated and captured in an isolated environment.
2. Snort monitors the traffic and generates alerts for traffic matching configured rules.
3. tcpdump captures network packets for PCAP-based analysis.
4. Snort alerts are processed and relevant features are extracted.
5. The Random Forest model classifies network activity into traffic categories.
6. PCAP files are analyzed to reconstruct network flows and identify potentially missed traffic.
7. The results are presented through the Streamlit dashboard.

## Machine Learning

The machine learning component uses a **Random Forest classifier** trained using the **NSL-KDD dataset**.

The model classifies network activity into categories such as:

* Normal
* Attack
* Probe / Scan
* Suspicious

The model also provides confidence scores for its classifications.

## PCAP Analysis

The PCAP analysis module examines captured network traffic and extracts flow-level characteristics such as:

* Protocol
* Packet count
* Flow duration
* SYN error rate
* Port diversity
* Payload size
* Source and destination information

These features are passed through the same machine learning pipeline to identify suspicious traffic that may not have generated a Snort alert.

## Streamlit Dashboard

The Streamlit dashboard provides an interactive interface for analyzing IDS results.

It includes:

* Snort alert summaries
* Machine learning classifications
* Protocol distribution
* Top source IP addresses
* Snort detection coverage
* Missed-flow analysis
* Traffic summaries
* Interactive visualizations

## Technologies Used

* Python
* Snort
* tcpdump
* Scapy
* Pandas
* Scikit-learn
* Joblib
* Streamlit
* Ubuntu
* Kali Linux
* Oracle VirtualBox
* NSL-KDD Dataset



## Installation

Clone the repository:

bash
git clone <your-repository-url>
cd <your-repository-name>


Create a Python virtual environment:

bash
python3 -m venv venv


Activate the virtual environment:

bash
source venv/bin/activate


Install the required Python packages:

bash
pip install -r requirements.txt


Make sure Snort and tcpdump are installed and configured on the monitoring system.

## Running the System

Start Snort and configure it to monitor the required network interface.

Capture network traffic using tcpdump:

bash
sudo tcpdump -i <interface> -w capture.pcap


Run the hybrid detection pipeline:

bash
python hybrid_pipeline.py


Run PCAP analysis:

bash
python pcap_analyzer.py


Start the Streamlit dashboard:

bash
streamlit run dashboard.py




## Testing

The system was tested across the main stages of the detection workflow, including:

* Snort startup and alert generation
* Network packet capture
* Attack traffic generation
* Machine learning classification
* PCAP analysis
* Missed-flow identification
* Dashboard loading and visualization
* End-to-end workflow testing

## Limitations

* The NSL-KDD dataset is relatively old and may not represent modern network traffic.
* Some features used in the Snort alert pipeline are estimated rather than directly extracted from packets.
* PCAP analysis operates in batch mode rather than real time.
* Snort detection coverage depends on the configured rule set.
* The system is designed as a prototype and is not intended to replace a production Security Operations Center (SOC).

## Future Improvements

* Real-time network monitoring
* Larger and more diverse training datasets
* Improved network feature engineering
* Advanced ensemble and deep learning models
* Docker-based deployment
* Cloud deployment
* Improved model explainability
* Better scalability for large network environments

## Security Notice

Use this project only in networks and environments where you have explicit authorization to capture and analyze traffic. Attack traffic should only be generated inside controlled and isolated test environments.


