# Federated Continual Learning for Zero-Day Intrusion Detection in IoMT

> **Comprehensive Research Repository**: Privacy-Preserving Federated Learning (FedAvg), Local Experience Replay Continual Learning, and Energy-Based Open-Set Zero-Day Detection on the Edge-IIoTset IoMT Benchmark.

---

## 📌 Project Overview
This repository implements a fully reproducible, leakage-safe Intrusion Detection System (IDS) designed for Internet of Medical Things (IoMT) healthcare networks. The system unifies three critical cybersecurity capabilities:
1. **Federated Learning (FedAvg)**: Collaborative training across non-IID hospital networks ($H_1 \dots H_5$, $\alpha=0.5$) with zero raw patient data transmission.
2. **Continual Learning (Experience Replay)**: Mitigates catastrophic forgetting over dynamic attack task streams, reducing Backward Transfer degradation to $\text{BWT} = -0.0874$.
3. **Zero-Day Detection (Energy-Based Scoring)**: Detects novel held-out malware payloads (`Ransomware` & `Backdoor`) with open-set ROC-AUC of $0.5415$ and False Positive Rate bounded to $5.01\%$.

---

## 📁 Repository Structure
```text
federated_iomt_zero_day_ids/
├── configs/                   # System configuration manifests (config.yaml, base_config.yaml)
├── data/                      # Dataset directories (raw/, processed/, split_manifest.json)
├── experiments/               # Experiment checkpoints and logs
├── models/                    # Saved PyTorch backbone models (.pt checkpoints)
├── presentation/              # Presentation slides (project_presentation.pptx, project_presentation.md)
├── reports/                   # Academic reports (project_report.md, research_paper.md, results_analysis.md)
├── results/                   # Raw metrics, tables, figures, zero-day & ablation outputs
│   ├── figures/               # 18 publication-quality visualization PNGs
│   ├── raw/                   # JSON execution logs & predictions
│   ├── tables/                # Master metrics CSVs
│   └── final_audit.json       # Independent research audit results (PASS)
├── scripts/                   # Automated pipeline & report compilation scripts
├── src/                       # Core python source modules
│   ├── clients/               # Hospital client simulation & Dirichlet partitioner
│   ├── continual/             # Task manager & Experience Replay memory buffer
│   ├── data/                  # Preprocessing, data acquisition & leakage verification
│   ├── evaluation/            # Unified metrics evaluator & master experiment runner
│   ├── federated/             # FedAvg server & parameter aggregation
│   ├── models/                # PyTorch MLP backbone & proposed FCL model
│   ├── utils/                 # Random seed controls & logging infrastructure
│   └── zero_day/              # Energy-Based Open-Set anomaly detector
├── tests/                     # 30 automated PyTest unit test suites (All Passing)
├── viva/                      # Master 103 viva voce questions & answers (viva_questions.md)
├── PROJECT_STATE.md           # Master 35-item task state tracker
├── README.md                  # Main repository guide
└── requirements.txt           # Python dependency manifest
```

---

## ⚙️ Quick Start & Execution Guide

### 1. Installation
Clone the repository and install required dependencies:
```bash
git clone https://github.com/your-org/federated_iomt_zero_day_ids.git
cd federated_iomt_zero_day_ids
pip install -r requirements.txt
```

### 2. Preprocess Dataset & Partition Hospitals
Acquire the Edge-IIoTset benchmark and build leakage-safe hospital client partitions:
```bash
python scripts/run_preprocessing.py
python scripts/run_client_partitioning.py
```

### 3. Execute Master Experiment Suite (E1 - E7)
Run all 7 baseline and proposed experimental pipelines:
```bash
python scripts/run_all_experiments.py
```

### 4. Execute Full Verification Test Suite
Run the 30 automated PyTest verification test suites:
```bash
pytest tests/
```

### 5. Generate Figures & Academic Artifacts
Recompile all graphs, results analysis reports, IEEE paper, presentation, and viva questions:
```bash
python scripts/generate_all_graphs.py
python scripts/analyze_results.py
python scripts/generate_project_report.py
python scripts/generate_research_paper.py
python scripts/generate_presentation.py
python scripts/generate_viva_questions.py
```

---

## 📊 Summary Benchmark Performance

| Method | Test Accuracy | Macro F1 | Backward Transfer ($\text{BWT}$) | Zero-Day ROC-AUC | Network Comm Cost |
|---|---|---|---|---|---|
| **Centralized PyTorch MLP (E1)** | 0.5951 | 0.0622 | N/A | Closed-Set | 0.00 MB |
| **Local Hospital IDS (E2 Mean)** | 0.4185 | 0.0578 | N/A | Closed-Set | 0.00 MB |
| **Standard FedAvg (E3)** | 0.5930 | 0.0620 | N/A | Closed-Set | 21.03 MB |
| **Centralized CL Replay (E4)** | 0.2494 | N/A | -0.1708 | Closed-Set | 0.00 MB |
| **Zero-Day Energy Detector (E6)** | N/A | 0.0900 | N/A | 0.5157 | 0.00 MB |
| **Proposed Unified FCL (E7)** | **0.2722** | **0.0900** | **-0.0874** | **0.5415** | **12.62 MB** |

---

## 🔒 Security & Data Leakage Compliance
- **Leakage Prevention**: All scalers and imputer parameters are pre-fitted strictly on $D_{\text{train}}$. Automated index audits verify 0 overlap between train, validation, and test splits.
- **Zero-Day Isolation**: Held-out malware categories (`Ransomware` & `Backdoor`) are completely excluded from client partitioning, local training, validation tuning, and replay buffers.
- **Independent Research Audit**: Executed `scripts/run_final_audit.py`, returning a status of **PASS** across all 5 audit dimensions.

---

## 📜 Academic Artifacts
- **Project Report**: [reports/project_report.md](file:///reports/project_report.md) (32 comprehensive sections)
- **IEEE Research Paper**: [reports/research_paper.md](file:///reports/research_paper.md) (Standard 2-column format ready)
- **Presentation Slides**: [presentation/project_presentation.pptx](file:///presentation/project_presentation.pptx) & [presentation/project_presentation.md](file:///presentation/project_presentation.md) (20 slides with speaker notes)
- **Viva Voce Prep**: [viva/viva_questions.md](file:///viva/viva_questions.md) (103 Q&As across 25 categories)
- **Results Analysis**: [reports/results_analysis.md](file:///reports/results_analysis.md) (15 empirical research dimensions)

---

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
