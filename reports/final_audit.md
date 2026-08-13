# Independent Final Data and Experiment Audit Report

> [!IMPORTANT]
> **Audit Status**: **PASS**  
> **Auditor**: Automated Independent Peer Reviewer  
> **Dataset**: Edge-IIoTset Cybersecurity Benchmark (IoT/IoMT Baseline)  
> **Audit Date**: 2026-08-12

---

## Executive Summary
This document provides an independent peer-review audit of data preprocessing, zero-day class isolation, federated aggregation mechanics, continual learning replay integrity, and empirical metrics consistency for the research project *"Federated Continual Learning for Zero-Day Intrusion Detection in IoMT"*.

---

## Audit Checklist & Verification Matrix

### 1. Data Integrity & Leakage Audit
- **Train / Validation / Test Index Overlap**: **`0 OVERLAP`** (`PASS`)
- **Duplicate Records**: **`0 DUPLICATES`** (`PASS`)
- **Preprocessing Leakage**: **`PASSED_CLEAN`** (`PASS`) — `StandardScaler` and `SimpleImputer` fitted exclusively on training set.
- **Target Leakage**: **`PASSED_CLEAN`** (`PASS`) — All target labels stripped before feature matrix generation.

### 2. Zero-Day Attack Isolation Audit
- **Withheld Attack Family**: Malware (`Ransomware` & `Backdoor`).
- **Zero-Day Samples in Training Split**: **`0`** (`PASS`)
- **Zero-Day Samples in Validation Split**: **`0`** (`PASS`)
- **Zero-Day Samples in Replay Buffer**: **`0`** (`PASS`)
- **Evaluation Isolation**: Withheld attacks introduced exclusively during open-set energy anomaly evaluation (`PASS`).

### 3. Federated Learning Mechanics Audit
- **FedAvg Aggregation Formula**: Mathematically verified FedAvg weight aggregation (`PASS`).
- **Data Privacy & Raw Data Sharing**: **`ZERO RAW DATA TRANSMISSION`** (`PASS`) — Hospital nodes exchange only model weight updates.
- **Non-IID Partitioning**: 5 simulated hospital clients partitioned under Dirichlet distribution (alpha=0.5).

### 4. Continual Learning Replay & Metrics Audit
- **Sequential Task Ordering**: Task 1 (Infrastructure DoS) -> Task 2 (Injection/MitM) -> Task 3 (Scanning) (`PASS`).
- **Replay Memory Bounds**: Maximum capacity M=500 enforced via uniform reservoir sampling (`PASS`).
- **Backward Transfer (BWT) Verification**: Calculated strictly via standard BWT formulation (`PASS`).

### 5. Empirical Metrics Consistency Audit
- **Sample Count Conservation**: N_train = 6519 + N_val = 1398 + N_test = 1398 + N_zero_day = 685 = 10000 (`PASS`).
- **Metric Bounds**: All calculated Precision, Recall, F1, and AUROC values lie within [0.0, 1.0] (`PASS`).
- **Zero Fabrication**: All reported numerical results originate from executed Python scripts persisted on disk (`PASS`).

---

## Final Certification
**VERDICT**: **ALL 5 RESEARCH AUDIT DIMENSIONS PASSED WITHOUT CRITICAL FAILURES.**
The dataset, experimental pipeline, and empirical results are fully verified for final publication and paper generation.
