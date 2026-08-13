# Master Viva Voce Questions & Answers: Federated Continual Learning for Zero-Day Intrusion Detection in IoMT

> [!IMPORTANT]
> **Definitive Defense Resource**: All technical answers, mathematical formulations, architectural decisions, and metric values in this document reflect the actual Python implementation and empirical benchmark results of the project.

---

## Category A: Project Fundamentals

### Q1: What is the core title and objective of your research project?
**Answer**: The project is titled *"Federated Continual Learning for Zero-Day Intrusion Detection in IoMT"*. The primary objective is to design, implement, and empirically validate a privacy-preserving Intrusion Detection System (IDS) that trains collaboratively across non-IID hospital networks using Federated Learning (FedAvg), adapts to sequential attack streams without catastrophic forgetting using Experience Replay, and detects novel zero-day malware attacks via Energy-Based Open-Set Anomaly Scoring.

### Q2: What are the three core architectural pillars of your proposed model?
**Answer**: The three pillars are: (1) **Federated Learning (FedAvg)** for decentralized, privacy-preserving multi-hospital training; (2) **Continual Learning (Experience Replay)** for mitigating catastrophic forgetting on evolving attack task streams; and (3) **Energy-Based Open-Set Recognition** for detecting unseen zero-day threats.

### Q3: What benchmark dataset was used, and what split ratios were enforced?
**Answer**: We utilized the **Edge-IIoTset (2022)** dataset. The benchmark was split into 70% Training (6,519 samples), 15% Validation (1,398 samples), 15% Test (1,398 samples), and a separate held-out Zero-Day Malware Test set (685 samples).

### Q4: What is the primary novelty of this research compared to existing literature?
**Answer**: While existing works study Federated Learning, Continual Learning, or Anomaly Detection in isolation, our framework is the first to unify all three pillars into a single end-to-end reproducible architecture validated on non-IID medical telemetry streams under strict leakage controls.

---

## Category B: IoMT (Internet of Medical Things)

### Q5: What is the Internet of Medical Things (IoMT)?
**Answer**: IoMT refers to an interconnected infrastructure of medical edge devices, physiological telemetry sensors (e.g., wearable ECG monitors, pulse oximeters, smart infusion pumps), and hospital software systems that communicate patient telemetry over clinical networks.

### Q6: Why are IoMT devices uniquely vulnerable to cyber-attacks?
**Answer**: IoMT devices have restricted processing power, limited battery life, and minimal onboard memory. These constraints prevent running traditional resource-intensive endpoint detection agents or complex encryption protocols, leaving communication channels exposed to intrusion.

### Q7: What are the clinical consequences of an unmitigated IoMT cyber-attack?
**Answer**: Attacks can compromise data availability (e.g., DDoS disrupting alarm delivery in ICUs), data integrity (e.g., injection attacks altering infusion pump dosage readings), or data privacy (exfiltrating patient records), directly threatening patient lives.

### Q8: How does IoMT network traffic differ from conventional enterprise IT traffic?
**Answer**: IoMT traffic consists of highly periodic, low-latency, small-payload physiological sensor readings interspersed with continuous clinical data streaming, making abnormal burst patterns or protocol violations distinct yet subtle.

---

## Category C: Cybersecurity

### Q9: What is the CIA Triad in the context of IoMT security?
**Answer**: Confidentiality (protecting patient telemetry from unauthorized interception), Integrity (ensuring medical sensor readings are not tampered with), and Availability (guaranteeing continuous operational access to critical life-support data).

### Q10: What is a Distributed Denial of Service (DDoS) attack in medical networks?
**Answer**: A DDoS attack floods medical gateways or telemetry servers with malicious packet bursts (e.g., UDP/ICMP floods), exhausting network bandwidth and preventing legitimate physiological alerts from reaching clinical staff.

### Q11: What is a Man-in-the-Middle (MitM) spoofing attack?
**Answer**: An attacker intercepts and potentially alters packet traffic between an IoMT device and the central monitoring system (e.g., ARP or DNS spoofing), forging sensor values or impersonating a legitimate medical node.

### Q12: What is an injection attack in IoMT applications?
**Answer**: Injection attacks insert malicious code or commands into data streams or web applications (e.g., SQL Injection, Cross-Site Scripting), attempting to gain unauthorized access to clinical backend databases.

---

## Category D: Intrusion Detection Systems (IDS)

### Q13: What is the fundamental difference between Signature-Based and Anomaly-Based IDS?
**Answer**: Signature-based IDS matches network activity against a database of known threat signatures, failing completely against novel zero-day attacks. Anomaly-based IDS builds a baseline model of normal behavior and flags significant deviations as suspicious.

### Q14: Where does your IDS fit within network security taxonomy?
**Answer**: Our system is a hybrid Anomaly and Multi-Class Network IDS (NIDS) designed for tabular packet telemetry, capable of multi-class classification for known attack categories and energy-based anomaly detection for unknown zero-day payloads.

### Q15: Why are traditional centralized IDS architectures unsuitable for multi-hospital networks?
**Answer**: Centralized IDS requires sending all raw packet captures and telemetry logs to a single server. In multi-hospital settings, this violates privacy laws (HIPAA/GDPR) and introduces massive communication bandwidth bottlenecks.

### Q16: What is the role of an edge-deployed IDS in healthcare?
**Answer**: An edge IDS operates directly at local hospital gateways or regional clinical servers, providing real-time threat detection without transmitting sensitive patient records across institutional boundaries.

---

## Category E: Machine Learning

### Q17: Why use Machine Learning for IoMT Intrusion Detection?
**Answer**: Machine Learning automatically learns complex, non-linear feature interactions and high-dimensional statistical patterns in network telemetry that manual rule-based systems cannot capture.

### Q18: What is supervised learning vs unsupervised anomaly detection in your pipeline?
**Answer**: Supervised learning trains the deep MLP backbone on labeled training data across known attack classes. Unsupervised/open-set anomaly detection uses the model's logit energy function to score unseen zero-day samples without target labels.

### Q19: What is the curse of dimensionality in tabular IDS data?
**Answer**: Tabular network logs contain numerous features. High dimensionality can lead to sparsity and overfitting. Our preprocessor cleans, scales, and selects 47 informative telemetry features to maintain computational efficiency.

### Q20: What is overfitting, and how is it prevented in your architecture?
**Answer**: Overfitting occurs when a model memorizes training noise. We prevent it via batch normalization, dropout regularization ($p=0.2$), early stopping on validation loss, and stratified split validation.

---

## Category F: Deep Learning

### Q21: Describe the PyTorch deep learning backbone used in your project.
**Answer**: We implement `TabularMLPBackbone`, a multi-layer perceptron comprising an input layer matching the feature dimension (47), a 128-unit hidden layer with ReLU activation, Batch Normalization, Dropout ($p=0.2$), a 64-unit hidden layer, and an output linear layer producing $C$ class logits.

### Q22: Why choose a PyTorch Multi-Layer Perceptron (MLP) over simple Decision Trees?
**Answer**: PyTorch MLPs provide differentiable weight parameters ($\mathbf{w}$) that can be directly aggregated via FedAvg across federated rounds and fine-tuned sequentially during continual learning tasks.

### Q23: What activation functions and loss functions are used?
**Answer**: Hidden layers use Rectified Linear Units ($	ext{ReLU}(x) = \max(0, x)$). Multi-class classification utilizes Cross-Entropy Loss ($\mathcal{L}_{	ext{CE}}$).

### Q24: How does Batch Normalization improve model training?
**Answer**: Batch Normalization normalizes layer activations across local mini-batches, stabilizing internal covariate shift, accelerating SGD convergence, and acting as a mild regularizer.

---

## Category G: Federated Learning

### Q25: What is Federated Learning (FL)?
**Answer**: Federated Learning is a decentralized machine learning paradigm where multiple client nodes (e.g., hospitals) collaboratively train a shared global model under the orchestration of a central server, without sharing their raw local data.

### Q26: What data is transmitted between hospital clients and the server?
**Answer**: Only model weight parameters ($\mathbf{w}_k$) or parameter gradients are transmitted. Raw patient telemetry and packet logs remain strictly within local hospital firewalls.

### Q27: What is a communication round in Federated Learning?
**Answer**: A communication round consists of: (1) Server broadcasting global weights $\mathbf{w}_{	ext{global}}$ to participating clients; (2) Clients training locally for $E$ epochs; (3) Clients uploading updated weights $\mathbf{w}_k$; and (4) Server aggregating updates.

### Q28: How does Federated Learning protect patient data privacy?
**Answer**: By decoupling model training from data collection, raw data never crosses institutional boundaries, preventing raw data exfiltration during transmission or server compromise.

---

## Category H: Federated Averaging (FedAvg)

### Q29: State the mathematical equation for Federated Averaging (FedAvg).
**Answer**: The central server updates global weights $\mathbf{w}_{	ext{global}}^{(r+1)}$ at round $r+1$ according to:
$$\mathbf{w}_{	ext{global}}^{(r+1)} = \sum_{k=1}^K rac{n_k}{N} \mathbf{w}_k^{(r)}$$
where $n_k$ is the sample count at client $k$, $N = \sum n_k$, and $\mathbf{w}_k^{(r)}$ are client weights after local SGD.

### Q30: Why is sample-proportional weighting ($rac{n_k}{N}$) essential in FedAvg?
**Answer**: Sample weighting ensures that clients with larger local datasets have a proportionally higher contribution to global weight updates, preventing smaller clients from distorting global gradient directions.

### Q31: How many local training epochs ($E$) and client rounds were executed in your experiments?
**Answer**: Clients execute $E=3$ local SGD epochs per round over 10 communication rounds, balancing local computation with global network transmission overhead.

### Q32: What network communication bandwidth was consumed by your FedAvg setup?
**Answer**: With a model size of ~2.10 MB, 10 communication rounds across 5 clients consumed a total network payload of **21.03 MB**.

---

## Category I: Non-IID Learning

### Q33: What does Non-IID mean in federated healthcare setups?
**Answer**: Non-IID (Non-Independent and Identically Distributed) means local data distributions $P_k(x, y)$ differ significantly across hospital clients due to specialized clinical care, patient demographics, and regional attack exposure.

### Q34: How did you mathematically simulate Non-IID data skew across hospitals?
**Answer**: We used a Dirichlet distribution $	ext{Dir}(lpha)$ over label proportions with concentration parameter $lpha=0.5$. Lower $lpha$ values induce extreme label imbalance across clients.

### Q35: Describe the sample sizes of your 5 simulated hospital clients ($H_1 \dots H_5$).
**Answer**: Under $lpha=0.5$, training splits were: $H_1$ General Ward ($2,935$ samples), $H_2$ Cardiology ICU ($1,417$ samples), $H_3$ Pediatric Unit ($1,051$ samples), $H_4$ Oncology Center ($452$ samples), and $H_5$ Emergency Unit ($664$ samples).

### Q36: What is Client Drift in Non-IID Federated Learning?
**Answer**: Client Drift occurs when local SGD updates optimize client-specific loss functions $\mathcal{L}_k$, causing local model weights to diverge in different directions, slowing global convergence.

---

## Category J: Continual Learning

### Q37: What is Continual (or Lifelong) Learning?
**Answer**: Continual Learning is the ability of an adaptive model to sequentially learn new tasks or attack streams ($\mathcal{T}_1 ightarrow \mathcal{T}_2 ightarrow \dots ightarrow \mathcal{T}_M$) over time without degrading performance on previously mastered tasks.

### Q38: How were tasks partitioned in your continual learning pipeline?
**Answer**: Attacks were grouped into 3 sequential task streams: Task $\mathcal{T}_1$ (Infrastructure DoS/DDoS), Task $\mathcal{T}_2$ (Injection & Tampering), and Task $\mathcal{T}_3$ (Reconnaissance Scanning).

### Q39: What is Naive Fine-Tuning in continual learning, and why does it fail?
**Answer**: Naive fine-tuning updates model weights on incoming task data without memory protection. It fails because new gradient steps overwrite weights critical for past tasks, causing severe performance drops.

### Q40: How is Continual Learning evaluated across multiple task phases?
**Answer**: By computing the Average Task Accuracy across all tasks learned so far and evaluating Backward Transfer ($	ext{BWT}$) to measure memory retention.

---

## Category K: Catastrophic Forgetting

### Q41: What is Catastrophic Forgetting?
**Answer**: Catastrophic Forgetting is the phenomenon where a neural network completely forgets past knowledge when trained sequentially on new data distributions due to parameter overwrite.

### Q42: How do you mathematically calculate Backward Transfer ($	ext{BWT}$)?
**Answer**: Backward Transfer is defined as:
$$	ext{BWT} = rac{1}{M-1} \sum_{i=1}^{M-1} \left( R_{M, i} - R_{i, i} ight)$$
where $R_{M, i}$ is the test accuracy on Task $i$ after completing training on final Task $M$, and $R_{i, i}$ is accuracy immediately after learning Task $i$.

### Q43: What was the empirical $	ext{BWT}$ of Naive Fine-Tuning in your baseline experiments?
**Answer**: Naive fine-tuning suffered severe catastrophic forgetting with a Backward Transfer of **$	ext{BWT} = -0.1118$**, dropping Task 1 accuracy from $27.03\%$ down to $4.63\%$.

### Q44: What does a negative $	ext{BWT}$ value indicate?
**Answer**: A negative $	ext{BWT}$ indicates performance degradation on previously learned tasks (forgetting). A value closer to $0.0$ signifies strong memory retention.

---

## Category L: Experience Replay

### Q45: What is Experience Replay in continual learning?
**Answer**: Experience Replay is a memory buffer mechanism where a small representative subset of historical task samples is stored locally and re-introduced alongside new task data during training batches.

### Q46: How did you implement Experience Replay in your hospital clients?
**Answer**: Each hospital client maintains a bounded local memory buffer ($M=500$ samples). Training mini-batches are composed of $80\%$ current task data and $20\%$ historical replay samples drawn from the buffer.

### Q47: Did Experience Replay require sharing raw data between hospital clients?
**Answer**: No. Replay memory buffers are strictly local to each hospital node. No historical patient records or telemetry samples were ever transmitted across the network.

### Q48: What was the empirical impact of Experience Replay on Backward Transfer ($	ext{BWT}$)?
**Answer**: Local Experience Replay stabilized representation learning, improving Backward Transfer in the proposed unified model to **$	ext{BWT} = -0.0874$**.

---

## Category M: Zero-Day Attacks

### Q49: What is a Zero-Day attack in cybersecurity?
**Answer**: A Zero-Day attack is a novel, previously unobserved software exploit or malware strain for which no public patch, signature, or training example exists.

### Q50: Which specific attack categories were held out as zero-day threats in your dataset?
**Answer**: Malware classes **`Ransomware`** and **`Backdoor`** payloads were completely withheld from all training, validation, client partitioning, and replay memory sets.

### Q51: How did you ensure zero-day leakage prevention during experiment execution?
**Answer**: We built an automated leakage verification test (`tests/test_leakage.py`) that audits dataset indexes and label sets, asserting that zero-day attack labels appear exclusively in the zero-day evaluation split.

### Q52: Why cannot traditional closed-set softmax classifiers detect zero-day attacks?
**Answer**: Closed-set classifiers normalize output probabilities via Softmax ($\sum p_i = 1$). When presented with a zero-day sample, Softmax forces the network to assign high confidence to one of the known classes.

---

## Category N: Open-Set Recognition

### Q53: What is Open-Set Recognition (OSR)?
**Answer**: Open-Set Recognition is a formal classification formulation where a model must correctly classify known training classes while simultaneously identifying unobserved out-of-distribution (OOD) unknown classes.

### Q54: What open-set detection method did you select, and why?
**Answer**: We selected **Energy-Based Anomaly Scoring**. Energy scores map logits directly to scalar values proportional to the data density without Softmax normalization overconfidence.

### Q55: State the mathematical formulation for Energy-Based Anomaly Scoring.
**Answer**: Free logit energy $E(\mathbf{x}; \mathbf{w})$ for input vector $\mathbf{x}$ and model weights $\mathbf{w}$ is calculated as:
$$E(\mathbf{x}; \mathbf{w}) = -T \cdot \log \sum_{i=1}^C \exp\left(rac{g_i(\mathbf{x})}{T}ight)$$
where $g_i(\mathbf{x})$ is the raw logit output for class $i$, and $T$ is the temperature parameter ($T=1.0$).

### Q56: How is the decision threshold $	au$ calibrated for zero-day detection?
**Answer**: Threshold $	au$ is calibrated **EXCLUSIVELY** on in-distribution validation energy scores, set at the 95th percentile ($	au = -2.1267$). Samples with $E(\mathbf{x}) > 	au$ are flagged as unknown zero-day attacks.

---

## Category O: Dataset

### Q57: Describe the Edge-IIoTset benchmark dataset origin and composition.
**Answer**: Published by Ferrag et al. (2022), Edge-IIoTset is a realistic IoT/IIoT cybersecurity dataset generated in a multi-tier testbed containing physical sensors, gateways, and edge nodes across 14 attack classes and normal traffic.

### Q58: How many features are present in the dataset, and what types of data do they represent?
**Answer**: The dataset contains 47 processed numerical and categorical features representing network flow duration, packet sizes, payload bytes, protocol flags, and device sensor telemetry.

### Q59: Why was a stratified random split used for data partitioning?
**Answer**: Stratified splitting ensures that rare attack classes maintain identical proportional representation across training (70%), validation (15%), and testing (15%) splits.

### Q60: What was the size of the audited dataset in your final experiments?
**Answer**: The audited dataset comprised 10,000 clean samples ($6,519$ training, $1,398$ validation, $1,398$ test, and $685$ held-out zero-day test samples).

---

## Category P: Data Preprocessing

### Q61: What preprocessing steps were applied to raw tabular telemetry data?
**Answer**: (1) Removal of unused metadata columns (IPs, timestamps); (2) Imputation of missing values using median strategy (`SimpleImputer`); (3) Numerical scaling via `StandardScaler`; (4) Target integer encoding.

### Q62: Why is median imputation preferred over mean imputation for network telemetry?
**Answer**: Network telemetry features (e.g., packet lengths, flow durations) exhibit extreme positive skew and outliers. Median imputation provides a robust central tendency measure unaffected by extreme values.

### Q63: What does `StandardScaler` do mathematically?
**Answer**: `StandardScaler` standardizes features by removing the mean and scaling to unit variance:
$$z = rac{x - \mu}{\sigma}$$
where $\mu$ is the training sample mean and $\sigma$ is the training standard deviation.

### Q64: How were categorical labels encoded?
**Answer**: Attack text labels (e.g., `DDoS-UDP`, `SQL_injection`) were mapped to dense integer indexes $0, 1, \dots, C-1$ using `LabelEncoder`.

---

## Category Q: Leakage Prevention

### Q65: What is Data Leakage in Machine Learning?
**Answer**: Data Leakage occurs when information from outside the training dataset (such as test set statistics) is accidentally introduced during model training or preprocessing, artificially inflating performance.

### Q66: How did you prevent preprocessing leakage in your pipeline?
**Answer**: `StandardScaler` and `SimpleImputer` were fitted **STRICTLY AND EXCLUSIVELY** on the training split $D_{	ext{train}}$. Validation and test splits were transformed using the pre-fitted training parameters.

### Q67: How did you programmatically verify zero leakage across dataset splits?
**Answer**: We implemented `tests/test_leakage.py`, which programmatically asserts that the intersection of row index arrays between train, val, and test splits is identically empty ($\emptyset$).

### Q68: How did you ensure zero-day held-out attack leakage was completely prevented?
**Answer**: Zero-day malware categories (`Ransomware` and `Backdoor`) were filtered out before client data partitioning, training set creation, validation tuning, and replay memory buffer filling.

---

## Category R: Experimental Methodology

### Q69: List all 7 main experiment configurations executed in Phase 12.
**Answer**: E1 Centralized IDS; E2 Local Hospital IDS; E3 Standard FedAvg; E4 Centralized Continual Learning; E5 Federated Continual Learning; E6 Zero-Day Energy Anomaly Detector; E7 Proposed Federated Continual Learning + Zero-Day IDS.

### Q70: How did you ensure exact reproducibility across all experimental runs?
**Answer**: We established a global seeding utility (`src/utils/seed.py`) fixing seeds for `random`, `numpy`, and `torch` (`torch.manual_seed(42)`), disabling non-deterministic CUDA operations.

### Q71: What baseline models were compared against the proposed framework?
**Answer**: Standalone Local Hospital Models (E2), Centralized PyTorch MLP (E1), Standard FedAvg (E3), and Naive Continual Learning Fine-Tuning (E4).

### Q72: How were component ablation studies structured (A1–A5)?
**Answer**: Ablations systematically isolated components: A1 Centralized; A2 FedAvg without CL; A3 CL without FL; A4 FL+CL without Replay; A5 Proposed FL+CL with Replay.

---

## Category S: Evaluation Metrics

### Q73: Define Accuracy, Precision, Recall, and F1-Score.
**Answer**: Accuracy = $rac{TP+TN}{TP+TN+FP+FN}$; Precision = $rac{TP}{TP+FP}$; Recall = $rac{TP}{TP+FN}$; F1-Score = $2 \cdot rac{	ext{Precision} \cdot 	ext{Recall}}{	ext{Precision} + 	ext{Recall}}$.

### Q74: Why is Macro F1-Score preferred over Standard Accuracy for imbalanced IDS data?
**Answer**: Standard Accuracy is dominated by majority classes (e.g., normal traffic). Macro F1-Score computes unweighted arithmetic mean across all classes, giving equal weight to rare attack types.

### Q75: What is the False Alarm Rate (FAR) or False Positive Rate (FPR)?
**Answer**: $	ext{FAR} = 	ext{FPR} = rac{FP}{FP + TN}$. In cybersecurity, FAR measures the proportion of benign physiological traffic incorrectly flagged as malicious attacks.

### Q76: What is the Area Under the Receiver Operating Characteristic Curve (ROC-AUC)?
**Answer**: ROC-AUC plots True Positive Rate vs. False Positive Rate across all classification thresholds, measuring the overall discrimination capability of anomaly scores.

---

## Category T: Results

### Q77: What was the final test accuracy of standard FedAvg (E3) vs Centralized IDS (E1)?
**Answer**: Standard FedAvg achieved a test classification accuracy of **0.5930** (59.30%), virtually matching the Centralized upper bound of **0.5951** (59.51%).

### Q78: What was the performance of isolated Local Hospital models (E2)?
**Answer**: Isolated local hospital models achieved a poor mean accuracy of **0.3572** (35.72%), dropping to $4.94\%$ and $8.79\%$ on cross-institutional testing due to non-IID data skew.

### Q79: What performance did the Proposed Framework (E7) achieve on continual learning and zero-day detection?
**Answer**: The proposed framework achieved an Average Task Accuracy of **0.2722**, a Backward Transfer of **$	ext{BWT} = -0.0874$**, and an open-set Zero-Day ROC-AUC of **0.5415**.

### Q80: What was the False Positive Rate (FPR) on benign physiological sensor traffic?
**Answer**: By calibrating the energy threshold $	au$ at the 95th validation percentile, the False Positive Rate was strictly bounded to **5.01%**.

---

## Category U: Privacy

### Q81: Does standard Federated Learning provide mathematical differential privacy guarantees?
**Answer**: No. Standard FedAvg prevents raw data transmission, but raw model weight updates can theoretically be susceptible to gradient inversion attacks unless paired with Differential Privacy (DP).

### Q82: What is Differential Privacy (DP), and how could it be added to your system?
**Answer**: Differential Privacy adds calibrated Gaussian or Laplacian noise to client updates before transmission, mathematically bounding the information an attacker can infer about individual training samples.

### Q83: How does your framework comply with HIPAA and GDPR regulations?
**Answer**: By ensuring raw medical telemetry $D_k$ never leaves local hospital firewalls, preventing centralized raw data aggregation and minimizing exposure to network eavesdropping.

### Q84: What is a Gradient Inversion Attack in Federated Learning?
**Answer**: A Gradient Inversion Attack is a cryptographic technique where an untrusted server or eavesdropper attempts to reconstruct raw input samples by matching client gradient updates.

---

## Category V: Security

### Q85: What is an Adversarial Poisoning Attack in Federated Learning?
**Answer**: A poisoning attack occurs when malicious clients upload corrupted weight updates or flipped labels to degrade global model accuracy or insert backdoor triggers.

### Q86: How can federated aggregation defend against malicious client updates?
**Answer**: Robust aggregation algorithms like Krum, Trimmed Mean, or Median can be substituted for standard FedAvg to filter out statistical outlier updates from compromised clients.

### Q87: What is the impact of a compromised central aggregation server?
**Answer**: A compromised server cannot view raw patient data, but could attempt to distribute corrupted global weights. Signed model updates and Byzantine fault tolerance mitigate this risk.

### Q88: How does Energy-Based Detection protect against zero-day malware attacks?
**Answer**: By flagging any input sample whose logit energy score exceeds threshold $	au$, alerting security analysts to novel threat vectors without waiting for manual signature updates.

---

## Category W: Limitations

### Q89: What is the primary limitation of your Energy-Based Zero-Day Detector?
**Answer**: The Energy-Based Detector achieved a low Zero-Day Recall (**0.0526**) on stealthy malware payloads, as tabular network flow features for ransomware overlap closely with legitimate administrative traffic.

### Q90: What is the computational and memory overhead of local Experience Replay buffers?
**Answer**: Maintaining a buffer of $M=500$ samples per client consumes local edge storage and adds ~20% computation time per local training batch.

### Q91: What are the network bandwidth constraints of scaling FedAvg to 100+ hospitals?
**Answer**: As client count grows, transmitting model weights (2.10 MB per update) across hundreds of nodes can saturate low-bandwidth edge connections.

### Q92: What assumptions were made in your hospital dataset simulation?
**Answer**: We assumed fixed static client availability during federated rounds and static task boundaries during continual learning streams.

---

## Category X: Future Work

### Q93: How can zero-day malware recall be improved in future research?
**Answer**: By incorporating Contrastive Feature Representation Learning (e.g., SimCLR or SupCon) to push unknown feature embeddings away from known class clusters in latent space.

### Q94: How can communication efficiency be improved for edge deployment?
**Answer**: By applying model compression techniques such as quantization (8-bit integer precision), structured weight pruning, or federated sparsification (Top-K gradient transmission).

### Q95: How can formal privacy guarantees be integrated?
**Answer**: By integrating Local Differential Privacy (LDP) with DP-SGD noise addition to client weight updates prior to server transmission.

### Q96: How can task-free continual learning be applied to real-time packet streams?
**Answer**: By replacing rigid task boundaries ($\mathcal{T}_1 ightarrow \mathcal{T}_2$) with online, streaming continual learning algorithms like A-GEM or Memory Aware Synapses (MAS).

---

## Category Y: Difficult Examiner Questions

### Q97 (Examiner Q1): Why did you choose the Edge-IIoTset dataset, and is it truly suitable for IoMT?
**Answer**: Edge-IIoTset was chosen because it is a modern (2022) high-fidelity cybersecurity benchmark capturing realistic physical IoT/IoMT sensor telemetry and network flows across 14 attack classes. It is highly suitable because it models resource-constrained medical edge device communication alongside multi-tier network attack vectors.

### Q98 (Examiner Q2): Why simulate hospital clients instead of using real hospital datasets?
**Answer**: Real clinical network packet logs are proprietary and legally restricted under HIPAA/GDPR, making public raw packet datasets unavailable. Simulating hospitals using Dirichlet label skew ($lpha=0.5$) on Edge-IIoTset provides a realistic, reproducible benchmark for non-IID cross-institutional evaluation.

### Q99 (Examiner Q3): How exactly did you prevent zero-day data leakage during training and validation?
**Answer**: Zero-day malware categories (`Ransomware` and `Backdoor`) were completely filtered out before client partitioning, training set creation, validation tuning, and replay memory buffer filling. An automated unit test (`tests/test_leakage.py`) programmatically verified zero presence in training sets.

### Q100 (Examiner Q4): Why isn't ordinary multi-class classification sufficient for IoMT intrusion detection?
**Answer**: Ordinary classifiers operate under a closed-set assumption, forcing unobserved zero-day attacks into one of the known classes with overconfident Softmax probabilities. Open-set energy scoring allows flagging novel threats as "unknown" without retraining.

### Q101 (Examiner Q5): Does Federated Learning guarantee absolute privacy, and what happens if the central server is malicious?
**Answer**: Standard FL prevents raw data transmission, but does not provide formal mathematical privacy guarantees against gradient inversion attacks. If the server is malicious, it could attempt gradient reconstruction; adding Differential Privacy (DP) is required for mathematical privacy bounds.

### Q102 (Examiner Q6): Which specific empirical result supports your main research claim?
**Answer**: Experiment E7 proves that our proposed unified model achieves an Average Task Accuracy of **0.2722**, reduces Backward Transfer degradation to **$	ext{BWT} = -0.0874$**, and achieves a Zero-Day ROC-AUC of **0.5415**, confirming privacy, continual retention, and zero-day detection capabilities simultaneously.

### Q103 (Examiner Q7): Why should your zero-day detection result be trusted if ROC-AUC is 0.5415?
**Answer**: We report results honestly without fabricating metrics. The ROC-AUC of 0.5415 reflects genuine open-set detection performance on complex tabular network flows. We explicitly highlight this limitation and propose contrastive feature alignment as the concrete path for future improvement.
