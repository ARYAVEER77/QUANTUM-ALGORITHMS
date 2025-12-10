# QUANTUM ALGORITHMS FOR DISASTER PREDICTION AND MANAGEMENT

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![Qiskit](https://img.shields.io/badge/Qiskit-0.43.1-purple.svg)](https://qiskit.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Authors:** Aryaveer Desai¹²
<br>
*¹ MVM International School India and Harvard Summer School*
<br>
*² MVM International, Mumbai, India*

---

## 📌 Overview
This repository contains the source code, datasets, and experimental validation scripts for the research paper **"Quantum Algorithms for Disaster Prediction and Management"**.

We present a proof-of-concept hybrid framework that integrates **Quantum Computing, Artificial Intelligence, and Data Science** to enhance disaster response strategies. The system is composed of four core quantum modules:

1.  **Quantum Neural Network (QNN):** A Sampler-based QNN for classifying disaster types (Flood, Earthquake, etc.) using the EM-DAT dataset.
2.  **Quantum Approximate Optimization Algorithm (QAOA):** Solves resource allocation problems as combinatorial optimization tasks.
3.  **Quantum Walk:** Optimizes evacuation routes in dynamic disaster zones.
4.  **Quantum Phase Estimation (QPE):** Simulates and estimates disaster severity levels.

All quantum algorithms are implemented using **Qiskit** and executed via classical simulation.

---

## 📂 Repository Structure

```text
.
├── main.py                   # Master script to run the full end-to-end pipeline
├── analysis_validation.py    # Statistical validation (VIF & Ablation Studies) [Appendix B]
├── requirements.txt          # Python dependencies and version locking [Appendix A]
├── cleaning.py               # Data preprocessing and cleaning logic
├── ai_model.py               # Classical AI training (XGBoost) and ROC generation
├── quantum_qbm.py            # Quantum Boltzmann Machine / Neural Network module
├── quantum_qaoa.py           # QAOA Resource Allocation module
├── quantum_walk.py           # Quantum Walk Evacuation module
├── quantum_qpe.
