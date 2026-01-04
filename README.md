# 🚦 UrbanPulse-Auditor
### High-Confidence Urban Energy Signal Verification System

**UrbanPulse-Auditor** is a high-fidelity backend verification engine that converts raw urban sensor signals (vibration, thermal, and device telemetry) into validated, auditable energy insights. It acts as the **Intelligence Layer** for smart city planning.

---

## 🧠 Why UrbanPulse?
Cities generate massive ambient energy—mechanical vibrations from metros, thermal exhaust from industries, and footstep resonance. 

* **The Problem:** Raw sensor data is noisy, erratic, and unreliable for infrastructure decisions.
* **The Solution:** A multi-layer verification pipeline that filters noise, extracts physics-based features, and uses ML to classify "Energy Gold."

---

## 🏗️ Integrated Backend Architecture
The system processes a **60-point high-density JSON array** through a four-stage "Gate" pipeline:

### **Stage 1: Signal Ingestion & Integrity (FastAPI)**
* **Technology:** FastAPI & Pydantic.
* **Gate 1:** Rejects audits if `speed_kmh > 15` (filters car engine noise) or `duration < 10s`.

### **Stage 2: Physics & Thermal Extraction (NumPy/SciPy)**
* **Kinetic Audit:** Uses **FFT (Fast Fourier Transform)** to find the Resonance Frequency (Hz) and **RMS** for Energy Intensity.
* **Thermal Audit:** Calculates **$\Delta T$ (Thermal Gradient)**. 
    * *Formula:* $\Delta T = T_{source} - T_{ambient}$.
    * *Logic:* $\Delta T > 10°C$ flags the site for Seebeck-Effect (TEG) harvesting.

### **Stage 3: ML Classification (Scikit-Learn)**
* **Model:** **Random Forest Classifier**.
* **Logic:** Matches the frequency "DNA" to known patterns (e.g., Metro: 10-20Hz, Generator: 50-60Hz).
* **Anomaly Detection:** Uses **Isolation Forest** to reject "fake" data like phone drops or manual shaking.

### **Stage 4: AI Reasoning (Gemini 1.5 Flash)**
* **Role:** Adds scientific context to the ML output, explaining *why* a specific node is feasible for deployment.

---

## 📊 System Design Chart


| Layer | Component | Library |
| :--- | :--- | :--- |
| **API** | Data Gateway | FastAPI |
| **Physics** | Thermal & Vibration Analysis | NumPy, SciPy |
| **ML** | Source Classification (Random Forest) | Scikit-Learn |
| **Storage** | Knowledge Base (Hotspots) | SQLite |
| **Intelligence** | Synthesis Agent | Gemini API |

---

## ⚙️ Tech Stack
* **Backend:** Python (FastAPI)
* **Math:** NumPy (RMS Calculation), SciPy (FFT Analysis)
* **Machine Learning:** Scikit-Learn (Classification & Anomaly Detection)
* **Database:** SQLite (Local Persistence via SQLAlchemy)
* **AI:** Google Gemini 1.5 Flash API
* **Frontend:** React.js (Live Dashboard)

---

## 🚀 Installation & Setup

1. **Clone & Install Dependencies**
   ```bash
   git clone [https://github.com/your-username/UrbanPulse-Auditor.git](https://github.com/your-username/UrbanPulse-Auditor.git)
   pip install -r requirements.txt