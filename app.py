import os
import sqlite3
import joblib
import numpy as np
import pandas as pd
import io
from datetime import datetime
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import google.generativeai as genai

# Import your friend's feature extractor
from features import get_fft_features 

# ==========================================
# 1. INITIALIZATION & CONFIGURATION
# ==========================================
load_dotenv()
app = FastAPI(title="UrbanPulse: High-Confidence Auditor")

# AI Setup
api_key = os.getenv("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)
    gemini_model = genai.GenerativeModel('models/gemini-1.5-flash')

# ML Model Loading
try:
    vibration_classifier = joblib.load("vibration_classifier_robust.pkl")
    print("✅ Machine Learning Layers Loaded")
except Exception as e:
    print(f"⚠️ Model Load Warning: {e}")

# ==========================================
# 2. DATABASE ARCHITECTURE
# ==========================================
def init_db():
    conn = sqlite3.connect('urbanpulse.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS vibration_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    location_name TEXT, lat REAL, lng REAL,
                    avg_rms REAL, peak_freq REAL, freq_jitter REAL,
                    ml_confidence REAL, verdict TEXT, ai_insight TEXT
                )''')
    c.execute('''CREATE TABLE IF NOT EXISTS thermal_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    location_name TEXT, lat REAL, lng REAL,
                    battery_temp REAL, cpu_load TEXT,
                    is_charging BOOLEAN, is_anomaly BOOLEAN, ai_insight TEXT
                )''')
    conn.commit()
    conn.close()

init_db()

# ==========================================
# 3. DATA SCHEMAS & HELPERS
# ==========================================
class VibrationAudit(BaseModel):
    accel_x: list[float]; accel_y: list[float]; accel_z: list[float]
    lat: float; lng: float; speed_kmh: float; duration_sec: float; location_name: str

class ThermalAudit(BaseModel):
    battery_temp: float; cpu_load: str; is_charging: bool
    lat: float; lng: float; location_name: str

def safe_get_peak(feat_obj):
    if feat_obj is None: return 0.0
    if hasattr(feat_obj, 'iloc'): return float(feat_obj.iloc[0]['peak_freq_x']) if not feat_obj.empty else 0.0
    if isinstance(feat_obj, np.ndarray) and feat_obj.size > 0:
        return float(feat_obj[0]) if feat_obj.ndim == 1 else float(feat_obj[0, 0])
    return 0.0

# ==========================================
# 4. DATASET EXPORT TOOL (CSV)
# ==========================================
@app.get("/export/datasets")
async def export_datasets():
    conn = sqlite3.connect('urbanpulse.db')
    df = pd.read_sql_query("SELECT * FROM vibration_logs WHERE verdict = 'VERIFIED'", conn)
    conn.close()
    stream = io.StringIO()
    df.to_csv(stream, index=False)
    response = StreamingResponse(iter([stream.getvalue()]), media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=verified_energy_data.csv"
    return response

# ==========================================
# 5. VIBRATION AUDIT PIPELINE
# ==========================================
@app.post("/audit/vibration")
async def audit_vibration(data: VibrationAudit):
    # --- LAYER 0: DATA INTEGRITY ---
    if data.duration_sec < 10:
        return {"verdict": "REJECTED", "reason": "Duration too short (<10s)"}
    if data.speed_kmh > 15:
        return {"verdict": "REJECTED", "reason": "Speed too high for structural audit"}

    # --- LAYER 2: PHYSICS & ML EXTRACTION ---
    df_full = pd.DataFrame({'accel_x': data.accel_x, 'accel_y': data.accel_y, 'accel_z': data.accel_z, 'label': 'idle'})
    X_features, _ = get_fft_features(df_full)
    peak_freq = safe_get_peak(X_features)
    mag = np.sqrt(np.array(data.accel_x)**2 + np.array(data.accel_y)**2 + np.array(data.accel_z)**2)
    rms_val = float(np.sqrt(np.mean(mag**2)))

    # --- LAYER 3: SIGNAL COHERENCE (STABILITY) ---
    mid = len(data.accel_x) // 2
    f1, _ = get_fft_features(df_full.iloc[:mid]); f2, _ = get_fft_features(df_full.iloc[mid:])
    freq_jitter = abs(safe_get_peak(f1) - safe_get_peak(f2))
    is_coherent = freq_jitter < 2.0

    # --- LAYER 4: SOCIAL/HISTORICAL GATE (Unique Signature Check) ---
    conn = sqlite3.connect('urbanpulse.db'); c = conn.cursor()
    c.execute("""SELECT COUNT(DISTINCT avg_rms) FROM vibration_logs 
                 WHERE lat BETWEEN ? AND ? 
                 AND timestamp >= datetime('now', '-1 day')""", 
              (data.lat-0.0001, data.lat+0.0001))
    recent_users = c.fetchone()[0] or 0

    # Logarithmic Trust Engine: Hits ~70% at 5 unique signals, capped at 87%
    if recent_users > 0:
        calculated_conf = 60 + (5.8 * np.log(recent_users + 1))
        ml_conf = min(calculated_conf, 87.0) / 100.0
    else:
        ml_conf = 0.60 

    # --- LAYER 5: VERIFICATION GATE ---
    is_verified = (ml_conf > 0.5) and (rms_val > 0.1) and is_coherent

    # --- LAYER 6: AI INSIGHT ---
    insight = "Signal verification failed."
    if is_verified:
        try:
            prompt = (f"Analyze urban energy: {data.location_name}, Freq={round(peak_freq,1)}Hz, "
                      f"Consensus Count={recent_users}. 20 words max.")
            response = gemini_model.generate_content(prompt)
            insight = response.text.strip()
        except Exception: insight = "Verified point recorded. AI reasoning offline."

        c.execute("""INSERT INTO vibration_logs 
                     (location_name, lat, lng, avg_rms, peak_freq, freq_jitter, ml_confidence, verdict, ai_insight) 
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                  (data.location_name, data.lat, data.lng, rms_val, peak_freq, freq_jitter, ml_conf, "VERIFIED", insight))
        conn.commit()
    
    conn.close()
    return {
        "verdict": "VERIFIED" if is_verified else "REJECTED",
        "logic_breakdown": {
            "speed_kmh": data.speed_kmh,
            "social_consensus_count": recent_users,
            "boosted_confidence": f"{ml_conf*100:.1f}%",
            "jitter_hz": round(freq_jitter, 2)
        },
        "ai_analysis": insight
    }

# ==========================================
# 6. THERMAL AUDIT PIPELINE
# ==========================================
@app.post("/audit/thermal")
async def audit_thermal(data: ThermalAudit):
    # --- LAYER 1: HARDWARE FILTER ---
    cpu_intensity = {"low": 15, "medium": 45, "high": 90}.get(data.cpu_load, 50)
    is_internal = data.is_charging or (cpu_intensity > 75)
    is_external_anomaly = (data.battery_temp > 42.0) and not is_internal

    # --- LAYER 2: THERMAL CONSENSUS (Unique Device Check) ---
    conn = sqlite3.connect('urbanpulse.db'); c = conn.cursor()
    c.execute("""SELECT COUNT(DISTINCT battery_temp) FROM thermal_logs 
                 WHERE lat BETWEEN ? AND ? 
                 AND is_anomaly = 1 
                 AND timestamp >= datetime('now', '-1 day')""", 
              (data.lat-0.0005, data.lat+0.0005))
    cluster_count = c.fetchone()[0] or 0
    
    # Apply Logarithmic Formula for Thermal Trust
    if is_external_anomaly:
        calc_conf = 60 + (5.8 * np.log(cluster_count + 1))
        final_thermal_conf = min(calc_conf, 87.0)
    else:
        final_thermal_conf = 0.0

    # --- LAYER 3: VERDICT & AI ---
    verdict_label = "VERIFIED WASTE HEAT" if is_external_anomaly else "INTERNAL DEVICE HEAT"
    ai_report = "Environment normal."
    
    if is_external_anomaly:
        try:
            prompt = (f"Thermal Waste: {data.location_name}, Temp={data.battery_temp}C, "
                      f"Consensus={cluster_count} users. Suggest industrial source. 15 words.")
            ai_report = gemini_model.generate_content(prompt).text.strip()
        except Exception: ai_report = "Anomaly detected. AI reasoning offline."

        c.execute("""INSERT INTO thermal_logs 
                     (location_name, lat, lng, battery_temp, cpu_load, is_charging, is_anomaly, ai_insight) 
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                  (data.location_name, data.lat, data.lng, data.battery_temp, 
                   data.cpu_load, data.is_charging, 1, ai_report))
        conn.commit()
    
    conn.close()
    return {
        "verdict": verdict_label,
        "is_harvestable": is_external_anomaly,
        "thermal_confidence": f"{final_thermal_conf:.1f}%",
        "logic_breakdown": {
            "internal_noise_detected": is_internal,
            "unique_nearby_hotspots": cluster_count
        },
        "ai_analysis": ai_report
    }


@app.get("/government/hotspots")
async def get_hotspots():
    conn = sqlite3.connect('urbanpulse.db')
    # This query groups nearby reports into "Zones" and calculates the average confidence
    query = """
        SELECT 
            location_name, 
            AVG(lat) as lat, 
            AVG(lng) as lng, 
            COUNT(DISTINCT id) as report_count,
            AVG(ml_confidence) * 100 as avg_trust,
            AVG(avg_rms) as intensity,
            ai_insight
        FROM vibration_logs 
        WHERE verdict = 'VERIFIED'
        GROUP BY location_name
        HAVING report_count >= 2
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df.to_dict(orient="records")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)