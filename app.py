import os
import json
import uuid
from datetime import datetime
from flask import Flask, request, jsonify, render_template
from supabase import create_client
from dotenv import load_dotenv
import numpy as np

load_dotenv()

app = Flask(__name__)

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

latest_reading = {}
pending_reading = {}  # Stores sensor data waiting for pH

# ─── Color Name Detection ────────────────────────────────────────
def detect_color_name(r, g, b):
    colors = {
        "Crystal Clear":   (255, 255, 255),
        "Light Green":     (144, 238, 144),
        "Green":           (0,   128,   0),
        "Yellow-Green":    (154, 205,  50),
        "Orange":          (255, 165,   0),
        "Red":             (255,   0,   0),
        "Blue":            (  0,   0, 255),
        "Light Blue":      (173, 216, 230),
        "Milky White":     (245, 245, 220),
        "Dark Amber":      (139,  90,  43),
        "Brown":           (139,  69,  19),
        "Dark Brown":      ( 92,  64,  51),
        "Clear Water":     (200, 220, 230),
        "Transparent":     (200, 200, 200),
    }
    min_dist = float('inf')
    closest = "Unknown"
    for name, (cr, cg, cb) in colors.items():
        dist = ((r-cr)**2 + (g-cg)**2 + (b-cb)**2) ** 0.5
        if dist < min_dist:
            min_dist = dist
            closest = name
    return closest

# ─── Coolant Condition Assessment ───────────────────────────────
def assess_condition(ph, tds, temp, turbidity, uv, vapor):
    score = 100

    if 7.0 <= ph <= 9.5:        pass
    elif 6.5 <= ph < 7.0 or 9.5 < ph <= 10.5: score -= 15
    else:                        score -= 35

    if tds < 500:               pass
    elif tds < 1500:            score -= 10
    elif tds < 3000:            score -= 25
    else:                        score -= 40

    if turbidity < 30:          pass
    elif turbidity < 60:        score -= 15
    else:                        score -= 30

    if uv < 100:                pass
    elif uv < 300:              score -= 10
    else:                        score -= 20

    if vapor < 200:             pass
    elif vapor < 400:           score -= 10
    else:                        score -= 20

    if score >= 80:   return "Excellent", score
    elif score >= 60: return "Good", score
    elif score >= 40: return "Fair - Monitor Closely", score
    elif score >= 20: return "Poor - Replace Soon", score
    else:             return "Critical - Replace Immediately", score

# ─── Application Matching ────────────────────────────────────────
def match_applications(ph, tds, temp, turbidity, r, g, b, uv, vapor):
    color_name = detect_color_name(r, g, b)
    results = []

    apps = [
        {
            "name": "CNC Machining Fluid",
            "icon": "⚙️",
            "category": "Industrial Machining",
            "criteria": lambda: (8.6<=ph<=9.5 and tds<500 and turbidity<20 and 20<=temp<=50 and vapor<300),
            "description": "Precision CNC operations require stable alkaline coolant.",
            "parameters": "pH 8.6–9.5, TDS <500 ppm, Turbidity <20 NTU"
        },
        {
            "name": "Petrol Engine Coolant",
            "icon": "🚗",
            "category": "Automotive — Light Duty",
            "criteria": lambda: (7.5<=ph<=11.0 and 150<=tds<=800 and turbidity<15 and uv<150 and vapor<250),
            "description": "Ethylene glycol coolant for petrol vehicles per ASTM D3306.",
            "parameters": "pH 7.5–11.0, TDS 150–800 ppm, Turbidity <15 NTU"
        },
        {
            "name": "Diesel Engine Coolant",
            "icon": "🚛",
            "category": "Automotive — Heavy Duty",
            "criteria": lambda: (8.0<=ph<=10.5 and 300<=tds<=1500 and turbidity<20 and uv<200 and vapor<300),
            "description": "Fully formulated coolant for heavy duty diesel per ASTM D6210.",
            "parameters": "pH 8.0–10.5, TDS 300–1500 ppm, Turbidity <20 NTU"
        },
        {
            "name": "Hydraulic System Fluid",
            "icon": "🔧",
            "category": "Industrial Hydraulics",
            "criteria": lambda: (6.5<=ph<=8.5 and tds<200 and turbidity<5 and uv<80 and vapor<200),
            "description": "Ultra-clean hydraulic fluid per ISO 11158.",
            "parameters": "pH 6.5–8.5, TDS <200 ppm, Turbidity <5 NTU"
        },
        {
            "name": "Brake System Fluid",
            "icon": "🛑",
            "category": "Automotive Safety",
            "criteria": lambda: (7.0<=ph<=11.5 and tds<100 and turbidity<3 and uv<50 and vapor<150),
            "description": "DOT-grade brake fluid per SAE J1703.",
            "parameters": "pH 7.0–11.5, TDS <100 ppm, Turbidity <3 NTU"
        },
        {
            "name": "Air Compressor Coolant",
            "icon": "💨",
            "category": "Industrial Pneumatics",
            "criteria": lambda: (6.5<=ph<=8.0 and tds<400 and turbidity<15 and uv<100 and vapor<300),
            "description": "Clean neutral coolant for air compressors per ISO 6743-3A.",
            "parameters": "pH 6.5–8.0, TDS <400 ppm, Turbidity <15 NTU"
        },
        {
            "name": "Gearbox Lubricant",
            "icon": "⚙️",
            "category": "Industrial Transmission",
            "criteria": lambda: (5.5<=ph<=8.5 and tds>=1500 and vapor>=300 and temp<=120),
            "description": "High additive gear lubricant per ISO 3448.",
            "parameters": "TDS >1500 ppm, Vapor >300 ppm, pH 5.5–8.5"
        },
        {
            "name": "EDM Machine Fluid",
            "icon": "🔬",
            "category": "Precision Manufacturing",
            "criteria": lambda: (6.5<=ph<=7.5 and tds<50 and turbidity<2 and uv<30 and vapor<100),
            "description": "Deionised dielectric fluid for EDM per ISO 4370.",
            "parameters": "pH 6.5–7.5, TDS <50 ppm, Turbidity <2 NTU"
        },
        {
            "name": "Heat Exchanger Fluid",
            "icon": "🌡️",
            "category": "Industrial Thermal",
            "criteria": lambda: (7.5<=ph<=9.5 and 500<=tds<=3000 and turbidity<15 and temp>=50 and uv<200),
            "description": "Inhibited glycol fluid for heat exchangers per TEMA.",
            "parameters": "pH 7.5–9.5, TDS 500–3000 ppm, Temp >50°C"
        },
        {
            "name": "Metal Forming Fluid",
            "icon": "🏭",
            "category": "Metal Processing",
            "criteria": lambda: (7.0<=ph<=9.5 and 400<=tds<=2000 and 20<=turbidity<=80 and vapor<400),
            "description": "Semisynthetic emulsion for metal forming per ISO 6743-7.",
            "parameters": "pH 7.0–9.5, TDS 400–2000 ppm, Turbidity 20–80 NTU"
        },
        {
            "name": "EV Battery Cooling",
            "icon": "🔋",
            "category": "Electric Vehicles",
            "criteria": lambda: (6.8<=ph<=7.5 and tds<100 and turbidity<5 and uv<50 and vapor<150),
            "description": "Dielectric coolant for EV battery management per SAE J2800.",
            "parameters": "pH 6.8–7.5, TDS <100 ppm, Turbidity <5 NTU"
        },
        {
            "name": "Food Grade Cooling",
            "icon": "🍃",
            "category": "Food & Beverage",
            "criteria": lambda: (6.5<=ph<=7.5 and tds<200 and turbidity<1 and uv<30 and vapor<100),
            "description": "NSF/ANSI 169 certified propylene glycol coolant.",
            "parameters": "pH 6.5–7.5, TDS <200 ppm, Turbidity <1 NTU"
        },
        {
            "name": "Pharmaceutical Cooling",
            "icon": "💊",
            "category": "Pharmaceutical",
            "criteria": lambda: (6.8<=ph<=7.2 and tds<10 and turbidity<1 and uv<20 and vapor<80),
            "description": "Water for Injection grade per USP/FDA 21 CFR.",
            "parameters": "pH 6.8–7.2, TDS <10 ppm, Turbidity <1 NTU"
        },
        {
            "name": "Solar Panel Cooling",
            "icon": "☀️",
            "category": "Renewable Energy",
            "criteria": lambda: (7.0<=ph<=8.5 and tds<500 and turbidity<10 and uv<120 and vapor<200),
            "description": "Propylene glycol solar thermal fluid per ASTM E2277.",
            "parameters": "pH 7.0–8.5, TDS <500 ppm, Turbidity <10 NTU"
        },
        {
            "name": "Nuclear Plant Cooling",
            "icon": "⚛️",
            "category": "Nuclear Energy",
            "criteria": lambda: (6.9<=ph<=7.1 and tds<5 and turbidity<1 and uv<10 and vapor<50),
            "description": "Demineralised water per IAEA safety standards.",
            "parameters": "pH 6.9–7.1, TDS <5 ppm, Turbidity <1 NTU"
        },
        {
            "name": "Data Center Cooling",
            "icon": "🖥️",
            "category": "IT Infrastructure",
            "criteria": lambda: (7.0<=ph<=8.5 and tds<100 and turbidity<5 and uv<50 and vapor<100),
            "description": "Non-conductive coolant per ASHRAE TC 9.9.",
            "parameters": "pH 7.0–8.5, TDS <100 ppm, Turbidity <5 NTU"
        },
        {
            "name": "Marine Engine Coolant",
            "icon": "⛵",
            "category": "Marine",
            "criteria": lambda: (7.5<=ph<=10.5 and 500<=tds<=2000 and turbidity<20 and uv<200),
            "description": "Corrosion-inhibited coolant per IACS/IMO standards.",
            "parameters": "pH 7.5–10.5, TDS 500–2000 ppm, Turbidity <20 NTU"
        },
        {
            "name": "Aircraft Engine Coolant",
            "icon": "✈️",
            "category": "Aerospace",
            "criteria": lambda: (8.0<=ph<=11.0 and 200<=tds<=1000 and turbidity<10 and temp>=60),
            "description": "High-performance coolant per MIL-PRF-23699.",
            "parameters": "pH 8.0–11.0, TDS 200–1000 ppm, Turbidity <10 NTU"
        },
        {
            "name": "General Industrial Use",
            "icon": "🏗️",
            "category": "General Industrial",
            "criteria": lambda: (6.0<=ph<=10.0 and tds<3000 and turbidity<50),
            "description": "Broad-spectrum coolant per WHO industrial guidelines.",
            "parameters": "pH 6.0–10.0, TDS <3000 ppm, Turbidity <50 NTU"
        },
    ]

    for a in apps:
        try:
            if a["criteria"]():
                results.append({
                    "name":        a["name"],
                    "icon":        a["icon"],
                    "category":    a["category"],
                    "description": a["description"],
                    "parameters":  a["parameters"]
                })
        except:
            pass

    return results, color_name

# ─── Diagnosis and Remedy ────────────────────────────────────────
def get_diagnosis_remedy(ph, tds, temp, turbidity, uv, vapor):
    issues = []
    remedies = []

    if ph < 6.5:
        issues.append("Highly acidic — severe corrosion risk")
        remedies.append("Add alkaline buffer immediately. Consider full replacement.")
    elif ph > 10.5:
        issues.append("Excessively alkaline — scaling risk")
        remedies.append("Dilute with distilled water. Add pH stabiliser.")
    elif ph < 7.0:
        issues.append("Slightly acidic coolant")
        remedies.append("Monitor closely. Add corrosion inhibitor.")

    if tds > 3000:
        issues.append("Extremely high TDS — severe contamination")
        remedies.append("Immediate replacement. Flush system thoroughly.")
    elif tds > 1500:
        issues.append("Elevated TDS — mineral buildup detected")
        remedies.append("Partial replacement with distilled water top-up.")

    if turbidity > 60:
        issues.append("High turbidity — heavily contaminated")
        remedies.append("Filter through fine mesh. Check for seal leaks.")
    elif turbidity > 30:
        issues.append("Moderate turbidity detected")
        remedies.append("Apply inline filtration. Monitor particle source.")

    if uv > 300:
        issues.append("Strong UV fluorescence — oil contamination")
        remedies.append("Isolate contamination source. Full drain required.")
    elif uv > 100:
        issues.append("Mild UV fluorescence — possible trace oil")
        remedies.append("Check seals and gaskets. Monitor trend.")

    if vapor > 400:
        issues.append("High chemical vapor — volatile compounds")
        remedies.append("Ensure ventilation. Check coolant compatibility.")

    if temp > 90:
        issues.append("Very high temperature detected")
        remedies.append("Check pump flow rate and heat exchanger.")

    if not issues:
        issues.append("All parameters within acceptable range")
        remedies.append("Coolant is in good condition. Monitor every 500 hours.")

    return " | ".join(issues), " | ".join(remedies)

# ─── Full Analysis Function ──────────────────────────────────────
def run_full_analysis(ph, tds, temp, turbidity,
                      uv, vapor, r, g, b, session_id):
    color_hex  = f"#{r:02x}{g:02x}{b:02x}"
    color_name = detect_color_name(r, g, b)
    condition, score = assess_condition(ph, tds, temp,
                                        turbidity, uv, vapor)
    apps, _ = match_applications(ph, tds, temp, turbidity,
                                  r, g, b, uv, vapor)
    diagnosis, remedy = get_diagnosis_remedy(ph, tds, temp,
                                              turbidity, uv, vapor)
    return {
        "session_id":           session_id,
        "ph_level":             ph,
        "tds_value":            tds,
        "temperature":          temp,
        "turbidity":            turbidity,
        "color_r":              r,
        "color_g":              g,
        "color_b":              b,
        "color_hex":            color_hex,
        "uv_fluorescence":      uv,
        "vapor_level":          vapor,
        "coolant_color_name":   color_name,
        "condition":            condition,
        "application_suggestions": apps,
        "diagnosis":            diagnosis,
        "remedy":               remedy,
        "score":                score,
        "applications":         apps
    }

# ================================================================
# ROUTES
# ================================================================

@app.route("/")
def index():
    history = supabase.table("analysis_history").select("*")\
        .order("created_at", desc=True).limit(10).execute().data
    return render_template("index.html", history=history)

# ─── Pico W sends sensor data (WITHOUT pH) ──────────────────────
@app.route("/sensor_data", methods=["POST"])
def sensor_data():
    global pending_reading
    data = request.get_json()

    session_id = data.get("session_id",
                           "PICO-" + str(uuid.uuid4())[:8])

    pending_reading = {
        "session_id":    session_id,
        "tds":           float(data.get("tds",         500)),
        "temperature":   float(data.get("temperature",  25)),
        "turbidity":     float(data.get("turbidity",    20)),
        "uv_fluorescence": float(data.get("uv_fluorescence", 50)),
        "vapor_level":   float(data.get("vapor_level",  100)),
        "color_r":       int(data.get("color_r",        200)),
        "color_g":       int(data.get("color_g",        200)),
        "color_b":       int(data.get("color_b",        200)),
        "received_at":   datetime.now().isoformat()
    }

    return jsonify({
        "status":     "RECEIVED",
        "session_id": session_id,
        "message":    "Enter pH on dashboard to complete analysis"
    }), 200

# ─── Website submits pH and triggers full analysis ───────────────
@app.route("/submit_ph", methods=["POST"])
def submit_ph():
    global pending_reading, latest_reading
    data = request.get_json()
    ph   = float(data.get("ph", 7.0))

    if not pending_reading:
        return jsonify({"error": "No sensor data received yet. Wait for Pico W reading."}), 400

    if ph < 0 or ph > 14:
        return jsonify({"error": "Invalid pH value. Enter between 0 and 14."}), 400

    # Run full analysis with pH
    result = run_full_analysis(
        ph             = ph,
        tds            = pending_reading["tds"],
        temp           = pending_reading["temperature"],
        turbidity      = pending_reading["turbidity"],
        uv             = pending_reading["uv_fluorescence"],
        vapor          = pending_reading["vapor_level"],
        r              = pending_reading["color_r"],
        g              = pending_reading["color_g"],
        b              = pending_reading["color_b"],
        session_id     = pending_reading["session_id"]
    )

    # Save to Supabase
    supabase.table("coolant_readings").insert({
        "session_id":           result["session_id"],
        "ph_level":             ph,
        "tds_value":            result["tds_value"],
        "temperature":          result["temperature"],
        "turbidity":            result["turbidity"],
        "color_r":              result["color_r"],
        "color_g":              result["color_g"],
        "color_b":              result["color_b"],
        "color_hex":            result["color_hex"],
        "uv_fluorescence":      result["uv_fluorescence"],
        "vapor_level":          result["vapor_level"],
        "coolant_color_name":   result["coolant_color_name"],
        "condition":            result["condition"],
        "application_suggestions": result["applications"],
        "diagnosis":            result["diagnosis"],
        "remedy":               result["remedy"]
    }).execute()

    supabase.table("analysis_history").insert({
        "session_id":      result["session_id"],
        "summary":         result["condition"],
        "top_application": result["applications"][0]["name"]
                           if result["applications"] else "General Industrial",
        "overall_score":   result["score"]
    }).execute()

    latest_reading = result
    pending_reading = {}

    return jsonify({
        "status":       "OK",
        "score":        result["score"],
        "condition":    result["condition"],
        "applications": result["applications"],
        "color_name":   result["coolant_color_name"],
        "diagnosis":    result["diagnosis"],
        "remedy":       result["remedy"]
    }), 200

# ─── Old analyze route kept for manual testing ──────────────────
@app.route("/analyze", methods=["POST"])
def analyze():
    global latest_reading
    data = request.get_json()

    ph        = float(data.get("ph",            7.0))
    tds       = float(data.get("tds",           500))
    temp      = float(data.get("temperature",    25))
    turbidity = float(data.get("turbidity",      20))
    r         = int(data.get("color_r",         200))
    g         = int(data.get("color_g",         200))
    b         = int(data.get("color_b",         200))
    uv        = float(data.get("uv_fluorescence", 50))
    vapor     = float(data.get("vapor_level",   100))
    session_id = data.get("session_id",
                           str(uuid.uuid4())[:8])

    result = run_full_analysis(ph, tds, temp, turbidity,
                                uv, vapor, r, g, b, session_id)

    supabase.table("coolant_readings").insert({
        "session_id":           result["session_id"],
        "ph_level":             ph,
        "tds_value":            result["tds_value"],
        "temperature":          result["temperature"],
        "turbidity":            result["turbidity"],
        "color_r":              result["color_r"],
        "color_g":              result["color_g"],
        "color_b":              result["color_b"],
        "color_hex":            result["color_hex"],
        "uv_fluorescence":      result["uv_fluorescence"],
        "vapor_level":          result["vapor_level"],
        "coolant_color_name":   result["coolant_color_name"],
        "condition":            result["condition"],
        "application_suggestions": result["applications"],
        "diagnosis":            result["diagnosis"],
        "remedy":               result["remedy"]
    }).execute()

    supabase.table("analysis_history").insert({
        "session_id":      result["session_id"],
        "summary":         result["condition"],
        "top_application": result["applications"][0]["name"]
                           if result["applications"] else "General Industrial",
        "overall_score":   result["score"]
    }).execute()

    latest_reading = result
    return jsonify({
        "status":       "OK",
        "score":        result["score"],
        "condition":    result["condition"],
        "applications": result["applications"]
    }), 200

@app.route("/latest")
def latest():
    return jsonify(latest_reading)

@app.route("/pending")
def pending():
    return jsonify(pending_reading)

@app.route("/history")
def history():
    data = supabase.table("coolant_readings").select("*")\
        .order("recorded_at", desc=True).limit(20).execute().data
    return jsonify(data)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=False)