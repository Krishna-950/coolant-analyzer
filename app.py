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

# ─── Color Name Detection ────────────────────────────────────────
def detect_color_name(r, g, b):
    colors = {
        "Crystal Clear":   (255, 255, 255),
        "Light Green":     (144, 238, 144),
        "Green":           (0,   128,  0),
        "Yellow-Green":    (154, 205,  50),
        "Orange":          (255, 165,   0),
        "Red":             (255,   0,   0),
        "Blue":            (0,     0, 255),
        "Light Blue":      (173, 216, 230),
        "Milky White":     (245, 245, 220),
        "Dark Amber":      (139,  90,  43),
        "Brown":           (139,  69,  19),
        "Dark Brown":      ( 92,  64,  51),
        "Purple":          (128,   0, 128),
        "Pink":            (255, 192, 203),
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

    # pH scoring
    if 7.0 <= ph <= 9.5:
        pass
    elif 6.5 <= ph < 7.0 or 9.5 < ph <= 10.5:
        score -= 15
    else:
        score -= 35

    # TDS scoring
    if tds < 500:
        pass
    elif tds < 1500:
        score -= 10
    elif tds < 3000:
        score -= 25
    else:
        score -= 40

    # Turbidity scoring
    if turbidity < 30:
        pass
    elif turbidity < 60:
        score -= 15
    else:
        score -= 30

    # UV fluorescence
    if uv < 100:
        pass
    elif uv < 300:
        score -= 10
    else:
        score -= 20

    # Vapor
    if vapor < 200:
        pass
    elif vapor < 400:
        score -= 10
    else:
        score -= 20

    if score >= 80:
        return "Excellent", score
    elif score >= 60:
        return "Good", score
    elif score >= 40:
        return "Fair - Monitor Closely", score
    elif score >= 20:
        return "Poor - Replace Soon", score
    else:
        return "Critical - Replace Immediately", score

# ─── Application Matching ────────────────────────────────────────
def match_applications(ph, tds, temp, turbidity, r, g, b, uv, vapor):
    color_name = detect_color_name(r, g, b)
    results = []

    apps = [
        {
            "name": "CNC Machining Fluid",
            "icon": "⚙️",
            "category": "Industrial Machining",
            "criteria": lambda: (8.5<=ph<=9.5 and tds<500 and turbidity<30),
            "description": "Ideal for precision CNC operations requiring clean, stable coolant.",
            "parameters": "pH 8.5–9.5, TDS <500, Low turbidity"
        },
        {
            "name": "Petrol Engine Coolant",
            "icon": "🚗",
            "category": "Automotive",
            "criteria": lambda: (7.0<=ph<=8.5 and 300<=tds<=1500 and temp<100),
            "description": "Suitable for petrol-powered passenger vehicles and light trucks.",
            "parameters": "pH 7–8.5, TDS 300–1500, Temp <100°C"
        },
        {
            "name": "Diesel Engine Coolant",
            "icon": "🚛",
            "category": "Heavy Duty Automotive",
            "criteria": lambda: (7.5<=ph<=9.0 and tds>=500 and temp>=80),
            "description": "Compatible with high-temperature diesel engine requirements.",
            "parameters": "pH 7.5–9, TDS >500, High temp tolerance"
        },
        {
            "name": "Hydraulic System Fluid",
            "icon": "🔧",
            "category": "Industrial Hydraulics",
            "criteria": lambda: (turbidity<20 and uv<100 and tds<300),
            "description": "Ultra-clean fluid meeting hydraulic system purity requirements.",
            "parameters": "Very low turbidity, No fluorescence, Low TDS"
        },
        {
            "name": "Brake System Fluid",
            "icon": "🛑",
            "category": "Automotive Safety",
            "criteria": lambda: (tds<200 and uv<80 and 6.5<=ph<=7.5),
            "description": "High purity fluid for critical brake system applications.",
            "parameters": "Low TDS, Clear, Neutral pH, No oil"
        },
        {
            "name": "Air Compressor Coolant",
            "icon": "💨",
            "category": "Industrial Pneumatics",
            "criteria": lambda: (6.8<=ph<=7.5 and tds<400 and turbidity<25),
            "description": "Clean neutral coolant for air compression systems.",
            "parameters": "Neutral pH, Low TDS, Clear color"
        },
        {
            "name": "Gearbox Lubricant",
            "icon": "⚡",
            "category": "Industrial Transmission",
            "criteria": lambda: (tds>=1000 and vapor>=200),
            "description": "High TDS fluid with vapor signature for gear lubrication.",
            "parameters": "High TDS, Dark amber color, Strong vapor"
        },
        {
            "name": "EDM Machine Fluid",
            "icon": "🔬",
            "category": "Precision Manufacturing",
            "criteria": lambda: (tds<100 and turbidity<10 and uv<50),
            "description": "Ultra-pure dielectric fluid for electrical discharge machining.",
            "parameters": "Ultra low TDS, Perfectly clear, No fluorescence"
        },
        {
            "name": "Heat Exchanger Fluid",
            "icon": "🌡️",
            "category": "Industrial Thermal",
            "criteria": lambda: (7.5<=ph<=9.0 and 500<=tds<=2000 and temp>=60),
            "description": "Thermally stable fluid for industrial heat exchange systems.",
            "parameters": "pH 7.5–9, Medium TDS, High temp"
        },
        {
            "name": "Metal Forming Fluid",
            "icon": "🏭",
            "category": "Metal Processing",
            "criteria": lambda: (7.0<=ph<=9.0 and 400<=tds<=1500),
            "description": "Milky emulsion fluid for metal forming and pressing operations.",
            "parameters": "Milky color, Medium TDS, Medium pH"
        },
        {
            "name": "EV Battery Cooling",
            "icon": "🔋",
            "category": "Electric Vehicles",
            "criteria": lambda: (tds<150 and uv<50 and 6.8<=ph<=7.2),
            "description": "Ultra-pure dielectric coolant for EV battery thermal management.",
            "parameters": "Ultra low TDS, Zero fluorescence, Neutral pH"
        },
        {
            "name": "Food Grade Cooling",
            "icon": "🍃",
            "category": "Food Processing",
            "criteria": lambda: (6.5<=ph<=7.5 and tds<300 and turbidity<15 and vapor<150),
            "description": "NSF-compliant coolant for food and beverage processing equipment.",
            "parameters": "pH 6.5–7.5, Very low TDS, Crystal clear"
        },
        {
            "name": "Pharmaceutical Cooling",
            "icon": "💊",
            "category": "Pharmaceutical",
            "criteria": lambda: (6.8<=ph<=7.2 and tds<100 and turbidity<5),
            "description": "Ultra-pure coolant meeting pharmaceutical manufacturing standards.",
            "parameters": "Ultrapure, Neutral pH, Crystal clear"
        },
        {
            "name": "Solar Panel Cooling",
            "icon": "☀️",
            "category": "Renewable Energy",
            "criteria": lambda: (6.5<=ph<=8.0 and tds<600 and turbidity<30),
            "description": "Corrosion-inhibited fluid for concentrated solar power systems.",
            "parameters": "Low TDS, Mild pH, Low turbidity"
        },
        {
            "name": "Nuclear Plant Cooling",
            "icon": "⚛️",
            "category": "Nuclear Energy",
            "criteria": lambda: (tds<50 and turbidity<5 and 6.9<=ph<=7.1),
            "description": "Demineralised water for reactor secondary cooling circuits.",
            "parameters": "Near-zero TDS, Perfect clarity, Neutral pH"
        },
        {
            "name": "Data Center Cooling",
            "icon": "🖥️",
            "category": "IT Infrastructure",
            "criteria": lambda: (tds<200 and uv<80 and turbidity<20),
            "description": "Non-conductive dielectric coolant for server immersion cooling.",
            "parameters": "Low TDS, No fluorescence, Crystal clear"
        },
        {
            "name": "Marine Engine Coolant",
            "icon": "⛵",
            "category": "Marine",
            "criteria": lambda: (7.5<=ph<=10.5 and tds>=500),
            "description": "Salt-water resistant coolant for marine propulsion systems.",
            "parameters": "pH 7.5–10.5, Medium TDS, Corrosion resistant"
        },
        {
            "name": "Aircraft Engine Coolant",
            "icon": "✈️",
            "category": "Aerospace",
            "criteria": lambda: (8.0<=ph<=11.0 and turbidity<15 and temp>=80),
            "description": "High-performance coolant for aircraft turbine and piston engines.",
            "parameters": "pH 8–11, High temp, Low turbidity"
        },
        {
            "name": "General Industrial Use",
            "icon": "🏗️",
            "category": "General",
            "criteria": lambda: (6.0<=ph<=10.0 and tds<3000),
            "description": "Broad-spectrum coolant for general industrial cooling applications.",
            "parameters": "Mid-range all parameters"
        },
    ]

    for a in apps:
        try:
            if a["criteria"]():
                results.append({
                    "name": a["name"],
                    "icon": a["icon"],
                    "category": a["category"],
                    "description": a["description"],
                    "parameters": a["parameters"]
                })
        except:
            pass

    return results, color_name

# ─── Diagnosis and Remedy ────────────────────────────────────────
def get_diagnosis_remedy(ph, tds, temp, turbidity, uv, vapor):
    issues = []
    remedies = []

    if ph < 6.5:
        issues.append("Coolant is highly acidic — risk of corrosion to metal components")
        remedies.append("Add alkaline buffer or pH booster. Consider full coolant replacement.")
    elif ph > 10.5:
        issues.append("Coolant is excessively alkaline — may cause scaling and deposits")
        remedies.append("Dilute with distilled water. Add pH stabiliser.")
    elif ph < 7.0:
        issues.append("Slightly acidic coolant detected")
        remedies.append("Monitor closely. Add corrosion inhibitor package.")

    if tds > 3000:
        issues.append("Extremely high dissolved solids — severe contamination detected")
        remedies.append("Immediate coolant replacement recommended. Flush system thoroughly.")
    elif tds > 1500:
        issues.append("Elevated TDS levels indicating mineral buildup or contamination")
        remedies.append("Partial coolant replacement with distilled water top-up.")

    if turbidity > 60:
        issues.append("High turbidity — coolant heavily contaminated with particles")
        remedies.append("Filter coolant through fine mesh filter. Check for seal leaks.")
    elif turbidity > 30:
        issues.append("Moderate turbidity detected")
        remedies.append("Apply inline filtration. Monitor particle source.")

    if uv > 300:
        issues.append("Strong UV fluorescence — oil or hydraulic fluid contamination detected")
        remedies.append("Isolate contamination source. Full system drain and clean required.")
    elif uv > 100:
        issues.append("Mild UV fluorescence — possible trace oil contamination")
        remedies.append("Check seals and gaskets. Monitor fluorescence trend.")

    if vapor > 400:
        issues.append("High chemical vapor signature — volatile compounds present")
        remedies.append("Ensure proper ventilation. Check coolant compatibility.")

    if temp > 90:
        issues.append("Very high coolant temperature detected")
        remedies.append("Check pump flow rate. Inspect heat exchanger efficiency.")

    if not issues:
        issues.append("All parameters within acceptable range")
        remedies.append("Coolant is in good condition. Continue regular monitoring every 500 hours or 3 months.")

    return " | ".join(issues), " | ".join(remedies)

# ─── Routes ──────────────────────────────────────────────────────
@app.route("/")
def index():
    history = supabase.table("analysis_history").select("*").order("created_at", desc=True).limit(10).execute().data
    return render_template("index.html", history=history)

@app.route("/analyze", methods=["POST"])
def analyze():
    global latest_reading
    data = request.get_json()

    ph        = float(data.get("ph", 7.0))
    tds       = float(data.get("tds", 500))
    temp      = float(data.get("temperature", 25.0))
    turbidity = float(data.get("turbidity", 20))
    r         = int(data.get("color_r", 200))
    g         = int(data.get("color_g", 200))
    b         = int(data.get("color_b", 200))
    uv        = float(data.get("uv_fluorescence", 50))
    vapor     = float(data.get("vapor_level", 100))
    session_id = data.get("session_id", str(uuid.uuid4())[:8])

    color_hex  = f"#{r:02x}{g:02x}{b:02x}"
    color_name = detect_color_name(r, g, b)
    condition, score = assess_condition(ph, tds, temp, turbidity, uv, vapor)
    apps, _ = match_applications(ph, tds, temp, turbidity, r, g, b, uv, vapor)
    diagnosis, remedy = get_diagnosis_remedy(ph, tds, temp, turbidity, uv, vapor)

    reading = {
        "session_id":          session_id,
        "ph_level":            ph,
        "tds_value":           tds,
        "temperature":         temp,
        "turbidity":           turbidity,
        "color_r":             r,
        "color_g":             g,
        "color_b":             b,
        "color_hex":           color_hex,
        "uv_fluorescence":     uv,
        "vapor_level":         vapor,
        "coolant_color_name":  color_name,
        "condition":           condition,
        "application_suggestions": apps,
        "diagnosis":           diagnosis,
        "remedy":              remedy
    }

    supabase.table("coolant_readings").insert(reading).execute()

    if apps:
        supabase.table("analysis_history").insert({
            "session_id":    session_id,
            "summary":       condition,
            "top_application": apps[0]["name"] if apps else "General Industrial",
            "overall_score": score
        }).execute()

    latest_reading = {**reading, "score": score, "applications": apps}
    return jsonify({"status": "OK", "score": score, "condition": condition, "applications": apps}), 200

@app.route("/latest")
def latest():
    return jsonify(latest_reading)

@app.route("/history")
def history():
    data = supabase.table("coolant_readings").select("*").order("recorded_at", desc=True).limit(20).execute().data
    return jsonify(data)

@app.route("/clear_latest", methods=["POST"])
def clear_latest():
    global latest_reading
    latest_reading = {}
    return jsonify({"status": "cleared"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=False)