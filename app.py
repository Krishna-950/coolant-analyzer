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
# ─── Application Matching (Industry Standard Ranges) ─────────────
def match_applications(ph, tds, temp, turbidity, r, g, b, uv, vapor):
    color_name = detect_color_name(r, g, b)
    results = []

    apps = [
        {
            # ASTM B860 / MSC Industrial Standard
            # pH >8.6 to inhibit bacterial growth (MSC Industrial)
            # TDS <500 ppm for clean machining
            # Turbidity <20 NTU for precision cutting
            # Temp 20–50°C normal operating range
            "name": "CNC Machining Fluid",
            "icon": "⚙️",
            "category": "Industrial Machining",
            "criteria": lambda: (
                8.6 <= ph <= 9.5 and
                tds < 500 and
                turbidity < 20 and
                20 <= temp <= 50 and
                vapor < 300
            ),
            "description": "Precision CNC operations require stable alkaline coolant to prevent bacterial growth and corrosion on metal surfaces.",
            "parameters": "pH 8.6–9.5, TDS <500 ppm, Turbidity <20 NTU, Temp 20–50°C"
        },
        {
            # ASTM D3306 Standard — Light Duty Petrol Engine
            # pH 7.5–11.0 (ASTM D3306 specification)
            # TDS 150–800 ppm typical in-service coolant
            # Temp operating range 80–105°C
            # Turbidity <10 NTU for new coolant
            "name": "Petrol Engine Coolant",
            "icon": "🚗",
            "category": "Automotive — Light Duty",
            "criteria": lambda: (
                7.5 <= ph <= 11.0 and
                150 <= tds <= 800 and
                turbidity < 15 and
                uv < 150 and
                vapor < 250
            ),
            "description": "Ethylene glycol based coolant for petrol passenger vehicles per ASTM D3306. Provides freeze and corrosion protection.",
            "parameters": "pH 7.5–11.0 (ASTM D3306), TDS 150–800 ppm, Turbidity <15 NTU"
        },
        {
            # ASTM D6210 Standard — Heavy Duty Diesel Engine
            # pH 8.0–10.5 (ASTM D6210 fully formulated)
            # TDS 300–1500 ppm (higher due to inhibitor package)
            # Higher temp tolerance up to 120°C
            "name": "Diesel Engine Coolant",
            "icon": "🚛",
            "category": "Automotive — Heavy Duty",
            "criteria": lambda: (
                8.0 <= ph <= 10.5 and
                300 <= tds <= 1500 and
                turbidity < 20 and
                uv < 200 and
                vapor < 300
            ),
            "description": "Fully formulated glycol coolant for heavy duty diesel engines per ASTM D6210. Contains nitrite/molybdate inhibitors.",
            "parameters": "pH 8.0–10.5 (ASTM D6210), TDS 300–1500 ppm, Turbidity <20 NTU"
        },
        {
            # ISO 11158 / DIN 51524 Hydraulic Fluid Standard
            # pH 6.5–8.5 for mineral oil based hydraulic fluid
            # TDS <200 ppm (extremely clean requirement)
            # Turbidity <5 NTU — very strict clarity needed
            # UV fluorescence very low — no oil contamination
            "name": "Hydraulic System Fluid",
            "icon": "🔧",
            "category": "Industrial Hydraulics",
            "criteria": lambda: (
                6.5 <= ph <= 8.5 and
                tds < 200 and
                turbidity < 5 and
                uv < 80 and
                vapor < 200
            ),
            "description": "Ultra-clean hydraulic fluid per ISO 11158. Strict purity requirements to protect precision hydraulic components.",
            "parameters": "pH 6.5–8.5 (ISO 11158), TDS <200 ppm, Turbidity <5 NTU, No fluorescence"
        },
        {
            # FMVSS 116 / SAE J1703 Brake Fluid Standard
            # pH 7.0–11.5 (SAE J1703)
            # TDS <100 ppm — extremely pure
            # Zero turbidity and UV — no contamination at all
            "name": "Brake System Fluid",
            "icon": "🛑",
            "category": "Automotive Safety",
            "criteria": lambda: (
                7.0 <= ph <= 11.5 and
                tds < 100 and
                turbidity < 3 and
                uv < 50 and
                vapor < 150
            ),
            "description": "DOT-grade brake fluid per SAE J1703/FMVSS 116. Extremely high purity required for safety-critical brake systems.",
            "parameters": "pH 7.0–11.5 (SAE J1703), TDS <100 ppm, Turbidity <3 NTU"
        },
        {
            # ISO 6743-3A Air Compressor Lubricant Standard
            # pH 6.5–8.0 neutral to slightly alkaline
            # TDS <400 ppm clean requirement
            # Turbidity <15 NTU
            "name": "Air Compressor Coolant",
            "icon": "💨",
            "category": "Industrial Pneumatics",
            "criteria": lambda: (
                6.5 <= ph <= 8.0 and
                tds < 400 and
                turbidity < 15 and
                uv < 100 and
                vapor < 300
            ),
            "description": "Clean neutral coolant for rotary screw and piston air compressors per ISO 6743-3A standards.",
            "parameters": "pH 6.5–8.0 (ISO 6743), TDS <400 ppm, Turbidity <15 NTU"
        },
        {
            # ISO 3448 / DIN 51517 Gear Oil Standard
            # TDS >1500 ppm (high additive content)
            # Vapor >300 ppm (EP additives signature)
            # Higher turbidity acceptable — typically amber colored
            "name": "Gearbox Lubricant",
            "icon": "⚙️",
            "category": "Industrial Transmission",
            "criteria": lambda: (
                5.5 <= ph <= 8.5 and
                tds >= 1500 and
                vapor >= 300 and
                temp <= 120
            ),
            "description": "High additive gear lubricant per ISO 3448/DIN 51517. Contains extreme pressure additives producing vapor signature.",
            "parameters": "TDS >1500 ppm (ISO 3448), Vapor >300 ppm, pH 5.5–8.5"
        },
        {
            # ISO 4370 / JIS B 4200 EDM Dielectric Standard
            # TDS <50 ppm — deionized water requirement
            # Turbidity <2 NTU — crystal clear mandatory
            # UV <30 — absolutely no contamination
            # pH 6.5–7.5 near neutral
            "name": "EDM Machine Fluid",
            "icon": "🔬",
            "category": "Precision Manufacturing",
            "criteria": lambda: (
                6.5 <= ph <= 7.5 and
                tds < 50 and
                turbidity < 2 and
                uv < 30 and
                vapor < 100
            ),
            "description": "Deionised dielectric fluid for EDM per ISO 4370. Ultra-pure water with near-zero conductivity mandatory.",
            "parameters": "pH 6.5–7.5 (ISO 4370), TDS <50 ppm, Turbidity <2 NTU, UV <30"
        },
        {
            # TEMA / ASHRAE Heat Exchanger Standards
            # pH 7.5–9.5 to prevent scaling and corrosion
            # TDS 500–3000 ppm (HVAC standard allows up to 3000)
            # Turbidity <15 NTU (HVAC specification)
            # Temp must be elevated — heat exchanger application
            "name": "Heat Exchanger Fluid",
            "icon": "🌡️",
            "category": "Industrial Thermal",
            "criteria": lambda: (
                7.5 <= ph <= 9.5 and
                500 <= tds <= 3000 and
                turbidity < 15 and
                temp >= 50 and
                uv < 200
            ),
            "description": "Inhibited glycol or water-based fluid for shell-and-tube heat exchangers per TEMA/ASHRAE standards.",
            "parameters": "pH 7.5–9.5 (TEMA), TDS 500–3000 ppm, Turbidity <15 NTU, Temp >50°C"
        },
        {
            # ISO 6743-7 Metal Forming Fluid Standard
            # pH 7.0–9.5 (emulsion stability range)
            # TDS 400–2000 ppm (emulsion coolant range)
            # Turbidity 20–80 NTU (milky emulsion is expected)
            "name": "Metal Forming Fluid",
            "icon": "🏭",
            "category": "Metal Processing",
            "criteria": lambda: (
                7.0 <= ph <= 9.5 and
                400 <= tds <= 2000 and
                20 <= turbidity <= 80 and
                vapor < 400
            ),
            "description": "Semisynthetic or soluble oil emulsion for metal forming and stamping per ISO 6743-7.",
            "parameters": "pH 7.0–9.5 (ISO 6743-7), TDS 400–2000 ppm, Turbidity 20–80 NTU (milky)"
        },
        {
            # ASTM D1816 / SAE J2800 EV Dielectric Coolant
            # TDS <100 ppm — low conductivity critical
            # UV <50 — dielectric purity mandatory
            # pH 6.8–7.5 near neutral
            # Turbidity <5 NTU — optically clear
            "name": "EV Battery Cooling",
            "icon": "🔋",
            "category": "Electric Vehicles",
            "criteria": lambda: (
                6.8 <= ph <= 7.5 and
                tds < 100 and
                turbidity < 5 and
                uv < 50 and
                vapor < 150
            ),
            "description": "Dielectric immersion coolant for EV battery thermal management per SAE J2800/ASTM D1816.",
            "parameters": "pH 6.8–7.5 (SAE J2800), TDS <100 ppm, Turbidity <5 NTU, Zero fluorescence"
        },
        {
            # NSF/ANSI 169 Food Grade Coolant Standard
            # pH 6.5–7.5 food safe range
            # TDS <200 ppm (food contact surface water standard)
            # Turbidity <1 NTU (food industry standard)
            # Absolutely no vapor or fluorescence
            "name": "Food Grade Cooling",
            "icon": "🍃",
            "category": "Food & Beverage Processing",
            "criteria": lambda: (
                6.5 <= ph <= 7.5 and
                tds < 200 and
                turbidity < 1 and
                uv < 30 and
                vapor < 100
            ),
            "description": "NSF/ANSI 169 certified propylene glycol coolant for food contact equipment. Zero toxicity mandatory.",
            "parameters": "pH 6.5–7.5 (NSF/ANSI 169), TDS <200 ppm, Turbidity <1 NTU"
        },
        {
            # USP Water for Injection / FDA 21 CFR Pharmaceutical
            # pH 6.8–7.2 extremely tight neutral range
            # TDS <10 ppm — WFI standard
            # Turbidity <0.5 NTU — optically pure
            # Zero UV, zero vapor — absolute purity
            "name": "Pharmaceutical Cooling",
            "icon": "💊",
            "category": "Pharmaceutical Manufacturing",
            "criteria": lambda: (
                6.8 <= ph <= 7.2 and
                tds < 10 and
                turbidity < 1 and
                uv < 20 and
                vapor < 80
            ),
            "description": "Water for Injection grade coolant per USP/FDA 21 CFR. Ultra-high purity for pharmaceutical manufacturing.",
            "parameters": "pH 6.8–7.2 (USP WFI), TDS <10 ppm, Turbidity <1 NTU"
        },
        {
            # ASTM E2277 Solar Thermal Fluid Standard
            # pH 7.0–8.5 for propylene glycol solar fluid
            # TDS <500 ppm
            # Turbidity <10 NTU
            "name": "Solar Panel Cooling",
            "icon": "☀️",
            "category": "Renewable Energy",
            "criteria": lambda: (
                7.0 <= ph <= 8.5 and
                tds < 500 and
                turbidity < 10 and
                uv < 120 and
                vapor < 200
            ),
            "description": "Propylene glycol solar thermal fluid per ASTM E2277. UV-stable formulation for concentrated solar power.",
            "parameters": "pH 7.0–8.5 (ASTM E2277), TDS <500 ppm, Turbidity <10 NTU"
        },
        {
            # IAEA / NRC Nuclear Grade Water Standard
            # TDS <5 ppm — demineralised water
            # Turbidity <0.5 NTU — virtually particle free
            # pH 6.9–7.1 extremely tight neutral
            # Zero UV, zero vapor — absolutely pure
            "name": "Nuclear Plant Cooling",
            "icon": "⚛️",
            "category": "Nuclear Energy",
            "criteria": lambda: (
                6.9 <= ph <= 7.1 and
                tds < 5 and
                turbidity < 1 and
                uv < 10 and
                vapor < 50
            ),
            "description": "Demineralised water for secondary cooling circuits per IAEA safety standards. Extreme purity mandatory.",
            "parameters": "pH 6.9–7.1 (IAEA), TDS <5 ppm, Turbidity <1 NTU"
        },
        {
            # ASHRAE TC 9.9 / IEC 60068 Data Center Cooling
            # TDS <100 ppm — non-conductive requirement
            # Turbidity <5 NTU — server-safe clarity
            # UV <50 — no oil contamination
            # pH 7.0–8.5
            "name": "Data Center Cooling",
            "icon": "🖥️",
            "category": "IT Infrastructure",
            "criteria": lambda: (
                7.0 <= ph <= 8.5 and
                tds < 100 and
                turbidity < 5 and
                uv < 50 and
                vapor < 100
            ),
            "description": "Non-conductive dielectric coolant for server immersion cooling per ASHRAE TC 9.9 / IEC 60068.",
            "parameters": "pH 7.0–8.5 (ASHRAE), TDS <100 ppm, Turbidity <5 NTU, No fluorescence"
        },
        {
            # IACS / IMO Marine Engine Coolant Standard
            # pH 7.5–10.5 (salt water corrosion resistance)
            # TDS 500–2000 ppm (marine inhibitor package)
            # Turbidity <20 NTU
            "name": "Marine Engine Coolant",
            "icon": "⛵",
            "category": "Marine",
            "criteria": lambda: (
                7.5 <= ph <= 10.5 and
                500 <= tds <= 2000 and
                turbidity < 20 and
                uv < 200 and
                temp <= 110
            ),
            "description": "Corrosion-inhibited glycol coolant for marine propulsion per IACS/IMO standards.",
            "parameters": "pH 7.5–10.5 (IACS/IMO), TDS 500–2000 ppm, Turbidity <20 NTU"
        },
        {
            # MIL-PRF-23699 / DEF STAN 91-098 Aircraft Coolant
            # pH 8.0–11.0 (MIL-PRF-23699 specification)
            # Turbidity <10 NTU (aerospace clarity standard)
            # Temp must be high — aircraft engine application
            # TDS 200–1000 ppm
            "name": "Aircraft Engine Coolant",
            "icon": "✈️",
            "category": "Aerospace",
            "criteria": lambda: (
                8.0 <= ph <= 11.0 and
                200 <= tds <= 1000 and
                turbidity < 10 and
                temp >= 60 and
                uv < 150
            ),
            "description": "High-performance coolant for aircraft turbine engines per MIL-PRF-23699 / DEF STAN 91-098.",
            "parameters": "pH 8.0–11.0 (MIL-PRF-23699), TDS 200–1000 ppm, Turbidity <10 NTU"
        },
        {
            # General Industrial — Broad catch-all range
            # Based on WHO industrial water quality guidelines
            "name": "General Industrial Use",
            "icon": "🏗️",
            "category": "General Industrial",
            "criteria": lambda: (
                6.0 <= ph <= 10.0 and
                tds < 3000 and
                turbidity < 50
            ),
            "description": "Broad-spectrum coolant for general industrial applications per WHO industrial water quality guidelines.",
            "parameters": "pH 6.0–10.0 (WHO), TDS <3000 ppm, Turbidity <50 NTU"
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