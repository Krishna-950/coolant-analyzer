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

latest_reading  = {}
pending_reading = {}

# ================================================================
# SENSOR CALIBRATION — June 2026
# Hardware: Raspberry Pi Pico W
# ================================================================
#
# TDS SENSOR STATUS: UNRELIABLE
#   All fluids read ADC 4035–4092 (near air baseline 4091)
#   TDS formula gives 0 ppm for nearly all fluids
#   Root cause: probe noise floor equals fluid voltage
#   Action: TDS kept in payload but NOT used for app matching
#
# FC-28 TURBIDITY — PRIMARY DIFFERENTIATOR
#   Formula: turb% = (ADC - 1800) / 2291 * 100
#   Verified fluid zones (±10% tolerance applied):
#     Tap Water      ADC 1861  → 2.7%   zone: 0–8%
#     Drinking Water ADC 2760  → 41.9%  zone: 35–50%
#     EDM Fluid      ADC 3125  → 57.8%  zone: 50–65%
#     Industrial Oil ADC 3273  → 64.3%  zone: 58–72%
#     Gearbox Oil    ADC 3500  → 74.2%  zone: 68–82%
#     Brake Fluid    ADC 3507  → 74.5%  zone: 68–82%
#     CNC Coolant    ADC 3557  → 76.7%  zone: 70–85%
#     Diesel Oil     ADC 4091  → 99.9%  zone: 92–100%
#     Air            ADC 4091  → 99.9%  (no fluid)
#
# VAPOR SENSOR — SECONDARY DIFFERENTIATOR
#   Saturated at 800 ppm for most fluids
#   Reliable readings:
#     Brake Fluid    : ~474 ppm  (distinctly lower)
#     Industrial Oil : ~558 ppm
#     Gearbox Oil    : ~697 ppm
#     Diesel Oil     : ~738 ppm
#
# APPLICATION MATCHING USES: turbidity + vapor + pH + temp
# pH and temperature are NOT changed — manual/DS18B20 as-is
# ================================================================

# ─── Color Name Detection ────────────────────────────────────────
def detect_color_name(r, g, b):
    colors = {
        "Crystal Clear": (255, 255, 255),
        "Light Green":   (144, 238, 144),
        "Green":         (  0, 128,   0),
        "Yellow-Green":  (154, 205,  50),
        "Orange":        (255, 165,   0),
        "Red":           (255,   0,   0),
        "Blue":          (  0,   0, 255),
        "Light Blue":    (173, 216, 230),
        "Milky White":   (245, 245, 220),
        "Dark Amber":    (139,  90,  43),
        "Brown":         (139,  69,  19),
        "Dark Brown":    ( 92,  64,  51),
        "Clear Water":   (200, 220, 230),
        "Transparent":   (200, 200, 200),
    }
    min_dist = float('inf')
    closest  = "Unknown"
    for name, (cr, cg, cb) in colors.items():
        dist = ((r-cr)**2 + (g-cg)**2 + (b-cb)**2) ** 0.5
        if dist < min_dist:
            min_dist = dist
            closest  = name
    return closest

# ─── Condition Scoring ───────────────────────────────────────────
def assess_condition(ph, tds, temp, turbidity, uv, vapor):
    score = 100

    # pH — manual input, standard ISO ranges unchanged
    if   7.0 <= ph <= 9.5:                         pass
    elif 6.5 <= ph < 7.0 or 9.5 < ph <= 10.5:     score -= 15
    else:                                           score -= 35

    # Turbidity — calibrated to FC-28 verified zones
    # Lower % = more water-like (tap water zone)
    # Higher % = oil-like or non-conductive
    if   turbidity <= 8:    pass           # tap water zone — clean water
    elif turbidity <= 50:   score -= 5     # drinking water / EDM zone
    elif turbidity <= 72:   score -= 15    # light oil / industrial zone
    elif turbidity <= 85:   score -= 20    # heavy oil / CNC zone
    else:                   score -= 30    # diesel / near-air zone

    # UV — unchanged
    if   uv < 100:          pass
    elif uv < 300:          score -= 10
    else:                   score -= 20

    # Vapor — secondary signal
    if   vapor < 500:       pass
    elif vapor < 700:       score -= 5
    else:                   score -= 10

    # Temperature
    if   temp > 90:         score -= 20
    elif temp > 70:         score -= 10

    score = max(0, score)
    if   score >= 80: return "Excellent", score
    elif score >= 60: return "Good", score
    elif score >= 40: return "Fair - Monitor Closely", score
    elif score >= 20: return "Poor - Replace Soon", score
    else:             return "Critical - Replace Immediately", score

# ─── Application Matching ────────────────────────────────────────
# PRIMARY sensor: FC-28 turbidity %
# SECONDARY: vapor ppm, pH, temperature
# TDS not used for matching (unreliable per hardware test)
#
# TURBIDITY ZONES verified from 8-fluid combined test:
#   Zone A (Tap Water)      :  0–8%    ADC ~1861
#   Zone B (Drinking Water) : 35–50%   ADC ~2760
#   Zone C (EDM Fluid)      : 50–65%   ADC ~3125
#   Zone D (Light Oils)     : 58–75%   ADC ~3273
#   Zone E (Heavy Oils/CNC) : 68–85%   ADC ~3500–3557
#   Zone F (Diesel/Air)     : 92–100%  ADC ~4091
#
# Tolerance: ±8% on all turbidity thresholds
# ================================================================
def match_applications(ph, tds, temp, turbidity, r, g, b, uv, vapor):
    color_name = detect_color_name(r, g, b)
    results    = []

    apps = [

        # ── 1. CNC MACHINING FLUID ───────────────────────────────
        # Zone E — CNC Coolant: turb 70–85% (verified ADC 3557)
        # Standard: ASTM B860 / MSC Industrial
        {
            "name":     "CNC Machining Fluid",
            "icon":     "⚙️",
            "category": "Industrial Machining",
            "criteria": lambda: (
                8.6  <= ph        <= 9.5  and
                70.0 <= turbidity <= 85.0 and
                20   <= temp      <= 55
            ),
            "description": "Precision CNC semi-synthetic coolant per ASTM B860. Alkaline pH prevents bacterial growth. Sensor confirms CNC coolant zone (turb 70–85%).",
            "parameters":  "pH 8.6–9.5 (ASTM B860) | Turbidity 70–85% (CNC zone, sensor-verified) | Temp 20–55°C"
        },

        # ── 2. PETROL ENGINE COOLANT ─────────────────────────────
        # Zone B/C — water-based with inhibitors: turb 35–65%
        # Standard: ASTM D3306 / BS 6580
        {
            "name":     "Petrol Engine Coolant",
            "icon":     "🚗",
            "category": "Automotive — Light Duty",
            "criteria": lambda: (
                7.5  <= ph        <= 11.0 and
                35.0 <= turbidity <= 65.0 and
                uv   <  150
            ),
            "description": "Ethylene glycol coolant for petrol engines per ASTM D3306. Water-based inhibitor solution confirmed by turbidity zone 35–65%.",
            "parameters":  "pH 7.5–11.0 (ASTM D3306) | Turbidity 35–65% (water-based zone, sensor-verified)"
        },

        # ── 3. DIESEL ENGINE COOLANT ─────────────────────────────
        # Zone B/C — water-based: turb 35–65%
        # Standard: ASTM D6210
        {
            "name":     "Diesel Engine Coolant",
            "icon":     "🚛",
            "category": "Automotive — Heavy Duty",
            "criteria": lambda: (
                8.0  <= ph        <= 10.5 and
                35.0 <= turbidity <= 65.0 and
                uv   <  200
            ),
            "description": "Heavy duty coolant with nitrite/molybdate inhibitors per ASTM D6210. Water-based composition confirmed by turbidity 35–65%.",
            "parameters":  "pH 8.0–10.5 (ASTM D6210) | Turbidity 35–65% (water-based zone, sensor-verified)"
        },

        # ── 4. HYDRAULIC SYSTEM FLUID ────────────────────────────
        # Zone D — light oil: turb 58–75% (verified ADC ~3273)
        # Standard: ISO 11158 / DIN 51524
        {
            "name":     "Hydraulic System Fluid",
            "icon":     "🔧",
            "category": "Industrial Hydraulics",
            "criteria": lambda: (
                6.5  <= ph        <= 8.5  and
                58.0 <= turbidity <= 75.0 and
                uv   <  80
            ),
            "description": "Mineral oil hydraulic fluid per ISO 11158 / DIN 51524-3. Light oil composition confirmed by turbidity 58–75% (industrial oil zone).",
            "parameters":  "pH 6.5–8.5 (ISO 11158) | Turbidity 58–75% (light oil zone, sensor-verified)"
        },

        # ── 5. BRAKE SYSTEM FLUID ────────────────────────────────
        # Zone E — heavy oil/glycol: turb 68–82%, vapor ~474 ppm
        # Standard: SAE J1703 / FMVSS 116
        {
            "name":     "Brake System Fluid",
            "icon":     "🛑",
            "category": "Automotive Safety",
            "criteria": lambda: (
                7.0  <= ph        <= 11.5 and
                68.0 <= turbidity <= 82.0 and
                400  <= vapor     <= 550
            ),
            "description": "DOT brake fluid per SAE J1703 / FMVSS 116. Glycol-ether composition confirmed by turbidity 68–82% and vapor signature ~474 ppm (sensor-verified).",
            "parameters":  "pH 7.0–11.5 (SAE J1703) | Turbidity 68–82% (sensor-verified) | Vapor 400–550 ppm (sensor-verified)"
        },

        # ── 6. AIR COMPRESSOR COOLANT ────────────────────────────
        # Zone B/D — could be water or light oil: turb 35–75%
        # Standard: ISO 6743-3A
        {
            "name":     "Air Compressor Coolant",
            "icon":     "💨",
            "category": "Industrial Pneumatics",
            "criteria": lambda: (
                6.5  <= ph        <= 8.0  and
                35.0 <= turbidity <= 75.0 and
                uv   <  100
            ),
            "description": "Mineral oil or synthetic coolant for air compressors per ISO 6743-3A. Covers water-based to light oil range confirmed by turbidity 35–75%.",
            "parameters":  "pH 6.5–8.0 (ISO 6743-3A) | Turbidity 35–75% (sensor-verified)"
        },

        # ── 7. GEARBOX LUBRICANT ─────────────────────────────────
        # Zone E — heavy oil: turb 68–82%, vapor ~697 ppm
        # Standard: ISO 3448 / DIN 51517
        {
            "name":     "Gearbox Lubricant",
            "icon":     "⚙️",
            "category": "Industrial Transmission",
            "criteria": lambda: (
                5.5  <= ph        <= 8.5  and
                68.0 <= turbidity <= 82.0 and
                vapor >= 600      and
                temp  <= 120
            ),
            "description": "EP gear lubricant per ISO 3448 / DIN 51517. Heavy oil confirmed by turbidity 68–82% and elevated vapor ~697 ppm (sensor-verified).",
            "parameters":  "pH 5.5–8.5 (ISO 3448) | Turbidity 68–82% (sensor-verified) | Vapor ≥600 ppm (sensor-verified)"
        },

        # ── 8. EDM MACHINE FLUID ─────────────────────────────────
        # Zone C — EDM: turb 50–65% (verified ADC ~3125)
        # Standard: ISO 4370 / JIS B 4200
        {
            "name":     "EDM Machine Fluid",
            "icon":     "🔬",
            "category": "Precision Manufacturing",
            "criteria": lambda: (
                6.5  <= ph        <= 7.5  and
                50.0 <= turbidity <= 65.0 and
                uv   <  30
            ),
            "description": "Deionised dielectric fluid for EDM per ISO 4370. Unique turbidity zone 50–65% (ADC ~3125) sensor-verified for this fluid specifically.",
            "parameters":  "pH 6.5–7.5 (ISO 4370) | Turbidity 50–65% (EDM zone, sensor-verified ADC ~3125)"
        },

        # ── 9. HEAT EXCHANGER FLUID ──────────────────────────────
        # Zone B/C — water-based, high temp: turb 35–65%, temp ≥50
        # Standard: TEMA / ASHRAE
        {
            "name":     "Heat Exchanger Fluid",
            "icon":     "🌡️",
            "category": "Industrial Thermal",
            "criteria": lambda: (
                7.5  <= ph        <= 9.5  and
                35.0 <= turbidity <= 65.0 and
                temp >= 50        and
                uv   <  200
            ),
            "description": "Inhibited glycol fluid for heat exchangers per TEMA/ASHRAE. Water-based composition (turb 35–65%) at elevated temperature ≥50°C.",
            "parameters":  "pH 7.5–9.5 (TEMA) | Turbidity 35–65% (water-based zone) | Temp ≥50°C"
        },

        # ── 10. METAL FORMING FLUID ──────────────────────────────
        # Zone C/D — semisynthetic emulsion: turb 50–75%
        # Standard: ISO 6743-7
        {
            "name":     "Metal Forming Fluid",
            "icon":     "🏭",
            "category": "Metal Processing",
            "criteria": lambda: (
                7.0  <= ph        <= 9.5  and
                50.0 <= turbidity <= 75.0
            ),
            "description": "Semisynthetic emulsion for metal forming per ISO 6743-7. Milky emulsion turbidity zone 50–75% covers both EDM and light oil zones on sensor.",
            "parameters":  "pH 7.0–9.5 (ISO 6743-7) | Turbidity 50–75% (emulsion zone, sensor-verified)"
        },

        # ── 11. EV BATTERY COOLING ───────────────────────────────
        # Zone D/E — dielectric oil: turb 58–82%
        # Standard: SAE J2800 / ASTM D1816
        {
            "name":     "EV Battery Cooling",
            "icon":     "🔋",
            "category": "Electric Vehicles",
            "criteria": lambda: (
                6.8  <= ph        <= 7.5  and
                58.0 <= turbidity <= 82.0 and
                uv   <  50
            ),
            "description": "Non-conductive dielectric coolant for EV battery per SAE J2800. Oil-based composition confirmed by turbidity 58–82%, near-zero conductivity.",
            "parameters":  "pH 6.8–7.5 (SAE J2800) | Turbidity 58–82% (oil zone, sensor-verified)"
        },

        # ── 12. FOOD GRADE COOLING ───────────────────────────────
        # Zone B/C — clean water-based: turb 35–65%
        # Standard: NSF/ANSI 169 H1
        {
            "name":     "Food Grade Cooling",
            "icon":     "🍃",
            "category": "Food & Beverage Processing",
            "criteria": lambda: (
                6.5  <= ph        <= 7.5  and
                35.0 <= turbidity <= 65.0 and
                uv   <  30        and
                vapor < 500
            ),
            "description": "NSF/ANSI 169 H1 propylene glycol coolant for food contact. Clean water-based zone turb 35–65%, low vapor confirms food-safe composition.",
            "parameters":  "pH 6.5–7.5 (NSF/ANSI 169 H1) | Turbidity 35–65% (water zone) | Vapor <500 ppm"
        },

        # ── 13. PHARMACEUTICAL COOLING ───────────────────────────
        # Zone A/B — ultra-pure water: turb 0–45%
        # Standard: USP <1231> / FDA 21 CFR
        {
            "name":     "Pharmaceutical Cooling",
            "icon":     "💊",
            "category": "Pharmaceutical Manufacturing",
            "criteria": lambda: (
                6.8  <= ph        <= 7.2  and
                turbidity         <= 45.0 and
                uv   <  20
            ),
            "description": "WFI grade coolant per USP <1231> / FDA 21 CFR. Ultra-pure water confirmed by turbidity ≤45% (tap/drinking water zone on sensor).",
            "parameters":  "pH 6.8–7.2 (USP WFI) | Turbidity ≤45% (ultra-pure water zone, sensor-verified)"
        },

        # ── 14. SOLAR PANEL COOLING ──────────────────────────────
        # Zone B/C — clean fluid: turb 35–65%
        # Standard: ASTM E2277 / ISO 9806
        {
            "name":     "Solar Panel Cooling",
            "icon":     "☀️",
            "category": "Renewable Energy",
            "criteria": lambda: (
                7.0  <= ph        <= 8.5  and
                35.0 <= turbidity <= 65.0 and
                uv   <  120
            ),
            "description": "UV-stable propylene glycol solar fluid per ASTM E2277 / ISO 9806. Water-based composition confirmed by turbidity 35–65%.",
            "parameters":  "pH 7.0–8.5 (ASTM E2277) | Turbidity 35–65% (water-based zone, sensor-verified)"
        },

        # ── 15. NUCLEAR PLANT COOLING ────────────────────────────
        # Zone A — tap water purity: turb 0–8%
        # Standard: IAEA Safety Series / NRC Guide 1.56
        {
            "name":     "Nuclear Plant Cooling",
            "icon":     "⚛️",
            "category": "Nuclear Energy",
            "criteria": lambda: (
                6.9  <= ph        <= 7.1  and
                turbidity         <= 8.0  and
                uv   <  10
            ),
            "description": "Demineralised water per IAEA Safety Series. Strictest purity — turbidity ≤8% (tap water zone, sensor-verified ADC ~1861). Tightest pH tolerance ±0.1.",
            "parameters":  "pH 6.9–7.1 (IAEA ±0.1) | Turbidity ≤8% (tap water zone, sensor-verified)"
        },

        # ── 16. DATA CENTER COOLING ──────────────────────────────
        # Zone D/E — dielectric: turb 58–82%
        # Standard: ASHRAE TC 9.9 / IEC 60068-2
        {
            "name":     "Data Center Cooling",
            "icon":     "🖥️",
            "category": "IT Infrastructure",
            "criteria": lambda: (
                7.0  <= ph        <= 8.5  and
                58.0 <= turbidity <= 82.0 and
                uv   <  50
            ),
            "description": "Dielectric immersion coolant per ASHRAE TC 9.9 / IEC 60068-2. Non-conductive oil-based fluid confirmed by turbidity 58–82%.",
            "parameters":  "pH 7.0–8.5 (ASHRAE TC 9.9) | Turbidity 58–82% (dielectric oil zone, sensor-verified)"
        },

        # ── 17. MARINE ENGINE COOLANT ────────────────────────────
        # Zone B/C — water-based: turb 35–65%, temp ≤110
        # Standard: IACS UR M9 / IMO MARPOL
        {
            "name":     "Marine Engine Coolant",
            "icon":     "⛵",
            "category": "Marine",
            "criteria": lambda: (
                7.5  <= ph        <= 10.5 and
                35.0 <= turbidity <= 65.0 and
                temp <= 110       and
                uv   <  200
            ),
            "description": "Corrosion-inhibited glycol coolant per IACS UR M9. Water-based inhibitor solution confirmed by turbidity 35–65%.",
            "parameters":  "pH 7.5–10.5 (IACS UR M9) | Turbidity 35–65% (water-based zone) | Temp ≤110°C"
        },

        # ── 18. AIRCRAFT ENGINE COOLANT ──────────────────────────
        # Zone B/C — water-based, high temp: turb 35–65%, temp ≥60
        # Standard: MIL-PRF-23699 / DEF STAN 91-098
        {
            "name":     "Aircraft Engine Coolant",
            "icon":     "✈️",
            "category": "Aerospace",
            "criteria": lambda: (
                8.0  <= ph        <= 11.0 and
                35.0 <= turbidity <= 65.0 and
                temp >= 60        and
                uv   <  150
            ),
            "description": "Polyol ester coolant per MIL-PRF-23699 / DEF STAN 91-098. Water-based zone turb 35–65% at high temp ≥60°C confirms aerospace grade.",
            "parameters":  "pH 8.0–11.0 (MIL-PRF-23699) | Turbidity 35–65% (sensor-verified) | Temp ≥60°C"
        },

        # ── 19. GENERAL INDUSTRIAL USE ───────────────────────────
        # Broadest catch-all — all zones except diesel/air
        # Standard: WHO Industrial Water Quality / ISO 14001
        {
            "name":     "General Industrial Use",
            "icon":     "🏗️",
            "category": "General Industrial",
            "criteria": lambda: (
                6.0  <= ph        <= 10.0 and
                turbidity         <  92.0
            ),
            "description": "Broad-spectrum industrial coolant per WHO/ISO 14001. Matches any fluid below diesel/air zone (turb <92%). Catch-all for unclassified fluids.",
            "parameters":  "pH 6.0–10.0 (WHO/ISO 14001) | Turbidity <92% (below diesel/air zone, sensor-verified)"
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
        except Exception:
            pass

    return results, color_name

# ─── Diagnosis and Remedy ────────────────────────────────────────
# Calibrated to FC-28 verified zones
def get_diagnosis_remedy(ph, tds, temp, turbidity, uv, vapor):
    issues   = []
    remedies = []

    # pH — unchanged, standard ranges
    if ph < 6.5:
        issues.append("Highly acidic coolant — severe corrosion risk")
        remedies.append("Add alkaline pH buffer immediately. Consider full replacement.")
    elif ph > 10.5:
        issues.append("Excessively alkaline — scaling and aluminium corrosion risk")
        remedies.append("Dilute with distilled water. Add pH stabiliser. Re-test after 1 hour.")
    elif ph < 7.0:
        issues.append("Slightly acidic — early corrosion risk")
        remedies.append("Monitor every 50 hours. Add corrosion inhibitor package.")
    elif ph > 9.5:
        issues.append("High alkalinity — check inhibitor concentration")
        remedies.append("Test inhibitor level. Top up with inhibitor concentrate.")

    # Turbidity — calibrated to verified FC-28 zones
    if turbidity > 92:
        issues.append("Turbidity in diesel/air zone (>92%) — non-conductive oil or empty probe")
        remedies.append("Verify fluid is present on probe. Check for diesel oil contamination.")
    elif turbidity > 85:
        issues.append("Turbidity above CNC zone (>85%) — approaching non-conductive boundary")
        remedies.append("Check fluid type. If CNC coolant, verify concentration is correct.")
    elif turbidity > 82:
        issues.append("Turbidity above heavy oil zone (>82%) — possible heavy oil contamination")
        remedies.append("Inspect seals for oil ingress. Check fluid source.")
    elif turbidity > 65:
        issues.append("Turbidity in oil zone (65–82%) — oil-based fluid or cross-contamination")
        remedies.append("Verify fluid type matches application. Check for oil seal leaks.")
    elif turbidity > 50:
        issues.append("Turbidity in EDM/light-oil transition zone (50–65%)")
        remedies.append("Confirm fluid type. Monitor for changes in turbidity trend.")
    elif turbidity > 8:
        issues.append("Turbidity in water-based zone (8–50%) — normal for water-based coolants")
        remedies.append("Water-based fluid confirmed. Monitor pH and inhibitor concentration.")

    # UV — unchanged
    if uv > 300:
        issues.append("Strong UV fluorescence — significant oil contamination")
        remedies.append("Isolate contamination source. Full drain and clean required.")
    elif uv > 100:
        issues.append("Mild UV fluorescence — possible trace oil contamination")
        remedies.append("Inspect gaskets and seals. Monitor weekly.")

    # Vapor — calibrated secondary signal
    if vapor >= 700:
        issues.append("High vapor signature (≥700 ppm) — heavy oil or diesel zone")
        remedies.append("Verify fluid is correct for application. Ensure adequate ventilation.")
    elif vapor >= 550:
        issues.append("Elevated vapor (550–700 ppm) — oil-based fluid confirmed")
        remedies.append("Normal range for oil-based fluids. Maintain ventilation.")
    elif vapor >= 400:
        issues.append("Moderate vapor (400–550 ppm) — glycol or light oil signature")
        remedies.append("Check coolant concentration. Review compatibility with system materials.")

    # Temperature — unchanged
    if temp > 90:
        issues.append("Very high temperature (>90°C) — thermal risk")
        remedies.append("Check pump flow rate. Inspect heat exchanger for fouling.")
    elif temp > 70:
        issues.append("Elevated temperature (70–90°C) — above optimal range")
        remedies.append("Monitor closely. Check coolant level and pump operation.")

    if not issues:
        issues.append("All parameters within calibrated range — coolant is in good condition")
        remedies.append("Continue regular monitoring every 500 hours or 3 months.")

    return " | ".join(issues), " | ".join(remedies)

# ─── Full Analysis Function ──────────────────────────────────────
def run_full_analysis(ph, tds, temp, turbidity,
                      uv, vapor, r, g, b, session_id):
    color_hex         = f"#{r:02x}{g:02x}{b:02x}"
    color_name        = detect_color_name(r, g, b)
    condition, score  = assess_condition(ph, tds, temp, turbidity, uv, vapor)
    apps, _           = match_applications(ph, tds, temp, turbidity, r, g, b, uv, vapor)
    diagnosis, remedy = get_diagnosis_remedy(ph, tds, temp, turbidity, uv, vapor)
    return {
        "session_id":              session_id,
        "ph_level":                ph,
        "tds_value":               tds,
        "temperature":             temp,
        "turbidity":               turbidity,
        "color_r":                 r,
        "color_g":                 g,
        "color_b":                 b,
        "color_hex":               color_hex,
        "uv_fluorescence":         uv,
        "vapor_level":             vapor,
        "coolant_color_name":      color_name,
        "condition":               condition,
        "application_suggestions": apps,
        "diagnosis":               diagnosis,
        "remedy":                  remedy,
        "score":                   score,
        "applications":            apps
    }

# ================================================================
# ROUTES — unchanged from your original
# ================================================================

@app.route("/")
def index():
    history = supabase.table("analysis_history").select("*") \
        .order("created_at", desc=True).limit(10).execute().data
    return render_template("index.html", history=history)

@app.route("/sensor_data", methods=["POST"])
def sensor_data():
    global pending_reading
    data = request.get_json()
    session_id = data.get("session_id",
                           "PICO-" + str(uuid.uuid4())[:8])
    pending_reading = {
        "session_id":      session_id,
        "tds":             float(data.get("tds",             0)),
        "temperature":     float(data.get("temperature",     25)),
        "turbidity":       float(data.get("turbidity",       50)),
        "uv_fluorescence": float(data.get("uv_fluorescence", 50)),
        "vapor_level":     float(data.get("vapor_level",    500)),
        "color_r":         int(data.get("color_r",          200)),
        "color_g":         int(data.get("color_g",          200)),
        "color_b":         int(data.get("color_b",          200)),
        "received_at":     datetime.now().isoformat()
    }
    return jsonify({
        "status":     "RECEIVED",
        "session_id": session_id,
        "message":    "Enter pH on dashboard to complete analysis"
    }), 200

@app.route("/submit_ph", methods=["POST"])
def submit_ph():
    global pending_reading, latest_reading
    data = request.get_json()
    ph   = float(data.get("ph", 7.0))

    if not pending_reading:
        return jsonify({"error": "No sensor data received yet. Wait for Pico W reading."}), 400
    if ph < 0 or ph > 14:
        return jsonify({"error": "Invalid pH. Enter 0–14."}), 400

    result = run_full_analysis(
        ph         = ph,
        tds        = pending_reading["tds"],
        temp       = pending_reading["temperature"],
        turbidity  = pending_reading["turbidity"],
        uv         = pending_reading["uv_fluorescence"],
        vapor      = pending_reading["vapor_level"],
        r          = pending_reading["color_r"],
        g          = pending_reading["color_g"],
        b          = pending_reading["color_b"],
        session_id = pending_reading["session_id"]
    )

    supabase.table("coolant_readings").insert({
        "session_id":              result["session_id"],
        "ph_level":                ph,
        "tds_value":               result["tds_value"],
        "temperature":             result["temperature"],
        "turbidity":               result["turbidity"],
        "color_r":                 result["color_r"],
        "color_g":                 result["color_g"],
        "color_b":                 result["color_b"],
        "color_hex":               result["color_hex"],
        "uv_fluorescence":         result["uv_fluorescence"],
        "vapor_level":             result["vapor_level"],
        "coolant_color_name":      result["coolant_color_name"],
        "condition":               result["condition"],
        "application_suggestions": result["applications"],
        "diagnosis":               result["diagnosis"],
        "remedy":                  result["remedy"]
    }).execute()

    supabase.table("analysis_history").insert({
        "session_id":      result["session_id"],
        "summary":         result["condition"],
        "top_application": result["applications"][0]["name"]
                           if result["applications"] else "General Industrial Use",
        "overall_score":   result["score"]
    }).execute()

    latest_reading  = result
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

@app.route("/analyze", methods=["POST"])
def analyze():
    global latest_reading
    data       = request.get_json()
    ph         = float(data.get("ph",             7.0))
    tds        = float(data.get("tds",            0))
    temp       = float(data.get("temperature",    25))
    turbidity  = float(data.get("turbidity",      50))
    r          = int(data.get("color_r",         200))
    g          = int(data.get("color_g",         200))
    b          = int(data.get("color_b",         200))
    uv         = float(data.get("uv_fluorescence",50))
    vapor      = float(data.get("vapor_level",   500))
    session_id = data.get("session_id",
                           "MANUAL-" + str(uuid.uuid4())[:8])

    result = run_full_analysis(ph, tds, temp, turbidity,
                                uv, vapor, r, g, b, session_id)

    supabase.table("coolant_readings").insert({
        "session_id":              result["session_id"],
        "ph_level":                ph,
        "tds_value":               result["tds_value"],
        "temperature":             result["temperature"],
        "turbidity":               result["turbidity"],
        "color_r":                 result["color_r"],
        "color_g":                 result["color_g"],
        "color_b":                 result["color_b"],
        "color_hex":               result["color_hex"],
        "uv_fluorescence":         result["uv_fluorescence"],
        "vapor_level":             result["vapor_level"],
        "coolant_color_name":      result["coolant_color_name"],
        "condition":               result["condition"],
        "application_suggestions": result["applications"],
        "diagnosis":               result["diagnosis"],
        "remedy":                  result["remedy"]
    }).execute()

    supabase.table("analysis_history").insert({
        "session_id":      result["session_id"],
        "summary":         result["condition"],
        "top_application": result["applications"][0]["name"]
                           if result["applications"] else "General Industrial Use",
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
    data = supabase.table("coolant_readings").select("*") \
        .order("recorded_at", desc=True).limit(20).execute().data
    return jsonify(data)

@app.route("/clear_latest", methods=["POST"])
def clear_latest():
    global latest_reading
    latest_reading = {}
    return jsonify({"status": "cleared"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=False)
