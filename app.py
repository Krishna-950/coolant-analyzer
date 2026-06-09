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
pending_reading = {}   # Stores sensor data waiting for pH from website

# ================================================================
# SENSOR CALIBRATION REFERENCE
# Prasanth's Pico W — June 2026 — All values sensor-verified
# ================================================================
# TDS SENSOR (GPIO 26):
#   All oils/non-conductive    : ADC 15–30,    TDS  0–2 ppm
#   CNC Coolant                : ADC 211–272,  TDS  63–82 ppm   (avg 73.5)
#   Drinking Water             : ADC 391–397,  TDS 118–120 ppm  (avg 119.5)
#   Tap Water                  : ADC 386–393,  TDS 117–119 ppm  (avg 118.0)
#   EDM Fluid                  : ADC 1093–1133,TDS 318–330 ppm  (avg 325)
#
# FC-28 TURBIDITY SENSOR (GPIO 27) — NEW SENSOR:
#   Formula: turb% = (ADC - 2100) / 1995 * 100
#   CNC Coolant    : ADC 2249–2556  turb  7–23%  (avg 14%)
#   Tap Water      : ADC 2806–2884  turb 35–39%  (avg 38%)
#   EDM Fluid      : ADC 2842–3079  turb 37–49%  (avg 43%)
#   Brake Fluid    : ADC 3513–3519  turb 71%
#   Gearbox Oil    : ADC 3520–3527  turb 71%
#   Diesel Oil     : ADC 4087       turb 99%
#   Air            : ADC 3900–4200  turb 90–105% (near 100%)
#
# TOLERANCE APPLIED: ±15% on all sensor thresholds
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
# Thresholds calibrated to actual sensor output ranges
def assess_condition(ph, tds, temp, turbidity, uv, vapor):
    score = 100

    # pH — manual entry, keep standard ranges
    if   7.0 <= ph <= 9.5:                           pass
    elif 6.5 <= ph < 7.0 or 9.5 < ph <= 10.5:       score -= 15
    else:                                             score -= 35

    # TDS — calibrated: sensor max ~330 ppm for water-based fluids
    # Oils always read 0–2 ppm (non-conductive)
    if   tds < 50:    pass           # oils + ultra-pure zone
    elif tds < 100:   score -= 5     # CNC coolant zone (63–82 ppm)
    elif tds < 200:   score -= 15    # drinking/tap water zone (~118 ppm)
    elif tds < 330:   score -= 25    # EDM/high mineral zone (~325 ppm)
    else:             score -= 40    # above sensor calibrated max

    # Turbidity — calibrated to new FC-28 sensor
    # Formula: turb% = (ADC - 2100) / 1995 * 100
    # CNC=7–23%, Water=35–39%, EDM=37–49%, Oils=71–99%
    if   turbidity < 25:  pass       # CNC coolant range — clean
    elif turbidity < 42:  score -= 10 # water-based fluids
    elif turbidity < 55:  score -= 20 # EDM / borderline
    elif turbidity < 75:  score -= 30 # oils (brake/gearbox)
    else:                 score -= 35  # diesel/air (non-conductive oils)

    # UV fluorescence — not sensor-calibrated yet, keep standard
    if   uv < 100:    pass
    elif uv < 300:    score -= 10
    else:             score -= 20

    # Vapor
    if   vapor < 200: pass
    elif vapor < 400: score -= 10
    else:             score -= 20

    score = max(0, score)
    if   score >= 80: return "Excellent", score
    elif score >= 60: return "Good", score
    elif score >= 40: return "Fair - Monitor Closely", score
    elif score >= 20: return "Poor - Replace Soon", score
    else:             return "Critical - Replace Immediately", score

# ─── Application Matching ────────────────────────────────────────
# ALL criteria calibrated to actual Pico W sensor output.
# Tolerance: TDS ±20 ppm, Turbidity ±5%, pH ±standard ISO range.
# Standard references preserved in description/parameters fields.
#
# KEY SENSOR ZONES (with ±15% tolerance):
#   TDS Zone 0 (oils):     0–5 ppm    (non-conductive)
#   TDS Zone 1 (CNC):      55–95 ppm  (measured 63–82 + tolerance)
#   TDS Zone 2 (water):    100–140 ppm(measured 117–120 + tolerance)
#   TDS Zone 3 (EDM):      295–355 ppm(measured 318–330 + tolerance)
#
#   Turb Zone 0 (CNC):     5–30%      (measured 7–23% + tolerance)
#   Turb Zone 1 (water):   30–45%     (measured 35–39% + tolerance)
#   Turb Zone 2 (EDM):     35–55%     (measured 37–49% + tolerance)
#   Turb Zone 3 (oils):    60–80%     (measured 71% + tolerance)
#   Turb Zone 4 (diesel):  85–100%    (measured 99% + tolerance)
# ================================================================
def match_applications(ph, tds, temp, turbidity, r, g, b, uv, vapor):
    color_name = detect_color_name(r, g, b)
    results    = []

    apps = [
        # ── 1. CNC MACHINING FLUID ───────────────────────────────
        # Standard: ASTM B860 / MSC Industrial
        # Sensor: TDS 55–95 ppm, Turbidity 5–30%
        {
            "name":     "CNC Machining Fluid",
            "icon":     "⚙️",
            "category": "Industrial Machining",
            "criteria": lambda ph=ph,tds=tds,temp=temp,turbidity=turbidity,vapor=vapor: (
                8.6 <= ph <= 9.5 and
                55  <= tds <= 95 and
                5   <= turbidity <= 30 and
                20  <= temp <= 55 and
                vapor < 300
            ),
            "description": "Precision CNC operations require stable alkaline semi-synthetic coolant to prevent bacterial growth, tool corrosion, and ensure consistent surface finish per ASTM B860.",
            "parameters":  "pH 8.6–9.5 (ASTM B860) | TDS 55–95 ppm (sensor) | Turbidity 5–30% (sensor) | Temp 20–55°C"
        },
        # ── 2. PETROL ENGINE COOLANT ─────────────────────────────
        # Standard: ASTM D3306 / BS 6580
        # Sensor: TDS 55–140 ppm (water-based + inhibitors), Turbidity 5–45%
        {
            "name":     "Petrol Engine Coolant",
            "icon":     "🚗",
            "category": "Automotive — Light Duty",
            "criteria": lambda ph=ph,tds=tds,turbidity=turbidity,uv=uv,vapor=vapor: (
                7.5 <= ph <= 11.0 and
                55  <= tds <= 140 and
                turbidity < 45 and
                uv   < 150 and
                vapor < 250
            ),
            "description": "Ethylene glycol based coolant for petrol passenger vehicles per ASTM D3306 / BS 6580. Provides freeze protection, boiling point elevation, and corrosion inhibition for aluminium and cast-iron engines.",
            "parameters":  "pH 7.5–11.0 (ASTM D3306) | TDS 55–140 ppm (sensor) | Turbidity <45% (sensor) | UV <150"
        },
        # ── 3. DIESEL ENGINE COOLANT ─────────────────────────────
        # Standard: ASTM D6210 / ASTM D3350
        # Sensor: TDS 55–140 ppm, Turbidity 5–45%
        {
            "name":     "Diesel Engine Coolant",
            "icon":     "🚛",
            "category": "Automotive — Heavy Duty",
            "criteria": lambda ph=ph,tds=tds,turbidity=turbidity,uv=uv,vapor=vapor: (
                8.0 <= ph <= 10.5 and
                55  <= tds <= 140 and
                turbidity < 45 and
                uv   < 200 and
                vapor < 300
            ),
            "description": "Fully formulated heavy duty engine coolant with nitrite/molybdate corrosion inhibitors per ASTM D6210. Designed for diesel engines in trucks, buses, and construction equipment.",
            "parameters":  "pH 8.0–10.5 (ASTM D6210) | TDS 55–140 ppm (sensor) | Turbidity <45% (sensor)"
        },
        # ── 4. HYDRAULIC SYSTEM FLUID ────────────────────────────
        # Standard: ISO 11158 / DIN 51524 Part 3
        # Sensor: TDS 0–5 ppm (oil-based), Turbidity 60–80%
        {
            "name":     "Hydraulic System Fluid",
            "icon":     "🔧",
            "category": "Industrial Hydraulics",
            "criteria": lambda ph=ph,tds=tds,turbidity=turbidity,uv=uv,vapor=vapor: (
                6.5 <= ph <= 8.5 and
                tds <= 5 and
                60  <= turbidity <= 80 and
                uv   < 80 and
                vapor < 200
            ),
            "description": "Ultra-clean mineral oil based hydraulic fluid per ISO 11158 / DIN 51524-3. Strict cleanliness class ISO 4406 required to protect precision servo valves and hydraulic pumps.",
            "parameters":  "pH 6.5–8.5 (ISO 11158) | TDS ≤5 ppm (oil zone, sensor) | Turbidity 60–80% (oil zone, sensor)"
        },
        # ── 5. BRAKE SYSTEM FLUID ────────────────────────────────
        # Standard: SAE J1703 / FMVSS 116 DOT 3/4/5.1
        # Sensor: TDS 0–5 ppm, Turbidity 65–80% (measured avg 71%)
        {
            "name":     "Brake System Fluid",
            "icon":     "🛑",
            "category": "Automotive Safety",
            "criteria": lambda ph=ph,tds=tds,turbidity=turbidity,uv=uv,vapor=vapor: (
                7.0 <= ph <= 11.5 and
                tds <= 5 and
                65  <= turbidity <= 80 and
                uv   < 50 and
                vapor < 150
            ),
            "description": "DOT-grade glycol ether brake fluid per SAE J1703 / FMVSS 116. Safety-critical fluid — any moisture ingress increases compressibility. Sensor reads ~71% turbidity due to glycol-ether composition.",
            "parameters":  "pH 7.0–11.5 (SAE J1703) | TDS ≤5 ppm (sensor) | Turbidity 65–80% (glycol-ether zone, sensor)"
        },
        # ── 6. AIR COMPRESSOR COOLANT ────────────────────────────
        # Standard: ISO 6743-3A / DIN 51506
        # Sensor: TDS 0–95 ppm (wide range), Turbidity 5–45%
        {
            "name":     "Air Compressor Coolant",
            "icon":     "💨",
            "category": "Industrial Pneumatics",
            "criteria": lambda ph=ph,tds=tds,turbidity=turbidity,uv=uv,vapor=vapor: (
                6.5 <= ph <= 8.0 and
                tds <= 95 and
                turbidity < 45 and
                uv   < 100 and
                vapor < 300
            ),
            "description": "Mineral oil or synthetic coolant/lubricant for rotary screw and reciprocating air compressors per ISO 6743-3A / DIN 51506. Controls heat, reduces wear, and seals compression stages.",
            "parameters":  "pH 6.5–8.0 (ISO 6743-3A) | TDS ≤95 ppm (sensor) | Turbidity <45% (sensor)"
        },
        # ── 7. GEARBOX LUBRICANT ─────────────────────────────────
        # Standard: ISO 3448 / DIN 51517 / AGMA 9005
        # Sensor: TDS 0–5 ppm (oil-based), Turbidity 65–80% (measured avg 71%)
        {
            "name":     "Gearbox Lubricant",
            "icon":     "⚙️",
            "category": "Industrial Transmission",
            "criteria": lambda ph=ph,tds=tds,temp=temp,turbidity=turbidity: (
                5.5 <= ph <= 8.5 and
                tds <= 5 and
                65  <= turbidity <= 80 and
                temp <= 120
            ),
            "description": "High extreme-pressure gear lubricant per ISO 3448 / DIN 51517 / AGMA 9005. Contains sulfur-phosphorus EP additives for hypoid and helical gears. Sensor reads ~71% turbidity (oil zone).",
            "parameters":  "pH 5.5–8.5 (ISO 3448) | TDS ≤5 ppm (oil zone, sensor) | Turbidity 65–80% (sensor) | Temp ≤120°C"
        },
        # ── 8. EDM MACHINE FLUID ─────────────────────────────────
        # Standard: ISO 4370 / JIS B 4200
        # Sensor: TDS 295–355 ppm (measured 318–330 + tolerance), Turbidity 35–55%
        {
            "name":     "EDM Machine Fluid",
            "icon":     "🔬",
            "category": "Precision Manufacturing",
            "criteria": lambda ph=ph,tds=tds,turbidity=turbidity,uv=uv,vapor=vapor: (
                6.5 <= ph <= 7.5 and
                295 <= tds <= 355 and
                35  <= turbidity <= 55 and
                uv   < 30 and
                vapor < 100
            ),
            "description": "Deionised dielectric fluid for wire/sinker EDM machining per ISO 4370 / JIS B 4200. Resistivity must be maintained for consistent spark erosion. Your sensor measures ~325 ppm and ~43% turbidity for this fluid.",
            "parameters":  "pH 6.5–7.5 (ISO 4370) | TDS 295–355 ppm (sensor-verified) | Turbidity 35–55% (sensor-verified)"
        },
        # ── 9. HEAT EXCHANGER FLUID ──────────────────────────────
        # Standard: TEMA / ASHRAE Handbook
        # Sensor: TDS 55–140 ppm, Turbidity 5–45%, Temp >=50°C
        {
            "name":     "Heat Exchanger Fluid",
            "icon":     "🌡️",
            "category": "Industrial Thermal",
            "criteria": lambda ph=ph,tds=tds,temp=temp,turbidity=turbidity,uv=uv: (
                7.5 <= ph <= 9.5 and
                55  <= tds <= 140 and
                turbidity < 45 and
                temp >= 50 and
                uv   < 200
            ),
            "description": "Inhibited ethylene or propylene glycol fluid for shell-and-tube and plate heat exchangers per TEMA standards. Scale and corrosion inhibitors maintain heat transfer efficiency.",
            "parameters":  "pH 7.5–9.5 (TEMA) | TDS 55–140 ppm (sensor) | Turbidity <45% (sensor) | Temp ≥50°C"
        },
        # ── 10. METAL FORMING FLUID ──────────────────────────────
        # Standard: ISO 6743-7 / DIN 51385
        # Sensor: TDS 55–140 ppm, Turbidity 30–55% (milky emulsion)
        {
            "name":     "Metal Forming Fluid",
            "icon":     "🏭",
            "category": "Metal Processing",
            "criteria": lambda ph=ph,tds=tds,turbidity=turbidity,vapor=vapor: (
                7.0 <= ph <= 9.5 and
                55  <= tds <= 140 and
                30  <= turbidity <= 55 and
                vapor < 400
            ),
            "description": "Semisynthetic or soluble oil emulsion for stamping, deep drawing, and roll forming per ISO 6743-7 / DIN 51385. Milky white appearance (30–55% turbidity on sensor) is normal and expected.",
            "parameters":  "pH 7.0–9.5 (ISO 6743-7) | TDS 55–140 ppm (sensor) | Turbidity 30–55% (milky emulsion zone, sensor)"
        },
        # ── 11. EV BATTERY COOLING ───────────────────────────────
        # Standard: SAE J2800 / ASTM D1816
        # Sensor: TDS 0–5 ppm (dielectric purity), Turbidity 60–80%
        {
            "name":     "EV Battery Cooling",
            "icon":     "🔋",
            "category": "Electric Vehicles",
            "criteria": lambda ph=ph,tds=tds,turbidity=turbidity,uv=uv,vapor=vapor: (
                6.8 <= ph <= 7.5 and
                tds <= 5 and
                60  <= turbidity <= 80 and
                uv   < 50 and
                vapor < 150
            ),
            "description": "Non-conductive dielectric immersion coolant for EV battery pack thermal management per SAE J2800 / ASTM D1816. Zero conductivity mandatory to prevent cell short-circuit.",
            "parameters":  "pH 6.8–7.5 (SAE J2800) | TDS ≤5 ppm (dielectric, sensor) | Turbidity 60–80% (sensor)"
        },
        # ── 12. FOOD GRADE COOLING ───────────────────────────────
        # Standard: NSF/ANSI 169 / H1 / 3H
        # Sensor: TDS 0–95 ppm, Turbidity 5–30% (clean fluid)
        {
            "name":     "Food Grade Cooling",
            "icon":     "🍃",
            "category": "Food & Beverage Processing",
            "criteria": lambda ph=ph,tds=tds,turbidity=turbidity,uv=uv,vapor=vapor: (
                6.5 <= ph <= 7.5 and
                tds <= 95 and
                turbidity < 30 and
                uv   < 30 and
                vapor < 100
            ),
            "description": "NSF/ANSI 169 H1 certified propylene glycol coolant for incidental food-contact applications. Zero toxicity mandatory. Used in beverage chillers, dairy processing, and meat refrigeration.",
            "parameters":  "pH 6.5–7.5 (NSF/ANSI 169 H1) | TDS ≤95 ppm (sensor) | Turbidity <30% (sensor)"
        },
        # ── 13. PHARMACEUTICAL COOLING ───────────────────────────
        # Standard: USP <1231> / FDA 21 CFR 165.110 / EP 3.2.9
        # Sensor: TDS 0–5 ppm (WFI purity), Turbidity 5–25%
        {
            "name":     "Pharmaceutical Cooling",
            "icon":     "💊",
            "category": "Pharmaceutical Manufacturing",
            "criteria": lambda ph=ph,tds=tds,turbidity=turbidity,uv=uv,vapor=vapor: (
                6.8 <= ph <= 7.2 and
                tds <= 5 and
                turbidity < 25 and
                uv   < 20 and
                vapor < 80
            ),
            "description": "Water for Injection (WFI) grade coolant per USP <1231> / FDA 21 CFR 165.110. Used in bioreactor cooling, autoclave jacketing, and cleanroom HVAC. Strictest purity requirement of all applications.",
            "parameters":  "pH 6.8–7.2 (USP WFI) | TDS ≤5 ppm (WFI zone, sensor) | Turbidity <25% (sensor)"
        },
        # ── 14. SOLAR PANEL COOLING ──────────────────────────────
        # Standard: ASTM E2277 / ISO 9806
        # Sensor: TDS 0–140 ppm, Turbidity 5–45%
        {
            "name":     "Solar Panel Cooling",
            "icon":     "☀️",
            "category": "Renewable Energy",
            "criteria": lambda ph=ph,tds=tds,turbidity=turbidity,uv=uv,vapor=vapor: (
                7.0 <= ph <= 8.5 and
                tds <= 140 and
                turbidity < 45 and
                uv   < 120 and
                vapor < 200
            ),
            "description": "UV-stable propylene glycol solar thermal fluid per ASTM E2277 / ISO 9806. Designed for concentrated solar power (CSP) and flat-plate collector systems with operating temperatures up to 200°C.",
            "parameters":  "pH 7.0–8.5 (ASTM E2277) | TDS ≤140 ppm (sensor) | Turbidity <45% (sensor)"
        },
        # ── 15. NUCLEAR PLANT COOLING ────────────────────────────
        # Standard: IAEA Safety Series / NRC Regulatory Guide 1.56
        # Sensor: TDS 0–5 ppm (demineralised), Turbidity 5–25%
        {
            "name":     "Nuclear Plant Cooling",
            "icon":     "⚛️",
            "category": "Nuclear Energy",
            "criteria": lambda ph=ph,tds=tds,turbidity=turbidity,uv=uv,vapor=vapor: (
                6.9 <= ph <= 7.1 and
                tds <= 5 and
                turbidity < 25 and
                uv   < 10 and
                vapor < 50
            ),
            "description": "High-purity demineralised water for secondary cooling circuits per IAEA Safety Series No. 50-SG-D5. Chloride and sulphate levels must be near zero to prevent stress corrosion cracking in stainless steel.",
            "parameters":  "pH 6.9–7.1 (IAEA ±0.1 tolerance) | TDS ≤5 ppm (demineralised zone, sensor) | Turbidity <25% (sensor)"
        },
        # ── 16. DATA CENTER COOLING ──────────────────────────────
        # Standard: ASHRAE TC 9.9 / IEC 60068-2 / ETSI EN 300 019
        # Sensor: TDS 0–5 ppm (dielectric), Turbidity 60–80%
        {
            "name":     "Data Center Cooling",
            "icon":     "🖥️",
            "category": "IT Infrastructure",
            "criteria": lambda ph=ph,tds=tds,turbidity=turbidity,uv=uv,vapor=vapor: (
                7.0 <= ph <= 8.5 and
                tds <= 5 and
                60  <= turbidity <= 80 and
                uv   < 50 and
                vapor < 100
            ),
            "description": "Non-conductive dielectric fluid for single-phase or two-phase server immersion cooling per ASHRAE TC 9.9 / IEC 60068-2. Must not degrade PCB materials or component coatings.",
            "parameters":  "pH 7.0–8.5 (ASHRAE TC 9.9) | TDS ≤5 ppm (dielectric zone, sensor) | Turbidity 60–80% (sensor)"
        },
        # ── 17. MARINE ENGINE COOLANT ────────────────────────────
        # Standard: IACS UR M9 / IMO MARPOL / MAN B&W TBO
        # Sensor: TDS 55–140 ppm, Turbidity 5–45%
        {
            "name":     "Marine Engine Coolant",
            "icon":     "⛵",
            "category": "Marine",
            "criteria": lambda ph=ph,tds=tds,temp=temp,turbidity=turbidity,uv=uv: (
                7.5 <= ph <= 10.5 and
                55  <= tds <= 140 and
                turbidity < 45 and
                uv   < 200 and
                temp <= 110
            ),
            "description": "Corrosion-inhibited glycol coolant for marine propulsion and auxiliary engines per IACS UR M9 / IMO MARPOL Annex I. Nitrite-borate or organic acid inhibitor packages required for cast iron liners.",
            "parameters":  "pH 7.5–10.5 (IACS UR M9) | TDS 55–140 ppm (sensor) | Turbidity <45% (sensor) | Temp ≤110°C"
        },
        # ── 18. AIRCRAFT ENGINE COOLANT ──────────────────────────
        # Standard: MIL-PRF-23699 / DEF STAN 91-098 / ASTM D6130
        # Sensor: TDS 55–140 ppm, Turbidity 5–45%, Temp >=60°C
        {
            "name":     "Aircraft Engine Coolant",
            "icon":     "✈️",
            "category": "Aerospace",
            "criteria": lambda ph=ph,tds=tds,temp=temp,turbidity=turbidity,uv=uv: (
                8.0 <= ph <= 11.0 and
                55  <= tds <= 140 and
                turbidity < 45 and
                temp >= 60 and
                uv   < 150
            ),
            "description": "High-performance polyol ester coolant for gas turbine engines per MIL-PRF-23699 / DEF STAN 91-098. Extremely high flash point (>260°C) and thermal stability up to 200°C mandatory.",
            "parameters":  "pH 8.0–11.0 (MIL-PRF-23699) | TDS 55–140 ppm (sensor) | Turbidity <45% (sensor) | Temp ≥60°C"
        },
        # ── 19. GENERAL INDUSTRIAL USE ───────────────────────────
        # Standard: WHO Industrial Water Quality / ISO 14001
        # Broadest catch-all — covers full sensor range
        {
            "name":     "General Industrial Use",
            "icon":     "🏗️",
            "category": "General Industrial",
            "criteria": lambda ph=ph,tds=tds,turbidity=turbidity: (
                6.0 <= ph <= 10.0 and
                tds <= 355 and
                turbidity < 85
            ),
            "description": "Broad-spectrum industrial coolant meeting WHO Industrial Water Quality guidelines / ISO 14001 environmental management standards. Suitable when precise application classification is not required.",
            "parameters":  "pH 6.0–10.0 (WHO/ISO 14001) | TDS ≤355 ppm (full sensor range) | Turbidity <85% (sensor)"
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
# All thresholds calibrated to actual sensor output
def get_diagnosis_remedy(ph, tds, temp, turbidity, uv, vapor):
    issues   = []
    remedies = []

    # pH — manual entry, standard thresholds
    if ph < 6.5:
        issues.append("Highly acidic coolant — severe corrosion risk to metal components")
        remedies.append("Add alkaline pH buffer immediately. Consider full coolant replacement.")
    elif ph > 10.5:
        issues.append("Excessively alkaline — risk of scaling, deposits, and aluminium corrosion")
        remedies.append("Dilute with distilled water. Add pH stabiliser. Re-test after 1 hour.")
    elif ph < 7.0:
        issues.append("Slightly acidic coolant — early stage corrosion risk")
        remedies.append("Monitor closely every 50 hours. Add corrosion inhibitor package.")
    elif ph > 9.5:
        issues.append("High alkalinity — monitor for inhibitor depletion")
        remedies.append("Test inhibitor concentration. Top up with inhibitor concentrate.")

    # TDS — calibrated to sensor zones
    if tds > 330:
        issues.append("TDS above calibrated range — severely contaminated or wrong fluid type")
        remedies.append("Immediate coolant replacement. Flush system with clean water. Check fill source.")
    elif tds > 140:
        issues.append("Elevated TDS — moderate contamination or mineral buildup detected")
        remedies.append("Partial coolant replacement (30%). Top up with demineralised water.")
    elif tds > 95:
        issues.append("TDS above CNC coolant range — possible water ingress or dilution")
        remedies.append("Check coolant concentration. Verify makeup water quality. Monitor trend.")

    # Turbidity — calibrated to new FC-28 sensor zones
    if turbidity > 80:
        issues.append("Very high turbidity — diesel oil or non-conductive fluid detected (sensor zone: 85–100%)")
        remedies.append("Verify fluid type is correct for your application. Check for oil contamination.")
    elif turbidity > 55:
        issues.append("High turbidity — oil-based fluid detected (brake/gearbox zone: 65–80%)")
        remedies.append("Check if oil-based fluid is correct for application. Inspect seals for cross-contamination.")
    elif turbidity > 42:
        issues.append("Moderate turbidity — EDM or water-based zone detected (35–55%)")
        remedies.append("Verify fluid matches intended application. Check filtration system.")
    elif turbidity > 25:
        issues.append("Slight turbidity elevation — water-based fluid range (30–42%)")
        remedies.append("Monitor trend. Check inline filter condition. Verify coolant concentration.")

    # UV fluorescence — standard thresholds
    if uv > 300:
        issues.append("Strong UV fluorescence — significant oil contamination detected")
        remedies.append("Isolate contamination source (seal failure or cross-fill). Full drain and clean required.")
    elif uv > 100:
        issues.append("Mild UV fluorescence — possible trace oil or lubricant contamination")
        remedies.append("Inspect gaskets, seals, and O-rings. Monitor fluorescence trend weekly.")

    # Vapor
    if vapor > 400:
        issues.append("High chemical vapor signature — volatile compounds exceeding safe limit")
        remedies.append("Ensure adequate ventilation in work area. Review coolant compatibility. Check temperature.")
    elif vapor > 200:
        issues.append("Elevated vapor signature — mild volatile compound presence")
        remedies.append("Check coolant temperature and concentration. Ensure proper ventilation.")

    # Temperature
    if temp > 90:
        issues.append("Very high coolant temperature — thermal runaway risk")
        remedies.append("Check coolant pump flow rate. Inspect heat exchanger fins for fouling. Reduce machine load.")
    elif temp > 70:
        issues.append("Elevated coolant temperature — above optimal operating range")
        remedies.append("Monitor closely. Check coolant level and pump operation.")

    if not issues:
        issues.append("All parameters within acceptable calibrated range — coolant is in good condition")
        remedies.append("Continue regular monitoring every 500 operating hours or 3 months. Next scheduled check: record date.")

    return " | ".join(issues), " | ".join(remedies)

# ─── Full Analysis Function ──────────────────────────────────────
def run_full_analysis(ph, tds, temp, turbidity,
                      uv, vapor, r, g, b, session_id):
    color_hex    = f"#{r:02x}{g:02x}{b:02x}"
    color_name   = detect_color_name(r, g, b)
    condition, score = assess_condition(ph, tds, temp, turbidity, uv, vapor)
    apps, _      = match_applications(ph, tds, temp, turbidity, r, g, b, uv, vapor)
    diagnosis, remedy = get_diagnosis_remedy(ph, tds, temp, turbidity, uv, vapor)
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
# ROUTES — unchanged from your original file
# ================================================================

@app.route("/")
def index():
    history = supabase.table("analysis_history").select("*") \
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
        "session_id":      session_id,
        "tds":             float(data.get("tds",             500)),
        "temperature":     float(data.get("temperature",      25)),
        "turbidity":       float(data.get("turbidity",        20)),
        "uv_fluorescence": float(data.get("uv_fluorescence",  50)),
        "vapor_level":     float(data.get("vapor_level",     100)),
        "color_r":         int(data.get("color_r",           200)),
        "color_g":         int(data.get("color_g",           200)),
        "color_b":         int(data.get("color_b",           200)),
        "received_at":     datetime.now().isoformat()
    }

    return jsonify({
        "status":     "RECEIVED",
        "session_id": session_id,
        "message":    "Enter pH on dashboard to complete analysis"
    }), 200

# ─── Website submits pH → triggers full analysis ─────────────────
@app.route("/submit_ph", methods=["POST"])
def submit_ph():
    global pending_reading, latest_reading
    data = request.get_json()
    ph   = float(data.get("ph", 7.0))

    if not pending_reading:
        return jsonify({"error": "No sensor data received yet. Wait for Pico W reading."}), 400

    if ph < 0 or ph > 14:
        return jsonify({"error": "Invalid pH value. Enter between 0 and 14."}), 400

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

# ─── Manual test panel — kept exactly as original ───────────────
@app.route("/analyze", methods=["POST"])
def analyze():
    global latest_reading
    data = request.get_json()

    ph         = float(data.get("ph",             7.0))
    tds        = float(data.get("tds",            500))
    temp       = float(data.get("temperature",     25))
    turbidity  = float(data.get("turbidity",       20))
    r          = int(data.get("color_r",          200))
    g          = int(data.get("color_g",          200))
    b          = int(data.get("color_b",          200))
    uv         = float(data.get("uv_fluorescence", 50))
    vapor      = float(data.get("vapor_level",    100))
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
