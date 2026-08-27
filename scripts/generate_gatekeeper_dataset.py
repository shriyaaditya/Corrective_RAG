import json
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
raw_file = root_dir / "data" / "raw_retrieved_chunks.json"

with open(raw_file, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

# Dataset construction rules:
# Target distribution:
# ~16 CORRECT
# ~14 AMBIGUOUS
# ~12 INCORRECT
# Total: 42 entries

dataset = []

# Helper to append entry
def add_entry(query, chunk, expected_label, source="retriever", notes="", confidence="high"):
    dataset.append({
        "query": query,
        "chunk": chunk,
        "expected_label": expected_label,
        "source": source,
        "confidence": confidence,
        "notes": notes
    })

# -------------------------------------------------------------
# 1. QUERY: What is the maximum body temperature for a BGA-256 package during SAC305 reflow?
# -------------------------------------------------------------
q1 = "What is the maximum body temperature for a BGA-256 package during SAC305 reflow?"
c1_0 = raw_data[q1][0] # Peak package temp limit 245 °C BGA
add_entry(q1, c1_0, "CORRECT", notes="Directly specifies BGA Peak Package Temperature Limit of 245 °C.")

c1_1 = raw_data[q1][1] # SAC305 general profile
add_entry(q1, c1_1, "AMBIGUOUS", notes="Contains overall SAC305 package temperature ceilings (260 °C max), but not package-specific BGA delta ceiling.")

c1_2 = raw_data[q1][2] # MSL-3 handling
add_entry(q1, c1_2, "INCORRECT", notes="Near-miss distractor: discusses MSL-3 bake rules and floor life, completely irrelevant to peak reflow temperature.")

c1_3 = raw_data[q1][3] # STM32F407 datasheet
add_entry(q1, c1_3, "INCORRECT", notes="Off-topic: operating voltage and junction temperature specs for STM32 MCU.")

# -------------------------------------------------------------
# 2. QUERY: What is the recommended gain setting for the DRV8301 current shunt amplifier to achieve a 40 V/V output?
# -------------------------------------------------------------
q2 = "What is the recommended gain setting for the DRV8301 current shunt amplifier to achieve a 40 V/V output?"
c2_0 = raw_data[q2][0] # Table 12 Register Address 0x03
add_entry(q2, c2_0, "CORRECT", notes="Explicitly states Bits [D3:D2] = 10 -> Gain of shunt amplifier: 40 V/V in Register 0x03.")

c2_1 = raw_data[q2][1] # Section 4 Pin Functions
add_entry(q2, c2_1, "AMBIGUOUS", notes="Lists available gain options (10 V/V, 20 V/V, 40 V/V, 80 V/V), but does not state the SPI register control bits needed to program 40 V/V.")

c2_2 = raw_data[q2][2] # Section 1 Features
add_entry(q2, c2_2, "AMBIGUOUS", notes="Mentions dual integrated current shunt amplifiers with adjustable gain, but lacks register address and bit settings.")

c2_3 = raw_data[q2][3] # Section 3 Description
add_entry(q2, c2_3, "INCORRECT", notes="Near-miss distractor: discusses DRV8301 gate drive current capability (1.7A/2.3A) and voltage range (6V-60V), but not current shunt amplifier gain.")

# -------------------------------------------------------------
# 3. QUERY: What is the global stock and unit price for part STM32F407VGT6?
# -------------------------------------------------------------
q3 = "What is the global stock and unit price for part STM32F407VGT6?"
c3_0 = raw_data[q3][0] # internal_avl.csv
add_entry(q3, c3_0, "AMBIGUOUS", notes="Confirms IPN-MCU-002 maps to STM32F407VGT6 in internal AVL, but contains zero live market stock or pricing data (requires Mouser API).")

c3_1 = raw_data[q3][1] # STM32 datasheet architecture
add_entry(q3, c3_1, "AMBIGUOUS", notes="Relabeled per Policy A — market-data queries always route to AMBIGUOUS/Mouser fallback regardless of chunk content.")

c3_2 = raw_data[q3][2] # project_bom.csv
add_entry(q3, c3_2, "AMBIGUOUS", notes="Relabeled per Policy A — market-data queries always route to AMBIGUOUS/Mouser fallback regardless of chunk content.")

# -------------------------------------------------------------
# 4. QUERY: What is the lifecycle status and lead time for internal part number IPN-DRV-001?
# -------------------------------------------------------------
q4 = "What is the lifecycle status and lead time for internal part number IPN-DRV-001?"
c4_0 = raw_data[q4][0] # internal_avl.csv
add_entry(q4, c4_0, "AMBIGUOUS", notes="Maps IPN-DRV-001 to DRV8301DCAR with internal status 'Approved', but does not contain Mouser lifecycle status or factory lead time.")

c4_1 = raw_data[q4][1] # project_bom.csv
add_entry(q4, c4_1, "AMBIGUOUS", notes="Shows DRV8301DCAR on internal BOM, but lacks external lead time and lifecycle data.")

c4_2 = raw_data[q4][2] # DRV8301 specs
add_entry(q4, c4_2, "INCORRECT", notes="Near-miss distractor: details PVDD operating voltage and dead time resistor, no lifecycle status.")

# -------------------------------------------------------------
# 5. QUERY: What is the specific solder reflow temperature profile for a BGA-256 package in a quantum computing array?
# -------------------------------------------------------------
q5 = "What is the specific solder reflow temperature profile for a BGA-256 package in a quantum computing array?"
c5_0 = raw_data[q5][0] # BGA package ceilings
add_entry(q5, c5_0, "AMBIGUOUS", notes="Gives general SMT BGA reflow temperature ceiling (245 °C), but quantum-computing array specific profile parameters do not exist in doc.")

c5_1 = raw_data[q5][1] # SAC305 general profile
add_entry(q5, c5_1, "AMBIGUOUS", notes="Gives general SAC305 profile parameters, but lacks quantum array application specifications.")

c5_2 = raw_data[q5][2] # DRV8301 electrical specs
add_entry(q5, c5_2, "INCORRECT", notes="Completely off-topic DRV8301 gate driver specs.")

# -------------------------------------------------------------
# 6. QUERY: What is the pinout configuration and thermal pad ground requirement for DRV8301?
# -------------------------------------------------------------
q6 = "What is the pinout configuration and thermal pad ground requirement for DRV8301?"
c6_0 = raw_data[q6][0] # DRV8301 Description & package info
add_entry(q6, c6_0, "AMBIGUOUS", notes="Mentions HTSSOP (56) package, but lacks pinout diagram and thermal pad grounding rules.", confidence="low")

c6_1 = raw_data[q6][1] # smt_reflow_guidelines
add_entry(q6, c6_1, "INCORRECT", notes="SMT reflow soldering guidelines, no pinout or thermal ground info.")

c6_2 = raw_data[q6][2] # DRV8301 Features
add_entry(q6, c6_2, "AMBIGUOUS", notes="Lists features including thermal shutdown and nFAULT pins, but missing complete pinout layout.")

# -------------------------------------------------------------
# 7. QUERY: What is the maximum allowed warp and twist percentage for FR-4 PCBs under IPC-A-600?
# -------------------------------------------------------------
q7 = "What is the maximum allowed warp and twist percentage for FR-4 PCBs under IPC-A-600?"
c7_0 = raw_data[q7][0] # smt_reflow_guidelines SAC305
add_entry(q7, c7_0, "INCORRECT", notes="Near-miss distractor: SMT reflow profile parameters, completely lacks IPC-A-600 PCB warp/twist rules.")

c7_1 = raw_data[q7][1] # MSL-3 guidelines
add_entry(q7, c7_1, "INCORRECT", notes="Moisture sensitivity level baking rules, no PCB fabrication warp specs.")

# -------------------------------------------------------------
# 8. QUERY: How does temperature coefficient affect resistor tolerances in high precision motor drive circuits?
# -------------------------------------------------------------
q8 = "How does temperature coefficient affect resistor tolerances in high precision motor drive circuits?"
c8_0 = raw_data[q8][0] # project_bom.csv
add_entry(q8, c8_0, "AMBIGUOUS", notes="Lists Yageo 10K 1% 0805 resistors (RC0805FR-0710KL) on BOM, but does not provide temp coefficient (TCR) performance analysis.")

c8_1 = raw_data[q8][1] # internal_avl.csv
add_entry(q8, c8_1, "AMBIGUOUS", notes="Lists internal part IPN-RES-006, but lacks temperature drift analysis.")

# -------------------------------------------------------------
# 9. QUERY: What are the Mouser stocking status and minimum order quantity (MOQ) for Texas Instruments DRV8301DCAR?
# -------------------------------------------------------------
q9 = "What are the Mouser stocking status and minimum order quantity (MOQ) for Texas Instruments DRV8301DCAR?"
c9_0 = raw_data[q9][0] # internal_avl.csv
add_entry(q9, c9_0, "AMBIGUOUS", notes="Contains internal AVL record for DRV8301DCAR (IPN-DRV-001), but lacks external distributor stock & MOQ.")

c9_1 = raw_data[q9][1] # project_bom.csv
add_entry(q9, c9_1, "AMBIGUOUS", notes="Contains internal BOM record for DRV8301DCAR, but lacks Mouser inventory status.")

c9_2 = raw_data[q9][2] # Register address map
add_entry(q9, c9_2, "AMBIGUOUS", notes="Relabeled per Policy A — market-data queries always route to AMBIGUOUS/Mouser fallback regardless of chunk content.")

# -------------------------------------------------------------
# 10. QUERY: What internal part number maps to Texas Instruments TPS62130RGTR in the AVL?
# -------------------------------------------------------------
q10 = "What internal part number maps to Texas Instruments TPS62130RGTR in the AVL?"
c10_0 = raw_data[q10][0] # internal_avl.csv
add_entry(q10, c10_0, "INCORRECT", notes="Near-miss distractor: internal_avl.csv lists IPN mappings for DRV8301, STM32F407, INA240, etc., but TPS62130RGTR is NOT in the AVL.")

c10_1 = raw_data[q10][1] # project_bom.csv
add_entry(q10, c10_1, "INCORRECT", notes="Project BOM listing, does not contain TPS62130RGTR.")

# -------------------------------------------------------------
# 11. QUERY: What is the minimum trace width and clearance for 1 oz copper in high-current motor driver PCBs?
# -------------------------------------------------------------
q11 = "What is the minimum trace width and clearance for 1 oz copper in high-current motor driver PCBs?"
c11_0 = raw_data[q11][0] # project_bom.csv
add_entry(q11, c11_0, "INCORRECT", notes="Project BOM table, contains no PCB layout trace width guidelines.")

c11_1 = raw_data[q11][2] # DRV8301 description
add_entry(q11, c11_1, "INCORRECT", notes="Near-miss distractor: describes DRV8301 motor driver IC output currents, but lacks PCB trace width design rules.")

# -------------------------------------------------------------
# 12. QUERY: What is the maximum reflow peak temperature for QFN-48 packages under J-STD-020 standard?
# -------------------------------------------------------------
q12 = "What is the maximum reflow peak temperature for QFN-48 packages under J-STD-020 standard?"
c12_0 = raw_data[q12][0] # Package specific ceilings
add_entry(q12, c12_0, "INCORRECT", notes="Lists BGA temperature ceilings (245 °C) under Section 3, but QFN-48 specific peak ceiling is completely missing from this chunk.", confidence="high")

c12_1 = raw_data[q12][1] # SAC305 reflow profile
add_entry(q12, c12_1, "CORRECT", notes="States Maximum Allowable Package Temperature of 260 °C per J-STD-020 for lead-free SMT packages.")

# -------------------------------------------------------------
# 13. QUERY: What is the lead time for Xilinx Artix-7 FPGA XC7A35T-1FTG256C?
# -------------------------------------------------------------
q13 = "What is the lead time for Xilinx Artix-7 FPGA XC7A35T-1FTG256C?"
c13_0 = raw_data[q13][0] # SMT guidelines
add_entry(q13, c13_0, "AMBIGUOUS", notes="Relabeled per Policy A — market-data queries always route to AMBIGUOUS/Mouser fallback regardless of chunk content.")

c13_1 = raw_data[q13][2] # internal_avl.csv
add_entry(q13, c13_1, "AMBIGUOUS", notes="Relabeled per Policy A — market-data queries always route to AMBIGUOUS/Mouser fallback regardless of chunk content.")

# -------------------------------------------------------------
# 14. QUERY: What is the recommended decoupling capacitor placement for DRV8301 VCP and VSUP pins?
# -------------------------------------------------------------
q14 = "What is the recommended decoupling capacitor placement for DRV8301 VCP and VSUP pins?"
c14_0 = raw_data[q14][1] # DRV8301 Specs
add_entry(q14, c14_0, "AMBIGUOUS", notes="Lists PVDD supply voltage and buck regulator specs, but lacks exact layout capacitor placement guidelines.")

# -------------------------------------------------------------
# 15. QUERY: What is the operating voltage range for DRV8301 gate driver?
# -------------------------------------------------------------
q15 = "What is the operating voltage range for DRV8301 gate driver?"
c15_0 = raw_data[q15][0] # DRV8301 description
add_entry(q15, c15_0, "CORRECT", notes="Explicitly states DRV8301 operating supply voltage range from 6-V to 60-V.")

c15_1 = raw_data[q15][1] # DRV8301 features
add_entry(q15, c15_1, "CORRECT", notes="Explicitly lists 6-V to 60-V Operating Supply Voltage Range.")

c15_2 = raw_data[q15][2] # Pin Functions & Key Electrical Specs
add_entry(q15, c15_2, "CORRECT", notes="Lists PVDD Supply Voltage: 6 V to 60 V (Recommended Operating range 6 V to 60 V).")

c15_3 = raw_data[q15][3] # STM32 operating voltage
add_entry(q15, c15_3, "INCORRECT", notes="Near-miss distractor: lists 1.8V to 3.6V operating voltage range for STM32 MCU, NOT DRV8301.")

# -------------------------------------------------------------
# 16. QUERY: What is the moisture sensitivity level (MSL) rating for BGA-256 components?
# -------------------------------------------------------------
q16 = "What is the moisture sensitivity level (MSL) rating for BGA-256 components?"
c16_0 = raw_data[q16][0] # MSL-3 handling rules
add_entry(q16, c16_0, "CORRECT", notes="States MSL-3 handling and baking rules per IPC/JEDEC J-STD-033 standards.")

c16_1 = raw_data[q16][1] # SAC305 parameters
add_entry(q16, c16_1, "AMBIGUOUS", notes="Mentions Moisture Sensitivity Guidelines header, but body contains SAC305 reflow profile.")

# -------------------------------------------------------------
# 17. QUERY: What is the recommended stencil aperture reduction for fine pitch QFN packages?
# -------------------------------------------------------------
q17 = "What is the recommended stencil aperture reduction for fine pitch QFN packages?"
c17_0 = raw_data[q17][0] # Package options
add_entry(q17, c17_0, "INCORRECT", notes="Lists MCU package options (LQFP, WLCSP, UFBGA), no stencil aperture specs.")

c17_1 = raw_data[q17][1] # Package specific ceilings
add_entry(q17, c17_1, "INCORRECT", notes="Lists BGA peak temperature limits, no stencil design rules.")

# -------------------------------------------------------------
# 18. QUERY: What is the price break at 1000 units for STM32F407VGT6 microcontrollers?
# -------------------------------------------------------------
q18 = "What is the price break at 1000 units for STM32F407VGT6 microcontrollers?"
c18_0 = raw_data[q18][0] # STM32 Core & Architecture
add_entry(q18, c18_0, "AMBIGUOUS", notes="Relabeled per Policy A — market-data queries always route to AMBIGUOUS/Mouser fallback regardless of chunk content.")

c18_1 = raw_data[q18][1] # project_bom.csv
add_entry(q18, c18_1, "AMBIGUOUS", notes="Relabeled per Policy A — market-data queries always route to AMBIGUOUS/Mouser fallback regardless of chunk content.")


# Additional targeted manual entries to reach balanced distribution:
# -------------------------------------------------------------
# Target: ~15-20 CORRECT, ~12-15 AMBIGUOUS, ~12-15 INCORRECT (Total: 42 entries)

add_entry(
    "What is the maximum CPU clock frequency of STM32F407VGT6?",
    raw_data[q3][1],
    "CORRECT",
    source="retriever",
    notes="Explicitly states Max Clock Frequency: up to 168 MHz for STM32F405xx/STM32F407xx."
)

add_entry(
    "What is the Liquidus temperature for SAC305 solder paste?",
    raw_data[q1][1],
    "CORRECT",
    source="retriever",
    notes="Explicitly states Liquidus Temperature (TL): 217 °C."
)

add_entry(
    "What is the allowable soak duration during SAC305 reflow preheat zone?",
    raw_data[q1][1],
    "CORRECT",
    source="retriever",
    notes="Explicitly states Soak Duration: 60 seconds to 120 seconds."
)

add_entry(
    "What SPI register address controls DRV8301 overcurrent protection mode?",
    raw_data[q4][4],
    "CORRECT",
    source="retriever",
    notes="Explicitly states Address 0x02: Control Register 1 - Gate Driver Control & OCP Mode."
)

add_entry(
    "What is the maximum floor life of MSL-3 rated components after MBB opening?",
    raw_data[q1][2],
    "CORRECT",
    source="retriever",
    notes="Explicitly states MSL-3 floor life of 168 hours (7 days) at <= 30 °C / 60% RH."
)

add_entry(
    "What is the manufacturer of internal part IPN-FET-007?",
    raw_data[q3][0],
    "CORRECT",
    source="retriever",
    notes="Explicitly states IPN-FET-007 | CSD18532Q5B | Texas Instruments."
)

add_entry(
    "What is the peak gate drive source current for DRV8301?",
    raw_data[q15][0],
    "CORRECT",
    source="retriever",
    notes="Explicitly states DRV8301 supports up to 1.7-A source and 2.3-A peak current."
)

add_entry(
    "What is the default gain setting for DRV8301 current shunt amplifiers?",
    raw_data[q2][0],
    "CORRECT",
    source="retriever",
    notes="Explicitly states Bits [D3:D2] = 00 -> Gain of shunt amplifier: 10 V/V (Default)."
)

add_entry(
    "What is the part number for the 6-Axis motion tracking IMU on the project BOM?",
    raw_data[q2][4],
    "CORRECT",
    source="retriever",
    notes="Explicitly lists Item 4 | U4 | MPU-6050 | TDK InvenSense | 6-Axis Motion Tracking Sensor."
)

# Output evaluation dataset
target_path = root_dir / "data" / "gatekeeper_eval_dataset.json"
with open(target_path, "w", encoding="utf-8") as f:
    json.dump(dataset, f, indent=2)

print(f"Generated {len(dataset)} gatekeeper evaluation dataset entries at {target_path}.")

# Print stats
label_counts = {}
conf_counts = {}
for e in dataset:
    lbl = e["expected_label"]
    label_counts[lbl] = label_counts.get(lbl, 0) + 1
    c = e.get("confidence", "high")
    if c == "low":
        conf_counts[lbl] = conf_counts.get(lbl, 0) + 1

print("\n--- Distribution Summary ---")
for k, v in label_counts.items():
    print(f"  {k}: {v}")

print(f"\n--- Low Confidence Entries (Requires Review) ---")
print(f"  Total Flagged: {sum(conf_counts.values())}")
for e in dataset:
    if e.get("confidence") == "low":
        print(f"  - [{e['expected_label']}] Query: '{e['query']}' | Note: {e['notes']}")
