# SMT Lead-Free SAC305 Reflow Soldering & Moisture Sensitivity Guidelines

## 1. SAC305 Lead-Free Reflow Profile Parameters
For Surface Mount Technology (SMT) assembly utilizing Lead-Free SAC305 (96.5% Sn / 3.0% Ag / 0.5% Cu) solder paste, the thermal profile parameters must adhere to the following specification:

- **Preheat / Soak Zone:**
  - Temperature Range: 150 °C to 200 °C
  - Soak Duration: 60 seconds to 120 seconds
  - Ramp-Up Rate: Maximum 3 °C/sec

- **Time Above Liquidus (TAL):**
  - Liquidus Temperature ($T_L$): 217 °C
  - Time Above Liquidus ($t_L$): 60 seconds to 90 seconds (Maximum 100 seconds)

- **Peak Package Temperature ($T_P$):**
  - Required Peak Temperature Range: 245 °C to 255 °C
  - Maximum Allowable Package Temperature: 260 °C (maximum 10 seconds at peak)

- **Cooling Zone:**
  - Ramp-Down Rate: 2 °C/sec to 4 °C/sec (Maximum 6 °C/sec to prevent thermal shock)

---

## 2. Component Moisture Sensitivity Level (MSL-3) Handling & Baking Rules
Per IPC/JEDEC J-STD-033 standards:

- **Floor Life:** MSL-3 components have a maximum allowable floor life of **168 hours** (7 days) at $\le 30 \text{ }^\circ\text{C} / 60\% \text{ RH}$ after opening the Moisture Barrier Bag (MBB).
- **Baking Requirements:**
  - High-Temp Bake (in trays): 125 °C for 12 to 24 hours.
  - Low-Temp Bake (in tape & reel): 40 °C (+5/-0 °C) at $< 5\% \text{ RH}$ for 192 hours (8 days).

---

## 3. Package Specific Ceilings (BGA, QFN, HTSSOP)
- **BGA Temperature Ceilings:** Peak Package Temperature Limit of 245 °C. Maximum delta Across BGA Package Body ($\Delta T$): $\le 5 \text{ }^\circ\text{C}$.
