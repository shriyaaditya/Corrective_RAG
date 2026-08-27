import json
import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(root_dir / "app"))

from app.services.retriever.qdrant_store import QdrantStore
from app.services.retriever.vector_retriever import VectorRetriever

store = QdrantStore(store_dir=str(root_dir / "qdrant_db"))
retriever = VectorRetriever(store)

queries = [
    "What is the maximum body temperature for a BGA-256 package during SAC305 reflow?",
    "What is the recommended gain setting for the DRV8301 current shunt amplifier to achieve a 40 V/V output?",
    "What is the global stock and unit price for part STM32F407VGT6?",
    "What is the lifecycle status and lead time for internal part number IPN-DRV-001?",
    "What is the specific solder reflow temperature profile for a BGA-256 package in a quantum computing array?",
    "What is the pinout configuration and thermal pad ground requirement for DRV8301?",
    "What is the maximum allowed warp and twist percentage for FR-4 PCBs under IPC-A-600?",
    "How does temperature coefficient affect resistor tolerances in high precision motor drive circuits?",
    "What are the Mouser stocking status and minimum order quantity (MOQ) for Texas Instruments DRV8301DCAR?",
    "What internal part number maps to Texas Instruments TPS62130RGTR in the AVL?",
    "What is the minimum trace width and clearance for 1 oz copper in high-current motor driver PCBs?",
    "What is the maximum reflow peak temperature for QFN-48 packages under J-STD-020 standard?",
    "What is the lead time for Xilinx Artix-7 FPGA XC7A35T-1FTG256C?",
    "What is the recommended decoupling capacitor placement for DRV8301 VCP and VSUP pins?",
    "What is the operating voltage range for DRV8301 gate driver?",
    "What is the moisture sensitivity level (MSL) rating for BGA-256 components?",
    "What is the recommended stencil aperture reduction for fine pitch QFN packages?",
    "What is the price break at 1000 units for STM32F407VGT6 microcontrollers?"
]

results = {}
for q in queries:
    chunks = retriever.retrieve(q, top_k=5)
    results[q] = [c.text for c in chunks]

with open(root_dir / "data" / "raw_retrieved_chunks.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print(f"Retrieved real chunks for {len(queries)} queries.")
