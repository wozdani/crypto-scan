from utils.stage_detectors import detect_stage_minus2
from utils.data_fetchers import get_test_symbols, get_all_data

print("🔍 Test Stage –2: Mikroanomalia Volume Spike")

for symbol in get_test_symbols():
    data = get_all_data(symbol)
    stage2_pass, signals, inflow = detect_stage_minus2(symbol, data)
    print(f"📊 {symbol} | ✅ PASS: {stage2_pass} | 🔬 Signals: {signals} | 💧 Inflow: {inflow}")