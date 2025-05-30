from stages.stage_minus2_1 import detect_stage_minus2_1
from utils.data_fetchers import get_test_symbols

print("🔍 Test Stage –2.1: Mikroanomalia Volume Spike")

for symbol in get_test_symbols():
    stage2_pass, signals, volume = detect_stage_minus2_1(symbol)
    print(f"📊 {symbol} | ✅ PASS: {stage2_pass} | 🔬 Signals: {signals} | 💧 Volume: {volume}")