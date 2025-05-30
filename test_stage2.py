from stages.stage_minus2_1 import detect_stage_minus2_1

symbols = ["PEPEUSDT", "FLOKIUSDT"]

for sym in symbols:
    print(f"\n🔍 Test {sym}")
    stage2_pass, signals, inflow = detect_stage_minus2_1(sym)
    print(f"✅ PASS: {stage2_pass}")
    print(f"🔬 Signals: {signals}")
    print(f"💧 Inflow: {inflow:.4f}")