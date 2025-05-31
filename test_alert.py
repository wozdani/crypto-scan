import time
from utils.alerts import send_alert
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Symulowane dane testowe
symbol = "PEPEUSDT"
ppwcs_score = 84  # silny sygnał
trigger = "stealth_acc + volume_spike"
tp_forecast = {"TP1": "+6%", "TP2": "+14%", "TP3": "+28%", "trailing": True}

# Wysyłka alertu
send_alert(symbol, ppwcs_score, trigger, tp_forecast)

# Drugi test – powinien nie wysłać z powodu cooldownu (60 min)
time.sleep(2)
send_alert(symbol, ppwcs_score + 1, trigger, tp_forecast)
