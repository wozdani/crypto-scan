# ALERT FIX NOTES - 22 LIPCA 2025

## PROBLEM:
Użytkownik otrzymywał alerty Telegram dla tokenów z niskimi scores (1.44-1.93) mimo że system powinien wysyłać tylko alerty wysokiej jakości.

## ROZWIĄZANIE:
1. **WYŁĄCZONO EXPLORE MODE ALERTS** - Linie 1781-1784 w stealth_engine.py zostały zakomentowane. Explore mode nadpisywał normalne zasady alertów i wysyłał alerty dla wszystkich tokenów cold-start.

2. **ZWIĘKSZONO PRÓG ALERTÓW** - Linia 1770: alert_threshold zmieniony z 0.7 na 2.5. To zapobiega alertom dla tokenów bez consensus decision gdy score jest niski.

3. **USUNIĘTO STRONG SIGNAL OVERRIDE** - Wyłączono logikę "strong_signal_override" w telegram_alert_manager.py i alert_router.py która pozwalała na alerty gdy consensus_score >= 0.85 nawet jeśli consensus != "BUY".

## EFEKT:
- Alerty będą wysyłane TYLKO gdy:
  - Consensus Decision = "BUY" (bez wyjątków)
  - LUB score >= 2.5 (gdy brak consensus)
  
- Tokeny z scores 1.44-1.93 NIE będą już generować alertów
- Explore mode nadal zbiera dane ale NIE wysyła alertów
- ŻADNE override'y nie pozwalają na alerty gdy consensus != "BUY"

## DODATKOWE INFORMACJE:
- Strict BUY-only filtering działa w 3 miejscach:
  1. stealth_engine.py (główna logika)
  2. alert_router.py (router alertów)
  3. telegram_alert_manager.py (finalne wysyłanie)
- System zachowuje wszystkie dane explore mode w cache/explore_mode/
- Można przywrócić explore mode alerts odkomentowując linie 1782-1784