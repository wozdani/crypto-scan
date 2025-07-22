# ALERT FIX NOTES - 22 LIPCA 2025

## PROBLEM:
Użytkownik otrzymywał alerty Telegram dla tokenów z niskimi scores (1.44-1.93) mimo że system powinien wysyłać tylko alerty wysokiej jakości.

## ROZWIĄZANIE:
1. **WYŁĄCZONO EXPLORE MODE ALERTS** - Linie 1781-1784 w stealth_engine.py zostały zakomentowane. Explore mode nadpisywał normalne zasady alertów i wysyłał alerty dla wszystkich tokenów cold-start.

2. **ZWIĘKSZONO PRÓG ALERTÓW** - Linia 1770: alert_threshold zmieniony z 0.7 na 2.5. To zapobiega alertom dla tokenów bez consensus decision gdy score jest niski.

## EFEKT:
- Alerty będą wysyłane TYLKO gdy:
  - Consensus Decision = "BUY" (niezależnie od score)
  - LUB score >= 2.5 (gdy brak consensus)
  
- Tokeny z scores 1.44-1.93 NIE będą już generować alertów
- Explore mode nadal zbiera dane ale NIE wysyła alertów

## DODATKOWE INFORMACJE:
- BUY-only filtering nadal działa w alert_router.py
- System zachowuje wszystkie dane explore mode w cache/explore_mode/
- Można przywrócić explore mode alerts odkomentowując linie 1782-1784