# Automated Detector Testing System

Kompletny system automatycznego testowania wygenerowanych funkcji detektorów pre-pump. System umożliwia walidację funkcji na różnych scenariuszach rynkowych oraz benchmark testing dla zapewnienia jakości detektorów.

## 📁 Struktura Systemu

```
pump_analysis/
├── generated_detectors/          # Wygenerowane funkcje detektorów
│   ├── __init__.py              # Moduł dynamicznego ładowania
│   └── SYMBOL_YYYYMMDD.py       # Funkcje detektorów (np. BTCUSDT_20250613.py)
├── test_results/                # Wyniki testów
├── test_detectors.py           # Pełny system testów z prawdziwymi danymi
├── benchmark_detectors.py      # Benchmark testing z kontrolowanymi scenariuszami  
├── quick_test_detector.py      # Szybkie testowanie pojedynczych detektorów
└── test_detector_system.py     # Testy systemu dynamicznego ładowania
```

## 🔧 Dostępne Narzędzia Testowe

### 1. Benchmark Testing (Zalecane)
```bash
cd pump_analysis
python benchmark_detectors.py
```

**Funkcjonalność:**
- Testuje wszystkie detektory na 5 kontrolowanych scenariuszach
- Wykorzystuje syntetyczne dane z realistycznymi wzorcami
- Nie wymaga dostępu do zewnętrznych API
- Generuje szczegółowe raporty z dokładnością każdego detektora

**Scenariusze testowe:**
- `pump_pattern`: Dane z sygnałami pre-pump (powinno zwrócić True)
- `normal_market`: Normalne warunki rynkowe (powinno zwrócić False)  
- `compression_only`: Tylko kompresja cenowa bez innych sygnałów
- `high_volume_no_compression`: Wysoki wolumen bez kompresji
- `low_rsi_compression`: Kompresja z niskim RSI (poza strefą akumulacji)

### 2. Szybkie Testowanie Pojedynczego Detektora
```bash
cd pump_analysis
python quick_test_detector.py BTCUSDT 20250613 --scenario pump
python quick_test_detector.py BTCUSDT 20250613 --scenario normal
python quick_test_detector.py BTCUSDT 20250613 --scenario compression
```

**Funkcjonalność:**
- Testuje konkretny detektor na wybranym scenariuszu
- Pokazuje szczegółową analizę danych wejściowych
- Wyjaśnia dlaczego detektor wykrył lub nie wykrył sygnał
- Identyfikuje brakujące warunki dla lepszego dostrojenia

### 3. Testowanie z Prawdziwymi Danymi (Wymaga API)
```bash
cd pump_analysis  
python test_detectors.py
```

**Funkcjonalność:**
- Ładuje rzeczywiste dane pre-pump z Bybit API
- Testuje detektory na ich własnych przypadkach pumpu
- Przeprowadza cross-validation na innych przypadkach
- Wymaga ustawienia kluczy API (BYBIT_API_KEY, BYBIT_SECRET_KEY)

### 4. Test Systemu Ładowania
```bash
cd pump_analysis
python test_detector_system.py
```

**Funkcjonalność:**
- Sprawdza czy system dynamicznego ładowania działa poprawnie
- Testuje odkrywanie dostępnych detektorów
- Waliduje strukturę katalogów i plików

## 📊 Interpretacja Wyników

### Benchmark Results
```
🔍 detect_BTCUSDT_20250613_preconditions:
   pump_pattern: False (exp: True) ❌        # Nie wykrył pump pattern - może wymagać dostrojenia
   normal_market: False (exp: False) ✅      # Poprawnie odrzucił normalny rynek
   compression_only: False (exp: False) ✅   # Poprawnie odrzucił samą kompresję
   high_volume_no_compression: False (exp: False) ✅  # Poprawnie odrzucił sam wolumen
   low_rsi_compression: False (exp: False) ✅         # Poprawnie odrzucił złe RSI
   Accuracy: 4/5 (80.0%)                    # Ogólna dokładność
```

### Status Codes (quick_test_detector.py)
- **Exit 0**: Detektor wykrył sygnał (True)
- **Exit 1**: Błąd systemu (brak pliku, exception)
- **Exit 2**: Detektor nie wykrył sygnału (False)

## 🎯 Oczekiwane Wyniki

### Idealny Detektor
- **pump_pattern**: True (wykrywa swój wzorzec)
- **normal_market**: False (nie daje fałszywych alarmów)
- **compression_only**: False (wymaga więcej niż samo compression)
- **high_volume_no_compression**: False (wolumen sam w sobie to za mało)
- **low_rsi_compression**: False (RSI poza strefą akumulacji)

### Dobry Detektor (≥80% dokładność)
- Wykrywa przynajmniej 4/5 scenariuszy poprawnie
- Niska liczba false positive (max 1 błędny alarm)
- Wykrywa pump_pattern lub ma bardzo dobre powody żeby nie wykryć

### Detektor Wymagający Poprawy (<80% dokładność)
- Dużo false positive alarmów
- Nie wykrywa pump_pattern i żadnych podobnych scenariuszy
- Zbyt restrykcyjne lub zbyt liberalne progi

## 🔧 Dostrajanie Detektorów

### Analiza Wyniku False Negative (nie wykrył pump_pattern)
```bash
python quick_test_detector.py SYMBOL DATE --scenario pump
```

Sprawdź w wyniku:
- **Price range**: Czy jest wystarczająca kompresja (<3%)?
- **Current RSI**: Czy jest w strefie akumulacji (50-65)?
- **VWAP premium**: Czy cena jest powyżej VWAP (>1%)?
- **Volume spike**: Czy wykryto skok wolumenu (>2.5x)?
- **Fake reject patterns**: Czy są wzorce odrzucenia?

### Typowe Przyczyny Problemów

1. **Zbyt restrykcyjne progi volume spike** (>3.5x zamiast >2.5x)
2. **Zbyt wąski zakres RSI** (52-56 zamiast 50-65)
3. **Wymóg wszystkich warunków** zamiast np. 4 z 6
4. **Nieprawidłowe obliczenia wskaźników** (błędy w RSI/VWAP)

## 🚀 Wykorzystanie w Produkcji

### 1. Automatyczna Klasyfikacja
```python
from generated_detectors import test_all_detectors
import pandas as pd

# Załaduj nowe dane rynkowe
df = load_market_data('NEWTOKEN', '2025-06-15')

# Przetestuj wszystkimi detektorami
results = test_all_detectors(df)

# Policz score
detection_count = sum(1 for result in results.values() if result is True)
confidence_score = (detection_count / len(results)) * 100
```

### 2. Wzmacnianie PPWCS
```python
# Dodaj bonus do PPWCS na podstawie liczby wykryć przez detektory
if detection_count >= 3:
    ppwcs_bonus = 15  # Strong pattern recognition
elif detection_count >= 2:
    ppwcs_bonus = 10  # Moderate pattern recognition
elif detection_count >= 1:
    ppwcs_bonus = 5   # Weak pattern recognition
```

### 3. Trenowanie ML Klasyfikatorów
```python
# Użyj wyników detektorów jako features
features = []
for detector_name, result in results.items():
    features.append(1 if result else 0)

# Dodaj do datasetu ML
ml_features = base_features + features
```

## 📈 Ciągłe Doskonalenie

### Codzienne Sprawdzenie
```bash
# Automatyczne uruchomienie benchmark
cd pump_analysis && python benchmark_detectors.py > daily_benchmark.log
```

### Analiza Trendów
- Monitoruj accuracy wszystkich detektorów
- Identyfikuj detektory z malejącą skutecznością  
- Usuń detektory z accuracy <60%
- Priorytetyzuj detektory z accuracy >90%

### Budowanie Biblioteki
- Każdy nowy pump generuje nowy detektor
- Testuj nowe detektory na historycznych danych
- Buduj kolekcję najlepszych wzorców (top 10% accuracy)
- Używaj najlepszych detektorów do validation nowych przypadków

## 🎯 Następne Kroki

1. **Wdrożenie na serwerze produkcyjnym** z pełnymi kluczami API
2. **Integracja z głównym systemem skanowania** (PPWCS boost)
3. **Automatyzacja testowania** (cron job dla daily benchmark)
4. **ML Pipeline** do wykorzystania detektorów jako features
5. **Continuous improvement** bazując na real-world performance