# GPT Learning System - Samodoskonalący System Uczenia Się

## Przegląd Systemu

System uczenia się GPT to zaawansowany mechanizm, który automatycznie zapisuje, testuje i ewoluuje funkcje detektora pre-pump generowane przez GPT-4o. System umożliwia ciągłe samodoskonalenie poprzez analizę skuteczności funkcji i automatyczne tworzenie ulepszonych wersji.

## Architektura Systemu

### Struktura Folderów
```
pump-analysis/
├── generated_functions/          # Zapisane funkcje GPT
│   ├── detect_pre_pump_btcusdt_20250618_170530.py
│   ├── detect_pre_pump_ethusdt_20250618_171045.py
│   └── ...
├── deprecated_functions/         # Funkcje przestarzałe
├── test_results/                # Wyniki testów
├── retrospective_tests/         # Testy retrospektywne
├── function_logs.json           # Logi skuteczności funkcji
├── gpt_recommendations.json     # Rekomendacje GPT
└── learning_system.py          # Główny system
```

### Główne Komponenty

#### 1. LearningSystem (learning_system.py)
Główna klasa zarządzająca całym systemem uczenia się:
- **save_gpt_function()**: Zapisuje nową funkcję z metadanymi
- **test_functions_on_new_pump()**: Testuje wszystkie funkcje na nowym pumpie
- **evolve_function()**: Tworzy ulepszoną wersję funkcji
- **retrospective_test_suite()**: Testy na ostatnich 20 pumpach
- **deprecate_function()**: Przenosi funkcję do deprecated

#### 2. FunctionPerformance (dataclass)
Struktura danych do śledzenia wydajności funkcji:
```python
@dataclass
class FunctionPerformance:
    function_name: str
    symbol: str
    pump_date: str
    active_signals: List[str]
    retrospective_test_passed: bool
    detection_timestamp: Optional[str]
    detection_accuracy_minutes: Optional[int]
    confidence_score: float
    version: int
```

## Przepływ Systemu Uczenia Się

### 1. 🔁 Zapisywanie Funkcji GPT
Gdy GPT generuje nową funkcję dla pumpa:

```python
# Automatyczne zapisywanie w main.py
function_path = self.learning_system.save_gpt_function(
    detector_function,     # Kod funkcji Python
    pump.symbol,          # Symbol (np. 'BTCUSDT')
    pump.start_time.strftime('%Y%m%d'),  # Data pumpa
    active_signals,       # Lista aktywnych sygnałów
    pre_pump_data        # Dane pre-pump
)
```

**Wynik**: Funkcja zapisana w `generated_functions/` z metadanymi i wpis w `function_logs.json`.

### 2. 🧠 Tworzenie Logu Skuteczności
Każda funkcja otrzymuje szczegółowy wpis w `function_logs.json`:

```json
{
  "functions": {
    "detect_pre_pump_btcusdt_20250618": {
      "created": "2025-06-18T17:05:30",
      "symbol": "BTCUSDT",
      "pump_date": "20250618",
      "active_signals": ["volume_spike", "compression", "rsi_neutral"],
      "version": 1,
      "tests": [],
      "performance": {
        "total_tests": 0,
        "successful_detections": 0,
        "accuracy_score": 0.0
      },
      "evolution": {
        "parent_function": null,
        "improvements": [],
        "deprecated": false
      }
    }
  }
}
```

### 3. 🔍 Testowanie na Nowych Pumpach
Gdy system wykryje nowy pump:

1. **Pobiera dane pre-pump** (2 godziny przed pumpem)
2. **Testuje wszystkie istniejące funkcje** na tych danych
3. **Sprawdza różne okna czasowe** (60, 45, 30, 15, 10, 5 minut przed)
4. **Zapisuje wyniki** w `function_logs.json`

```python
# Automatyczne testowanie w main.py
learning_test_results = self.learning_system.test_functions_on_new_pump(
    pump_data,           # Dane o nowym pumpie
    pre_pump_candles    # DataFrame z danymi candle
)
```

### 4. 🔄 Ewolucja Funkcji
System może automatycznie ewoluować funkcje:

#### Warunki Ewolucji:
- **Bliska detekcja**: Funkcja wykryła pump 5-15 minut po czasie
- **Częściowy sukces**: Funkcja działa dla podobnych symboli
- **Nowe wzorce**: Pojawiły się nowe sygnały

#### Proces Ewolucji:
```python
evolved_path = learning_system.evolve_function(
    function_name="detect_pre_pump_btcusdt_20250618",
    improvement_reason="Dodano detekcję fake_reject patterns",
    new_function_code=improved_code
)
```

**Wynik**: Nowa funkcja z sufiksem `_v2`, `_v3`, etc.

### 5. ✅ Rekomendacje Systemu
System generuje automatyczne rekomendacje:

#### Typy Rekomendacji:
- **create_new_function**: Potrzeba nowej funkcji dla unikalnego wzorca
- **evolve_functions**: Ulepszenie istniejących funkcji
- **promote_functions**: Promowanie najlepszych funkcji
- **deprecate_functions**: Usunięcie nieefektywnych funkcji

## Testy Retrospektywne

### Automatyczne Testy (co 12h)
System automatycznie uruchamia testy retrospektywne:

```python
# W scheduler.py - co 12 godzin
recent_pumps = get_recent_pumps_data(days=1)  # Ostatnie 24h
retro_results = learning_system.retrospective_test_suite(recent_pumps)
```

### Statystyki Wydajności
System oblicza:
- **Accuracy Rate**: % pomyślnych detekcji
- **Average Detection Time**: Średni czas detekcji przed pumpem
- **Best Performing Functions**: Top 5 najlepszych funkcji
- **Functions to Deprecate**: Funkcje o accuracy < 10%

## Integracja z Telegram

Wyniki systemu uczenia się są automatycznie dodawane do powiadomień Telegram:

```
🧠 SYSTEM UCZĄCY:
📊 Testowano 15 istniejących funkcji
✅ 3 funkcji wykryło pump
🏆 Najlepsza: detect_pre_pump_btcusdt_v3
🔶 2 funkcji było blisko wykrycia
```

## Klucze Konfiguracyjne

### Environment Variables
```bash
OPENAI_API_KEY=sk-...           # Wymagane dla GPT
TELEGRAM_BOT_TOKEN=...          # Powiadomienia
TELEGRAM_CHAT_ID=...            # Kanał powiadomień
```

### Parametry Systemu
```python
# W learning_system.py
MIN_ACCURACY_FOR_PROMOTION = 0.7    # 70% accuracy
MIN_TESTS_FOR_DEPRECATION = 5       # Minimum testów
CLOSE_DETECTION_THRESHOLD = 15      # Minut po pumpie
```

## Użycie Systemu

### Podstawowe Operacje

#### 1. Inicjalizacja
```python
learning_system = LearningSystem()
```

#### 2. Zapisanie Nowej Funkcji
```python
function_path = learning_system.save_gpt_function(
    function_code="def detect_pre_pump_btcusdt()...",
    symbol="BTCUSDT",
    pump_date="20250618",
    active_signals=["volume_spike", "compression"],
    pre_pump_data=analysis_data
)
```

#### 3. Test na Nowym Pumpie
```python
test_results = learning_system.test_functions_on_new_pump(
    pump_data={"symbol": "ETHUSDT", "start_time": "..."},
    pre_pump_candles=candle_dataframe
)
```

#### 4. Ewolucja Funkcji
```python
evolved_path = learning_system.evolve_function(
    function_name="detect_pre_pump_btcusdt",
    improvement_reason="Added VWAP analysis",
    new_function_code=improved_code
)
```

#### 5. Testy Retrospektywne
```python
retro_results = learning_system.retrospective_test_suite(
    recent_pumps_data=last_20_pumps
)
```

#### 6. Podsumowanie Systemu
```python
summary = learning_system.get_learning_summary()
print(f"Aktywnych funkcji: {summary['active_functions']}")
print(f"Średnia skuteczność: {summary['avg_accuracy']:.1%}")
```

## Struktura Wygenerowanych Funkcji

Każda wygenerowana funkcja ma standardową strukturę:

```python
"""
GPT Generated Detector Function
Generated: 2025-06-18T17:05:30
Symbol: BTCUSDT
Pump Date: 20250618
Active Signals: volume_spike, compression, rsi_neutral
Pump Increase: 18.5%
Duration: 25 minutes
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional

def detect_pre_pump_btcusdt_20250618(data: pd.DataFrame) -> Dict:
    """
    Detect pre-pump conditions for BTCUSDT based on 2025-06-18 pump analysis
    
    Args:
        data: DataFrame with OHLCV data
        
    Returns:
        Dict with detection results
    """
    # Implementation code here...
    
    return {
        'signal_detected': bool,
        'confidence': float,
        'active_signals': list,
        'detection_reason': str
    }

# Metadata for learning system
FUNCTION_METADATA = {
    "function_name": "detect_pre_pump_btcusdt_20250618",
    "symbol": "BTCUSDT",
    "pump_date": "20250618",
    "active_signals": ["volume_spike", "compression", "rsi_neutral"],
    "generated_timestamp": "2025-06-18T17:05:30",
    "version": 1
}
```

## Monitoring i Debugowanie

### Logi Systemu
System generuje szczegółowe logi:
```
🧠 Learning system initialized
💾 Saved GPT function: detect_pre_pump_btcusdt_20250618_170530.py
🧪 Testing existing functions on new pump ETHUSDT...
📊 Learning test complete: 15 functions tested, 3 detected
✅ Function saved to learning system: /path/to/function.py
```

### Pliki Danych
- **function_logs.json**: Główne logi skuteczności
- **test_results/**: Szczegółowe wyniki testów
- **retrospective_tests/**: Wyniki testów retrospektywnych
- **gpt_recommendations.json**: Rekomendacje systemu

## Wydajność Systemu

### Optymalizacje
- **Lazy Loading**: Funkcje ładowane tylko przy testowaniu
- **Parallel Testing**: Równoległe testowanie funkcji
- **Smart Caching**: Cache'owanie danych pre-pump
- **Efficient Storage**: Kompresja logów starszych niż 30 dni

### Limity
- **Maksymalnie 1000 funkcji** w generated_functions/
- **Automatyczne archiwizowanie** funkcji starszych niż 90 dni
- **Maksymalnie 5MB** dla function_logs.json

## Przyszłe Rozszerzenia

### Planowane Funkcje
1. **ML Model Training**: Użycie najlepszych funkcji do treningu modeli
2. **Pattern Recognition**: Automatyczne grupowanie podobnych wzorców
3. **Cross-Symbol Learning**: Transfer learning między symbolami
4. **Real-time Optimization**: Dynamiczna optymalizacja parametrów
5. **Advanced Evolution**: Genetic algorithms dla ewolucji funkcji

### Integracja z PPWCS
System uczenia się może być zintegrowany z głównym systemem PPWCS:
- **Dynamic Scoring**: Wykorzystanie najlepszych funkcji w scoringu
- **Pattern Boost**: Dodatkowe punkty za rozpoznane wzorce
- **Early Warning**: Detekcja pre-pump przed głównym systemem

## Troubleshooting

### Częste Problemy

#### 1. Brak Funkcji w generated_functions/
```bash
# Sprawdź uprawnienia
ls -la pump-analysis/generated_functions/
# Sprawdź logi
tail -n 50 pump-analysis/function_logs.json
```

#### 2. Błędy Testowania Funkcji
```python
# Sprawdź składnię funkcji
python -m py_compile generated_functions/funkcja.py
# Sprawdź DataFrame
print(pre_pump_candles.head())
```

#### 3. Niska Skuteczność Funkcji
- Sprawdź jakość danych pre-pump
- Przeanalizuj parametry detekcji
- Rozważ ewolucję funkcji

#### 4. Problemy z Pamięcią
```bash
# Archiwizuj stare funkcje
mv generated_functions/old_* deprecated_functions/
# Kompresuj logi
gzip function_logs_backup.json
```

System uczenia się GPT to potężne narzędzie do ciągłego doskonalenia detekcji pre-pump. Dzięki automatyzacji i inteligentnej ewolucji funkcji, system może adaptować się do zmieniających się warunków rynkowych i odkrywać nowe wzorce pump.