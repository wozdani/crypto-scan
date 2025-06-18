# Pump Analysis System 🚀

Automatyczny system wykrywania i analizy pump'ów kryptowalut wykorzystujący sztuczną inteligencję do generowania insightów z rzeczywistych przypadków rynkowych.

## Funkcjonalności

### 🔍 Wykrywanie Pump'ów
- Analiza danych świecowych 5-minutowych z Bybit API
- Wykrycie wzrostów ≥15% w ciągu 30 minut
- Analiza ostatnich 7 dni dla wybranych tokenów

### 📊 Analiza Pre-Pump
- Szczegółowa analiza 60 minut przed pump'em
- Wskaźniki techniczne: RSI, VWAP, wolumen
- Wykrywanie fake reject'ów i kompresji cenowej
- Identyfikacja poziomów wsparcia/oporu

### 🤖 Analiza GPT
- Automatyczne generowanie analizy przez GPT-4o
- Identyfikacja kluczowych sygnałów ostrzegawczych
- Praktyczne wnioski do zastosowania w przyszłości
- Odpowiedzi w języku polskim

### 📱 Powiadomienia Telegram
- Automatyczne wysyłanie analiz na kanał Telegram
- Formatowanie dla czytelności
- Obsługa długich wiadomości (podział na części)

## Instalacja

1. **Sklonuj/pobierz projekt**
2. **Zainstaluj zależności:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Skonfiguruj zmienne środowiskowe:**
   ```bash
   cp .env.example .env
   # Edytuj .env i dodaj swoje klucze API
   ```

## Konfiguracja

### Wymagane zmienne środowiskowe:

```bash
# OpenAI API
OPENAI_API_KEY=sk-...

# Telegram Bot
TELEGRAM_BOT_TOKEN=123456789:ABC...
TELEGRAM_CHAT_ID=-1001234567890

# Opcjonalne parametry
MIN_PUMP_INCREASE_PCT=15.0
DETECTION_WINDOW_MINUTES=30
ANALYSIS_DAYS_BACK=7
MAX_SYMBOLS_TO_ANALYZE=30
```

### Uzyskanie kluczy API:

1. **OpenAI API Key:**
   - Idź na https://platform.openai.com/api-keys
   - Utwórz nowy klucz API

2. **Telegram Bot:**
   - Napisz do @BotFather na Telegram
   - Utwórz nowego bota: `/newbot`
   - Skopiuj token bota

3. **Telegram Chat ID:**
   - Dodaj bota do swojego kanału/grupy
   - Napisz wiadomość na kanale
   - Idź na: `https://api.telegram.org/bot<TOKEN>/getUpdates`
   - Znajdź chat_id w odpowiedzi

## Uruchomienie

```bash
python main.py
```

## Struktura Projektu

```
pump_analysis/
├── main.py              # Główny skrypt systemu
├── requirements.txt     # Zależności Python
├── .env.example        # Przykład konfiguracji
├── README.md           # Dokumentacja
└── pump_data/          # Folder na zapisane analizy (tworzony automatycznie)
```

## Jak to działa

### 1. Wykrywanie Pump'ów
System pobiera dane z ostatnich 7 dni dla najpopularniejszych tokenów USDT z Bybit i szuka wzrostów ≥15% w ciągu 30 minut.

### 2. Analiza Pre-Pump
Dla każdego wykrytego pump'a system analizuje 60 minut poprzedzających ruch, licząc wskaźniki techniczne i wykrywając wzorce.

### 3. Generowanie Analizy GPT
Wszystkie dane pre-pump są formatowane w prompt i wysyłane do GPT-4o, który generuje szczegółową analizę w języku polskim.

### 4. Wysyłka na Telegram
Kompletna analiza jest automatycznie wysyłana na skonfigurowany kanał Telegram oraz zapisywana do pliku JSON.

## Przykład Analizy

```
🎯 WYKRYTY PUMP - ANALIZA PRE-PUMP

💰 Symbol: PEPEUSDT
📈 Wzrost: +18.5%
⏰ Czas pumpu: 2025-06-18 14:30:00 UTC
💵 Cena przed: $0.000012
🚀 Cena szczyt: $0.000014
📊 Volume spike: 3.2x

📝 ANALIZA GPT (60 min przed pumpem):

Na podstawie analizy danych pre-pump można zidentyfikować kilka kluczowych sygnałów...
[szczegółowa analiza GPT]
```

## Automatyzacja

Aby uruchamiać system automatycznie co 12 godzin, możesz:

1. **Na serwerze Linux (cron):**
   ```bash
   # Edytuj crontab
   crontab -e
   
   # Dodaj linię (uruchomienie co 12h)
   0 */12 * * * cd /path/to/pump_analysis && python main.py
   ```

2. **Na Replit:**
   System może być uruchamiany przez Always On lub cykliczne uruchomienia

## Uwagi Techniczne

- System używa Bybit API bez wymagania kluczy (publiczne dane)
- OpenAI API wymaga płatnego konta z dostępem do GPT-4o
- Telegram Bot API jest darmowy
- Wszystkie analizy są zapisywane lokalnie w formacie JSON
- System obsługuje rate limiting i błędy API

## Bezpieczeństwo

- Nie commituj pliku `.env` do repozytorium
- Używaj silnych tokenów API
- Regularnie rotuj klucze dostępu
- Ogranicz dostęp do kanału Telegram

## Rozwiązywanie Problemów

### Błąd: "Missing required environment variables"
Sprawdź czy plik `.env` zawiera wszystkie wymagane klucze.

### Błąd: "OpenAI API error"
Sprawdź czy masz wystarczające środki na koncie OpenAI i dostęp do GPT-4o.

### Błąd: "Telegram API error"
Sprawdź czy bot ma uprawnienia do pisania na kanale i czy chat_id jest poprawny.

### Brak wykrytych pump'ów
To normalne - pump'y nie występują codziennie. System może nie znajdować pump'ów przez kilka dni.

## Rozwój

System jest modularny i można łatwo:
- Dodać nowe wskaźniki techniczne
- Zmienić kryteria wykrywania pump'ów
- Rozszerzyć analizę o więcej chainów
- Dodać więcej źródeł danych

## Licencja

Projekt do użytku prywatnego i edukacyjnego.