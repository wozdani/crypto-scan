from utils.data_fetchers import get_all_data
import numpy as np

def detect_volume_spike(symbol):
    """
    Wykrywa nagły skok wolumenu: Z-score > 2.5 lub 3x średnia z poprzednich 4 świec.
    """
    data = get_all_data(symbol)
    if not data or not data["prev_candle"]:
        return False, 0.0

    try:
        current_volume = float(data["volume"])
        prev_volume = float(data["prev_candle"][5])

        # Dla testów używamy tylko jednej wcześniejszej świecy (w przyszłości 4)
        volumes = [prev_volume, current_volume]
        mean_volume = np.mean(volumes)
        std_volume = np.std(volumes)

        z_score = (current_volume - mean_volume) / std_volume if std_volume > 0 else 0
        spike_detected = z_score > 2.5 or current_volume > 3 * prev_volume

        return spike_detected, current_volume
    except Exception as e:
        print(f"❌ Błąd w volume spike dla {symbol}: {e}")
        return False, 0.0

def detect_stage_minus2_1(symbol):
    """
    Główna funkcja detekcji Stage –2.1 (mikroanomalii).
    Zwraca:
        - stage2_pass (czy przechodzi dalej)
        - signals (słownik aktywnych mikroanomalii)
        - dex_inflow_volume (float)
    """
    signals = {
        "social_spike": False,  # TODO
        "whale_tx": False,      # TODO
        "orderbook_anomaly": False,  # TODO
        "volume_spike": False,
        "dex_inflow": False,    # TODO
    }

    # Realna logika volume spike
    volume_spike_detected, volume = detect_volume_spike(symbol)
    if volume_spike_detected:
        signals["volume_spike"] = True

    # Finalna decyzja: aktywowany jeśli co najmniej 1 aktywny sygnał
    stage2_pass = any(signals.values())

    return stage2_pass, signals, volume
