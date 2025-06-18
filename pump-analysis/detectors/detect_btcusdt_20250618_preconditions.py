
def detect_btc_test_preconditions(df):
    """Test detector function for GPT Memory Engine validation"""
    
    if len(df) < 4:
        return False
    
    # RSI analysis
    rsi = df['rsi_14'].iloc[-1]
    if not (45 <= rsi <= 60):
        return False
    
    # Volume spike detection
    recent_volume = df['volume'].iloc[-3:].mean()
    baseline_volume = df['volume'].iloc[-10:-3].mean()
    if recent_volume < baseline_volume * 1.5:
        return False
    
    # Compression check
    price_range = df['high'].iloc[-5:].max() - df['low'].iloc[-5:].min()
    avg_range = (df['high'] - df['low']).iloc[-20:].mean()
    if price_range > avg_range * 0.8:
        return False
    
    return True
