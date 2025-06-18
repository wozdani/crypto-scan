# GPT Function History Context System

## Overview

The GPT Function History Context System is an advanced AI enhancement that enables GPT-4o to learn from previously generated detector functions, creating a continuous learning loop that improves pattern recognition and detection accuracy over time.

## Architecture

### Core Components

1. **Function History Storage** (`function_history.json`)
   - Rolling window of 5 most recent detector functions
   - Metadata: symbol, date, function code, pump increase %, creation timestamp
   - Automatic persistence and loading across system restarts

2. **Context Formatting System**
   - Structured presentation of historical functions for GPT consumption
   - Includes implementation guidelines and pattern comparison instructions
   - Formatted as Polish language context for consistency with system prompts

3. **Integration Points**
   - Automatic history updates after each detector function generation
   - Context injection into GPT prompts during new function creation
   - Seamless integration with existing pump analysis workflow

### Data Flow

```
1. GPT generates new detector function
2. Function automatically added to history (max 5 entries)
3. History persisted to function_history.json
4. Next GPT request includes historical context
5. GPT compares new case with previous patterns
6. Enhanced function generated with learned improvements
```

## Key Features

### Historical Context Awareness
- GPT receives 3-5 previous detector functions as reference material
- Pattern comparison enables logic evolution and improvement
- Avoids duplication of identical detection logic

### Enhanced Prompt Engineering
- System prompts include explicit instructions for historical analysis
- Guidelines for pattern comparison and logic refinement
- Encouragement to build upon successful previous implementations

### Memory Persistence
- JSON-based storage survives system restarts
- Maintains learning continuity across analysis sessions
- Rolling window prevents memory bloat while preserving relevant context

### Pattern Recognition Improvement
- GPT identifies similar market conditions from historical examples
- Adapts successful detection logic from previous cases
- Evolves thresholds and conditions based on performance data

## Technical Implementation

### Function History Management

```python
class GPTAnalyzer:
    def __init__(self, api_key: str):
        self.function_history_file = "function_history.json"
        self.max_history_size = 5
        self.function_history = self._load_function_history()
    
    def _add_to_function_history(self, symbol: str, date: str, 
                                function_code: str, pump_increase: float):
        # Add new function to beginning of list
        # Maintain maximum size limit
        # Save to persistent storage
```

### Context Formatting

```python
def _format_function_history_context(self) -> str:
    # Format historical functions for GPT consumption
    # Include pattern comparison guidelines
    # Provide implementation suggestions
```

### GPT Integration

```python
def generate_detector_function(self, pre_pump_data: Dict, 
                              pump_event: 'PumpEvent') -> str:
    # Load historical context
    history_context = self._format_function_history_context()
    
    # Combine with new analysis prompt
    full_prompt = history_context + prompt
    
    # Enhanced system prompt with historical awareness instructions
```

## Usage Examples

### Adding Function to History
```python
analyzer._add_to_function_history(
    'BTCUSDT',
    '20250618',
    detector_function_code,
    25.5  # pump increase percentage
)
```

### Context Preview
```
=== WCZEŚNIEJSZE FUNKCJE DETEKCJI ===
Poniżej znajdują się ostatnie wygenerowane funkcje detekcyjne...

# Funkcja 1: ETHUSDT (+18.5%) - 20250617
def detect_ETHUSDT_20250617_preconditions(df):
    """RSI oversold with volume confirmation pattern"""
    # Implementation details...

=== KONIEC HISTORII FUNKCJI ===

UWAGI DO NOWEJ FUNKCJI:
- Porównaj nowy przypadek z powyższymi wzorcami
- Unikaj powielania identycznej logiki
- Wykorzystaj najlepsze elementy z poprzednich funkcji
```

## Benefits

### Continuous Learning
- Each generated function improves the knowledge base
- Pattern recognition becomes more sophisticated over time
- Successful detection logic is preserved and evolved

### Reduced Logic Duplication
- GPT avoids creating identical detection patterns
- Encourages innovation and refinement of existing approaches
- Better adaptation to unique market conditions

### Performance Optimization
- Learns from pump performance data (percentage increases)
- Prioritizes patterns from high-performance historical cases
- Evolves detection thresholds based on real market outcomes

## Testing and Validation

### Test Suite Coverage
- Function history loading and persistence
- Context formatting and structure validation
- Integration with main pump analysis workflow
- Memory limit enforcement (5-function maximum)
- Historical function ordering (newest first)

### Production Validation
```bash
cd pump-analysis
python test_function_history.py
```

## Integration with Learning System

The Function History Context System works alongside the existing Learning System:

1. **Function History**: Provides context for GPT generation
2. **Learning System**: Manages function storage, testing, and evolution
3. **Combined Effect**: Creates comprehensive AI improvement mechanism

## Future Enhancements

### Pattern Classification
- Categorize functions by market conditions
- Provide specialized context based on similar scenarios
- Enhanced pattern matching algorithms

### Performance Metrics
- Track function success rates over time
- Weight historical context by performance data
- Automatic promotion of high-performing patterns

### Advanced Context Selection
- Smart selection of most relevant historical functions
- Market condition matching for context relevance
- Dynamic context sizing based on pattern complexity

## Configuration

### Environment Variables
- No additional configuration required
- Uses existing OPENAI_API_KEY for GPT access
- Function history file location: `pump-analysis/function_history.json`

### Customization Options
- `max_history_size`: Adjustable rolling window size (default: 5)
- `function_history_file`: Custom storage location
- Context formatting templates for different languages

## Monitoring and Maintenance

### Log Messages
- Function addition: `📚 Added function to history: detect_SYMBOL_DATE_preconditions`
- Context generation: Automatic during GPT function generation
- Error handling: Graceful degradation if history loading fails

### File Management
- Automatic cleanup of deprecated functions
- JSON format ensures human readability
- Backup and restore capabilities for critical function preservation

This system represents a significant advancement in AI-powered cryptocurrency market analysis, creating a self-improving detection mechanism that becomes more sophisticated with each pump analysis.