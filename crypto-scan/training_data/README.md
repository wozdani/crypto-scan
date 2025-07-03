# Vision-AI Training Data Structure

This directory contains the standardized training data for CLIP model development in the crypto trend detection system.

## ✅ STANDARDIZED STRUCTURE (July 2025)

```
training_data/
├── charts/                    # 🎯 UNIFIED: All training data here
│   ├── SYMBOL_EXCHANGE_setup_score-XXX_TIMESTAMP.png      # Chart images
│   ├── SYMBOL_EXCHANGE_setup_score-XXX_TIMESTAMP_metadata.json  # Metadata
│   └── (optional) SYMBOL_EXCHANGE_setup_score-XXX_TIMESTAMP_label.txt
├── failed_charts/             # 🚫 Invalid symbols and failed generations
└── dataset_summary.json       # 📊 Dataset quality statistics
```

## 🔥 KEY IMPROVEMENTS

### 1. UNIFIED FOLDER STRUCTURE
- **Before**: Separate folders `charts/`, `metadata/`, `labels/` (confusing)
- **After**: Everything in `charts/` with paired files (clean)

### 2. CLIP DATA LOADER
- Intelligent filtering: excludes `invalid_symbol`, `setup_analysis`, `unknown`
- Quality validation: minimum TJDE scores, authentic data only
- Label extraction: from GPT analysis and setup_label fields

### 3. DATA QUALITY PROTECTION
- TOP 5 TJDE token selection only
- Invalid symbol detection and blocking
- Safety Cap integration (score ≤0.25 for invalid setups)

## 📊 USAGE EXAMPLES

### Load Training Data for CLIP
```python
from clip_data_loader import CLIPDataLoader

loader = CLIPDataLoader(
    charts_dir="training_data/charts",
    exclude_invalid=True,
    min_tjde_score=0.25
)

training_data = loader.load_training_data()
print(f"Dataset ready: {len(training_data)} quality samples")
```

### Filter by Setup Types
```python
momentum_data = loader.filter_by_setup_types(
    training_data, 
    ['momentum_follow', 'trend_continuation', 'breakout_pattern']
)
```

## 🛡️ DATA QUALITY METRICS

- **Total files**: ~2060 chart files
- **Quality dataset**: ~221 validated samples
- **Quality ratio**: 10.7% (high standards)
- **Top setups**: trend_continuation (86), momentum_follow (48), breakout_pattern (38)

## 🚫 DEPRECATED (Removed July 2025)

- ~~`metadata/` folder~~ → Moved to `charts/` as paired files
- ~~`labels/` folder~~ → Integrated into metadata JSON
- ~~`clip/` folder~~ → Unused legacy structure
- ~~`labels.jsonl`~~ → Replaced by individual metadata files

## 🎯 VISION-AI READY

The standardized structure enables:
- ✅ Clean CLIP model training with quality datasets
- ✅ Automatic invalid symbol filtering
- ✅ GPT analysis integration for semantic labeling
- ✅ TOP 5 TJDE token selection for premium data quality
- ✅ Safety Cap protection against false high scores