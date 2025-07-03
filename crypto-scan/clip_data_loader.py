#!/usr/bin/env python3
"""
CLIP Data Loader for Vision-AI Training
Ustandaryzowany loader danych treningowych z folderów charts/ dla modelu CLIP
"""

import os
import json
import glob
from typing import List, Dict, Tuple, Optional, Set
from pathlib import Path
import logging


class CLIPDataLoader:
    """
    Loader danych treningowych dla modelu CLIP z ustandaryzowaną strukturą folderów
    
    Struktura danych:
    training_data/charts/
    ├── SYMBOL_EXCHANGE_setup_score-XXX_TIMESTAMP.png
    ├── SYMBOL_EXCHANGE_setup_score-XXX_TIMESTAMP_metadata.json
    └── (opcjonalnie) SYMBOL_EXCHANGE_setup_score-XXX_TIMESTAMP_label.txt
    """
    
    def __init__(self, 
                 charts_dir: str = "training_data/charts",
                 exclude_invalid: bool = True,
                 exclude_failed: bool = True,
                 min_tjde_score: float = 0.0):
        """
        Initialize CLIP data loader
        
        Args:
            charts_dir: Folder z wykresami i metadanymi
            exclude_invalid: Wyklucz invalid_symbol i corrupted data
            exclude_failed: Wyklucz extraction_failed przypadki
            min_tjde_score: Minimalny TJDE score do filtrowania
        """
        self.charts_dir = Path(charts_dir)
        self.exclude_invalid = exclude_invalid
        self.exclude_failed = exclude_failed
        self.min_tjde_score = min_tjde_score
        
        # Statistics tracking
        self.stats = {
            'total_files': 0,
            'valid_pairs': 0,
            'missing_metadata': 0,
            'missing_charts': 0,
            'invalid_symbols': 0,
            'extraction_failed': 0,
            'low_tjde_score': 0,
            'corrupted_data': 0,
            'final_dataset_size': 0
        }
        
        self.logger = logging.getLogger(__name__)
        
    def load_training_data(self) -> List[Dict]:
        """
        Wczytaj wszystkie prawidłowe dane treningowe z folderu charts/
        
        Returns:
            Lista słowników z danymi treningowymi:
            {
                'image_path': str,
                'metadata_path': str, 
                'label_text': str,
                'setup_label': str,
                'tjde_score': float,
                'symbol': str,
                'exchange': str,
                'timestamp': str
            }
        """
        if not self.charts_dir.exists():
            self.logger.error(f"Charts directory not found: {self.charts_dir}")
            return []
        
        print(f"🔍 [CLIP LOADER] Scanning training data in: {self.charts_dir}")
        
        # Znajdź wszystkie pliki PNG (wykresy)
        chart_files = list(self.charts_dir.glob("*.png"))
        self.stats['total_files'] = len(chart_files)
        
        print(f"📊 Found {len(chart_files)} chart files")
        
        training_data = []
        
        for chart_path in chart_files:
            # Pomiń placeholdery i failed charts
            if self._is_placeholder_file(chart_path):
                continue
                
            # Znajdź odpowiadający plik metadata
            metadata_path = self._find_metadata_file(chart_path)
            
            if not metadata_path:
                self.stats['missing_metadata'] += 1
                continue
            
            # Wczytaj i waliduj metadane
            metadata = self._load_and_validate_metadata(metadata_path)
            
            if not metadata:
                continue
                
            # Sprawdź kryteria filtrowania
            if not self._passes_filters(metadata):
                continue
            
            # Wyciągnij label text z GPT analysis
            label_text = self._extract_label_text(metadata)
            
            if not label_text:
                self.stats['extraction_failed'] += 1
                if self.exclude_failed:
                    continue
                label_text = "unknown_pattern"
            
            # Dodaj do datasetu
            training_item = {
                'image_path': str(chart_path),
                'metadata_path': str(metadata_path),
                'label_text': label_text,
                'setup_label': metadata.get('setup_label', 'unknown'),
                'tjde_score': metadata.get('tjde_score', 0.0),
                'symbol': metadata.get('symbol', 'UNKNOWN'),
                'exchange': metadata.get('exchange', 'UNKNOWN'),
                'timestamp': metadata.get('timestamp', ''),
                'gpt_analysis': metadata.get('gpt_analysis', ''),
                'market_phase': metadata.get('market_phase', 'unknown')
            }
            
            training_data.append(training_item)
            self.stats['valid_pairs'] += 1
        
        self.stats['final_dataset_size'] = len(training_data)
        
        # Posortuj po TJDE score (najlepsze pierwsze)
        training_data.sort(key=lambda x: x['tjde_score'], reverse=True)
        
        self._print_statistics()
        
        return training_data
    
    def _find_metadata_file(self, chart_path: Path) -> Optional[Path]:
        """Znajdź odpowiadający plik metadata dla wykresu"""
        # Spróbuj różne formaty nazw metadanych
        base_name = chart_path.stem
        
        possible_metadata_names = [
            f"{base_name}_metadata.json",
            f"{base_name}.json",
            f"{base_name}_meta.json"
        ]
        
        for meta_name in possible_metadata_names:
            metadata_path = chart_path.parent / meta_name
            if metadata_path.exists():
                return metadata_path
        
        return None
    
    def _load_and_validate_metadata(self, metadata_path: Path) -> Optional[Dict]:
        """Wczytaj i waliduj plik metadanych"""
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            # Podstawowa walidacja struktury
            if not isinstance(metadata, dict):
                self.stats['corrupted_data'] += 1
                return None
                
            return metadata
            
        except (json.JSONDecodeError, FileNotFoundError, UnicodeDecodeError) as e:
            self.logger.warning(f"Failed to load metadata {metadata_path}: {e}")
            self.stats['corrupted_data'] += 1
            return None
    
    def _passes_filters(self, metadata: Dict) -> bool:
        """Sprawdź czy dane przechodzą filtry jakości"""
        
        # Filter 1: Invalid symbols
        if self.exclude_invalid:
            if metadata.get('invalid_symbol', False):
                self.stats['invalid_symbols'] += 1
                return False
            
            if metadata.get('status') == 'invalid_symbol':
                self.stats['invalid_symbols'] += 1
                return False
        
        # Filter 2: TJDE score threshold
        tjde_score = metadata.get('tjde_score', 0.0)
        if tjde_score < self.min_tjde_score:
            self.stats['low_tjde_score'] += 1
            return False
        
        # Filter 3: Blocked from training
        if metadata.get('blocked_from_training', False):
            self.stats['invalid_symbols'] += 1
            return False
        
        # Filter 4: Exclude low-quality setup labels (Safety Cap protection)
        setup_label = metadata.get('setup_label', '').strip()
        invalid_setups = ['setup_analysis', 'unknown', 'no_clear_pattern', 'unable_analyze', 'sorry_analyze']
        
        if self.exclude_invalid and setup_label in invalid_setups:
            self.stats['invalid_symbols'] += 1
            return False
        
        # Filter 5: Require authentic data for training
        if not metadata.get('authentic_data', True):
            self.stats['invalid_symbols'] += 1
            return False
        
        return True
    
    def _extract_label_text(self, metadata: Dict) -> Optional[str]:
        """Wyciągnij tekst labela z metadanych GPT"""
        
        # Priorytet 1: setup_label (najlepsze źródło)
        setup_label = metadata.get('setup_label', '').strip()
        if setup_label and setup_label not in ['unknown', 'setup_analysis', 'no_clear_pattern']:
            return setup_label
        
        # Priorytet 2: GPT analysis commentary
        gpt_analysis = metadata.get('gpt_analysis', '').strip()
        if gpt_analysis:
            # Wyciągnij pierwszy znaczący opis z GPT commentary
            lines = gpt_analysis.split('\n')
            for line in lines:
                line = line.strip()
                if len(line) > 10 and not line.startswith('[') and not line.startswith('Score:'):
                    # Usuń znaki specjalne i skróć do rozumnej długości
                    clean_line = self._clean_text_for_clip(line)
                    if clean_line:
                        return clean_line
        
        # Priorytet 3: Market phase jako fallback
        market_phase = metadata.get('market_phase', '').strip()
        if market_phase and market_phase != 'unknown':
            return f"{market_phase}_pattern"
        
        return None
    
    def _clean_text_for_clip(self, text: str) -> str:
        """Oczyść tekst dla CLIP training"""
        # Usuń znaki specjalne i skróć
        clean = text.replace('*', '').replace('#', '').replace('**', '')
        clean = ' '.join(clean.split())  # Usuń nadmiarowe spacje
        
        # Skróć do maksymalnie 50 znaków dla CLIP
        if len(clean) > 50:
            clean = clean[:47] + "..."
        
        return clean.lower().strip()
    
    def _is_placeholder_file(self, file_path: Path) -> bool:
        """Sprawdź czy plik to placeholder/failed chart"""
        filename = file_path.name.lower()
        
        placeholder_indicators = [
            'tradingview_failed',
            'placeholder',
            'failed',
            'error',
            'invalid'
        ]
        
        return any(indicator in filename for indicator in placeholder_indicators)
    
    def _print_statistics(self):
        """Wyświetl statystyki wczytanych danych"""
        print(f"\n📊 [CLIP LOADER] Training Data Statistics:")
        print(f"   📁 Total chart files found: {self.stats['total_files']}")
        print(f"   ✅ Valid chart-metadata pairs: {self.stats['valid_pairs']}")
        print(f"   🗂️ Final dataset size: {self.stats['final_dataset_size']}")
        
        if self.stats['missing_metadata'] > 0:
            print(f"   ⚠️ Missing metadata files: {self.stats['missing_metadata']}")
        
        if self.stats['invalid_symbols'] > 0:
            print(f"   🚫 Invalid symbols filtered: {self.stats['invalid_symbols']}")
        
        if self.stats['extraction_failed'] > 0:
            print(f"   ❌ Label extraction failed: {self.stats['extraction_failed']}")
        
        if self.stats['low_tjde_score'] > 0:
            print(f"   📉 Low TJDE score filtered: {self.stats['low_tjde_score']}")
        
        if self.stats['corrupted_data'] > 0:
            print(f"   💥 Corrupted data files: {self.stats['corrupted_data']}")
        
        # Quality ratio
        if self.stats['total_files'] > 0:
            quality_ratio = (self.stats['final_dataset_size'] / self.stats['total_files']) * 100
            print(f"   🎯 Data quality ratio: {quality_ratio:.1f}%")
    
    def get_statistics(self) -> Dict:
        """Zwróć statystyki jako słownik"""
        return self.stats.copy()
    
    def filter_by_setup_types(self, training_data: List[Dict], 
                             setup_types: List[str]) -> List[Dict]:
        """
        Filtruj dane treningowe po typach setupów
        
        Args:
            training_data: Lista danych treningowych
            setup_types: Lista dozwolonych typów setupów
            
        Returns:
            Przefiltrowane dane
        """
        filtered = [
            item for item in training_data
            if item['setup_label'] in setup_types
        ]
        
        print(f"🔍 [FILTER] Filtered {len(filtered)} items from {len(training_data)} by setup types: {setup_types}")
        
        return filtered
    
    def get_setup_distribution(self, training_data: List[Dict]) -> Dict[str, int]:
        """Zwróć rozkład typów setupów w datasecie"""
        distribution = {}
        
        for item in training_data:
            setup = item['setup_label']
            distribution[setup] = distribution.get(setup, 0) + 1
        
        return dict(sorted(distribution.items(), key=lambda x: x[1], reverse=True))
    
    def save_dataset_summary(self, training_data: List[Dict], 
                           output_path: str = "training_data/dataset_summary.json"):
        """Zapisz podsumowanie datasetu do pliku"""
        summary = {
            'dataset_size': len(training_data),
            'statistics': self.get_statistics(),
            'setup_distribution': self.get_setup_distribution(training_data),
            'top_performers': [
                {
                    'symbol': item['symbol'],
                    'setup_label': item['setup_label'],
                    'tjde_score': item['tjde_score'],
                    'image_path': os.path.basename(item['image_path'])
                }
                for item in training_data[:10]  # Top 10
            ],
            'timestamp': str(Path().cwd() / 'timestamp')
        }
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            print(f"💾 [CLIP LOADER] Dataset summary saved to: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save dataset summary: {e}")


def main():
    """Test the CLIP data loader"""
    print("🚀 Testing CLIP Data Loader")
    print("=" * 50)
    
    # Initialize loader with quality filters
    loader = CLIPDataLoader(
        charts_dir="training_data/charts",
        exclude_invalid=True,
        exclude_failed=True,
        min_tjde_score=0.25  # Minimum quality threshold
    )
    
    # Load training data
    training_data = loader.load_training_data()
    
    if not training_data:
        print("❌ No valid training data found!")
        return
    
    print(f"\n🎯 CLIP Dataset Ready: {len(training_data)} samples")
    
    # Show setup distribution
    distribution = loader.get_setup_distribution(training_data)
    print(f"\n📊 Setup Distribution:")
    for setup, count in distribution.items():
        print(f"   • {setup}: {count} samples")
    
    # Show top samples
    print(f"\n🔥 Top 5 Samples by TJDE Score:")
    for i, item in enumerate(training_data[:5], 1):
        print(f"   {i}. {item['symbol']} ({item['setup_label']}) - TJDE {item['tjde_score']:.3f}")
        print(f"      Label: {item['label_text'][:50]}...")
        print(f"      Image: {os.path.basename(item['image_path'])}")
    
    # Save summary
    loader.save_dataset_summary(training_data)
    
    print(f"\n✅ CLIP Data Loader test completed successfully!")
    print(f"   Dataset is ready for Vision-AI training with {len(training_data)} quality samples")


if __name__ == "__main__":
    main()