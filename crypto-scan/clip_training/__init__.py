"""
CLIP Training Module for Real-time Chart Learning
Moduł trenowania CLIP do uczenia się na wykresach w czasie rzeczywistym
"""

from .clip_model import CLIPChartModel, get_clip_model
from .dataset_loader import CLIPDatasetLoader
from .clip_trainer import CLIPOnlineTrainer, get_clip_trainer, train_clip_on_new_data
from .clip_predictor import CLIPChartPredictor, get_clip_predictor, predict_clip_labels

__all__ = [
    'CLIPChartModel',
    'CLIPDatasetLoader', 
    'CLIPOnlineTrainer',
    'CLIPChartPredictor',
    'get_clip_model',
    'get_clip_trainer',
    'get_clip_predictor',
    'train_clip_on_new_data',
    'predict_clip_labels'
]