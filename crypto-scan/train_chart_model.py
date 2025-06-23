#!/usr/bin/env python3
"""
Chart Pattern Computer Vision Model Training
Trains PyTorch model on labeled chart images for pattern recognition
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from pathlib import Path
from PIL import Image
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class ChartDataset(Dataset):
    """Dataset class for chart pattern images"""
    
    def __init__(self, charts_dir: str, transform=None, train_mode=True):
        self.charts_dir = Path(charts_dir)
        self.transform = transform
        self.train_mode = train_mode
        
        # Label mapping
        self.label_map = {
            'breakout_continuation': 0,
            'pullback_setup': 1,
            'range': 2,
            'fakeout': 3,
            'exhaustion': 4,
            'retest_confirmation': 5
        }
        self.id_to_label = {v: k for k, v in self.label_map.items()}
        
        # Load labeled images
        self.samples = self._load_samples()
        
        print(f"[DATASET] Loaded {len(self.samples)} samples")
        self._print_class_distribution()
    
    def _load_samples(self) -> List[Tuple[str, int]]:
        """Load image paths and labels from filenames"""
        samples = []
        
        # Find all PNG files with labels in filename
        for chart_file in self.charts_dir.glob("*.png"):
            label = self._extract_label_from_filename(chart_file.name)
            if label is not None:
                samples.append((str(chart_file), label))
        
        return samples
    
    def _extract_label_from_filename(self, filename: str) -> Optional[int]:
        """Extract label from filename pattern: SYMBOL_TIMESTAMP_LABEL.png"""
        for label_name, label_id in self.label_map.items():
            if label_name in filename:
                return label_id
        return None
    
    def _print_class_distribution(self):
        """Print distribution of classes in dataset"""
        class_counts = {}
        for _, label in self.samples:
            label_name = self.id_to_label[label]
            class_counts[label_name] = class_counts.get(label_name, 0) + 1
        
        print("[DATASET] Class distribution:")
        for label_name, count in class_counts.items():
            print(f"  {label_name}: {count}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label


class ChartPatternCNN(nn.Module):
    """Custom CNN for chart pattern recognition"""
    
    def __init__(self, num_classes=6, pretrained=True):
        super(ChartPatternCNN, self).__init__()
        
        # Use ResNet18 as backbone
        self.backbone = models.resnet18(pretrained=pretrained)
        
        # Replace classifier
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)


class ChartModelTrainer:
    """Chart pattern model trainer"""
    
    def __init__(self, charts_dir: str = "data/chart_training/charts"):
        self.charts_dir = charts_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.train_loader = None
        self.val_loader = None
        
        # Model save path
        self.model_dir = Path("data/chart_training/models")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = self.model_dir / "chart_pattern_model.pth"
        
        print(f"[TRAINER] Using device: {self.device}")
    
    def prepare_data(self, train_split=0.8, batch_size=16):
        """Prepare data loaders for training"""
        
        # Data transforms
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(0.3),
            transforms.RandomRotation(5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Create full dataset
        full_dataset = ChartDataset(self.charts_dir, transform=train_transform)
        
        if len(full_dataset) == 0:
            raise ValueError("No labeled charts found. Run label_charts_with_gpt.py first.")
        
        # Split dataset
        train_size = int(train_split * len(full_dataset))
        val_size = len(full_dataset) - train_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size]
        )
        
        # Apply validation transforms to validation set
        val_dataset.dataset.transform = val_transform
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
        )
        self.val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )
        
        print(f"[DATA] Train samples: {len(train_dataset)}")
        print(f"[DATA] Validation samples: {len(val_dataset)}")
        
        return full_dataset.label_map
    
    def create_model(self, num_classes=6):
        """Create and initialize model"""
        self.model = ChartPatternCNN(num_classes=num_classes, pretrained=True)
        self.model.to(self.device)
        
        print(f"[MODEL] Created ChartPatternCNN with {num_classes} classes")
        return self.model
    
    def train_model(self, num_epochs=20, learning_rate=0.001):
        """Train the chart pattern model"""
        
        if self.model is None:
            raise ValueError("Model not created. Call create_model() first.")
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        
        # Training history
        train_losses = []
        val_accuracies = []
        best_val_acc = 0.0
        
        print(f"[TRAIN] Starting training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            # Training phase
            self.model.train()
            running_loss = 0.0
            
            for batch_idx, (images, labels) in enumerate(self.train_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
            
            # Validation phase
            val_acc = self._validate_model()
            
            # Update learning rate
            scheduler.step()
            
            # Record metrics
            epoch_loss = running_loss / len(self.train_loader)
            train_losses.append(epoch_loss)
            val_accuracies.append(val_acc)
            
            print(f"[EPOCH {epoch+1:2d}/{num_epochs}] "
                  f"Loss: {epoch_loss:.4f} | Val Acc: {val_acc:.3f}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self._save_model()
                print(f"[SAVE] New best model saved (acc: {val_acc:.3f})")
        
        print(f"[TRAIN] Training completed. Best validation accuracy: {best_val_acc:.3f}")
        
        # Plot training history
        self._plot_training_history(train_losses, val_accuracies)
        
        return best_val_acc
    
    def _validate_model(self) -> float:
        """Validate model and return accuracy"""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        return correct / total
    
    def _save_model(self):
        """Save model state"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_class': 'ChartPatternCNN',
            'num_classes': 6,
            'label_map': {
                'breakout_continuation': 0,
                'pullback_setup': 1,
                'range': 2,
                'fakeout': 3,
                'exhaustion': 4,
                'retest_confirmation': 5
            }
        }, self.model_path)
    
    def _plot_training_history(self, train_losses: List, val_accuracies: List):
        """Plot training history"""
        try:
            epochs = range(1, len(train_losses) + 1)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            # Training loss
            ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
            ax1.set_title('Training Loss')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend()
            
            # Validation accuracy
            ax2.plot(epochs, val_accuracies, 'r-', label='Validation Accuracy')
            ax2.set_title('Validation Accuracy')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy')
            ax2.legend()
            
            plt.tight_layout()
            plot_path = self.model_dir / "training_history.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"[PLOT] Training history saved: {plot_path}")
            
        except Exception as e:
            print(f"[PLOT] Failed to save training plot: {e}")
    
    def evaluate_model(self, label_map: Dict) -> Dict:
        """Comprehensive model evaluation"""
        
        if self.model is None:
            print("[EVAL] Loading saved model for evaluation...")
            self.load_model()
        
        self.model.eval()
        
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Classification report
        id_to_label = {v: k for k, v in label_map.items()}
        target_names = [id_to_label[i] for i in range(len(label_map))]
        
        report = classification_report(
            all_labels, all_predictions, 
            target_names=target_names, 
            output_dict=True
        )
        
        print("[EVAL] Classification Report:")
        print(classification_report(all_labels, all_predictions, target_names=target_names))
        
        return report
    
    def load_model(self) -> bool:
        """Load saved model"""
        try:
            if not self.model_path.exists():
                print(f"[LOAD] Model file not found: {self.model_path}")
                return False
            
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            self.model = ChartPatternCNN(num_classes=checkpoint['num_classes'])
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            print(f"[LOAD] Model loaded successfully from {self.model_path}")
            return True
            
        except Exception as e:
            print(f"[LOAD] Failed to load model: {e}")
            return False


def main():
    """Main function for model training"""
    print("🧠 Chart Pattern Computer Vision Model Training")
    print("=" * 50)
    
    # Check for labeled data
    charts_dir = "data/chart_training/charts"
    if not Path(charts_dir).exists():
        print("❌ Charts directory not found. Run generate_chart_snapshot.py first.")
        return
    
    # Initialize trainer
    trainer = ChartModelTrainer(charts_dir)
    
    try:
        # Prepare data
        label_map = trainer.prepare_data(batch_size=8)  # Small batch for limited data
        
        # Create model
        trainer.create_model(num_classes=len(label_map))
        
        # Train model
        best_acc = trainer.train_model(num_epochs=15, learning_rate=0.001)
        
        # Evaluate model
        evaluation = trainer.evaluate_model(label_map)
        
        print(f"\n📊 Training Summary:")
        print(f"  Best Validation Accuracy: {best_acc:.3f}")
        print(f"  Model Saved: {trainer.model_path}")
        print(f"  Dataset Size: {len(trainer.train_loader.dataset) + len(trainer.val_loader.dataset)}")
        
        print(f"\n✅ Model training completed!")
        print(f"📁 Model files: {trainer.model_dir}")
        
    except ValueError as e:
        print(f"❌ Training failed: {e}")
        print("Run label_charts_with_gpt.py to create labeled training data.")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")


if __name__ == "__main__":
    main()