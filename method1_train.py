import warnings
warnings.filterwarnings("ignore")

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import StratifiedKFold
import argparse
import librosa
import torchvision.models as models
from transformers import AutoConfig, AutoModel
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import torchmetrics
from collections import Counter

GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop', 
          'jazz', 'metal', 'pop', 'reggae', 'rock']
GENRE_TO_IDX = {genre: idx for idx, genre in enumerate(GENRES)}

class Method1Dataset(Dataset):
    def __init__(self, data_paths, feature_type='mel', model_type='convnext'):
        self.segments = []
        self.labels = []
        
        for base_name, genre in data_paths:
            genre_idx = GENRE_TO_IDX[genre]
            genre_dir = os.path.join('gtzan_preprocessed', 'method1_30sec', genre)
            
            if not os.path.exists(genre_dir):
                continue
                
            if model_type == 'mert':
                audio_path = os.path.join(genre_dir, f"{base_name}.wav")
                if os.path.exists(audio_path):
                    try:
                        y, sr = librosa.load(audio_path, sr=22050)
                        target_samples = 22050 * 15
                        y = self._pad_or_crop_audio(y, target_samples)
                        self.segments.append(torch.FloatTensor(y))
                        self.labels.append(genre_idx)
                    except:
                        continue
            else:
                feature_path = os.path.join(genre_dir, f"{base_name}_{feature_type}.npy")
                if os.path.exists(feature_path):
                    try:
                        features = np.load(feature_path)
                        self.segments.append(torch.FloatTensor(features).unsqueeze(0))
                        self.labels.append(genre_idx)
                    except:
                        continue

    def _pad_or_crop_audio(self, audio, target_samples):
        if len(audio) < target_samples:
            audio = np.pad(audio, (0, target_samples - len(audio)), mode='constant')
        elif len(audio) > target_samples:
            start_idx = (len(audio) - target_samples) // 2
            audio = audio[start_idx:start_idx + target_samples]
        return audio
        
    def __len__(self):
        return len(self.segments)
    
    def __getitem__(self, idx):
        return self.segments[idx], self.labels[idx]

class MERT(nn.Module):
    def __init__(self, freeze_feature_extractor=True):
        super(MERT, self).__init__()
        config = AutoConfig.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True)
        if not hasattr(config, "conv_pos_batch_norm"):
            setattr(config, "conv_pos_batch_norm", False)
        self.mert = AutoModel.from_pretrained("m-a-p/MERT-v1-95M", config=config, trust_remote_code=True)
        if freeze_feature_extractor:
            self.freeze()

    def forward(self, input_values):
        outputs = self.mert(input_values, output_hidden_states=False)
        return outputs.last_hidden_state

    def freeze(self):
        for param in self.mert.parameters():
            param.requires_grad = False

class MERTClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.mert = MERT(freeze_feature_extractor=True)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        mert_output = self.mert(x)
        features = mert_output.mean(dim=1)
        return self.classifier(features)

class ConvNeXtClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.channel_adapter = nn.Conv2d(1, 3, kernel_size=1)
        
        try:
            self.backbone = models.convnext_tiny(weights=None)
        except TypeError:
            self.backbone = models.convnext_tiny(pretrained=False)
        
        num_features = self.backbone.classifier[2].in_features
        
        self.backbone.classifier = nn.Sequential(
            nn.Flatten(1),
            nn.LayerNorm(num_features),
            nn.Dropout(0.2),
            nn.Linear(num_features, num_classes)
        )
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        nn.init.kaiming_normal_(self.channel_adapter.weight, mode='fan_out', nonlinearity='relu')
        if self.channel_adapter.bias is not None:
            nn.init.constant_(self.channel_adapter.bias, 0)
        
        for m in self.backbone.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(0)
        x = self.channel_adapter(x)
        return self.backbone(x)

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(0)
        features = self.features(x)
        features = features.view(features.size(0), -1)
        return self.classifier(features)

class Method1LightningModule(pl.LightningModule):
    def __init__(self, model_type='convnext', num_classes=10, learning_rate=0.001):
        super().__init__()
        self.save_hyperparameters()
        
        if model_type == 'convnext':
            self.model = ConvNeXtClassifier(num_classes)
        elif model_type == 'simple_cnn':
            self.model = SimpleCNN(num_classes)
        elif model_type == 'mert':
            self.model = MERTClassifier(num_classes)
        
        self.criterion = nn.CrossEntropyLoss()
        
        # Metrics
        self.train_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_f1 = torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average='macro')
        self.val_precision = torchmetrics.Precision(task="multiclass", num_classes=num_classes, average='macro')
        self.val_recall = torchmetrics.Recall(task="multiclass", num_classes=num_classes, average='macro')
        
        # Store best metrics
        self.best_val_metrics = {
            'accuracy': 0.0,
            'f1': 0.0,
            'precision': 0.0,
            'recall': 0.0
        }
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        data, target = batch
        outputs = self.model(data)
        loss = self.criterion(outputs, target)
        preds = torch.argmax(outputs, dim=1)
        
        self.train_accuracy(preds, target)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', self.train_accuracy, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        data, target = batch
        outputs = self.model(data)
        loss = self.criterion(outputs, target)
        preds = torch.argmax(outputs, dim=1)
        
        self.val_accuracy(preds, target)
        self.val_f1(preds, target)
        self.val_precision(preds, target)
        self.val_recall(preds, target)
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', self.val_accuracy, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_f1', self.val_f1, on_step=False, on_epoch=True)
        self.log('val_precision', self.val_precision, on_step=False, on_epoch=True)
        self.log('val_recall', self.val_recall, on_step=False, on_epoch=True)
        return loss
    
    def on_validation_epoch_end(self):
        # Get current metrics
        current_acc = self.val_accuracy.compute()
        current_f1 = self.val_f1.compute()
        current_precision = self.val_precision.compute()
        current_recall = self.val_recall.compute()
        
        # Update best metrics if current accuracy is better
        if current_acc > self.best_val_metrics['accuracy']:
            self.best_val_metrics['accuracy'] = current_acc.item()
            self.best_val_metrics['f1'] = current_f1.item()
            self.best_val_metrics['precision'] = current_precision.item()
            self.best_val_metrics['recall'] = current_recall.item()
    
    def configure_optimizers(self):
        if self.hparams.model_type == 'convnext':
            lr = self.hparams.learning_rate * 1
        else:
            lr = self.hparams.learning_rate
            
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-6
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_acc",
                "frequency": 1
            }
        }

def get_data_paths():
    data_paths = []
    method_dir = os.path.join('gtzan_preprocessed', 'method1_30sec')
    
    if not os.path.exists(method_dir):
        return data_paths
    
    for genre in GENRES:
        genre_dir = os.path.join(method_dir, genre)
        if os.path.exists(genre_dir):
            track_names = set()
            for file in os.listdir(genre_dir):
                if file.endswith('.wav'):
                    base_name = file.replace('.wav', '')
                    track_names.add(base_name)
                elif file.endswith('_mel.npy') or file.endswith('_cqt.npy'):
                    base_name = file.replace('_mel.npy', '').replace('_cqt.npy', '')
                    track_names.add(base_name)
            
            for base_name in track_names:
                data_paths.append((base_name, genre))
    
    return data_paths

def train_single_configuration(feature, model, data_paths, epochs=50, batch_size=32, 
                             folds=5, learning_rate=0.001, num_workers=4):
    """Train a single configuration and return results."""
    file_paths = [path for path, _ in data_paths]
    labels = [GENRE_TO_IDX[genre] for _, genre in data_paths]
    
    label_counts = Counter(labels)
    min_samples_per_class = min(label_counts.values())
    
    if min_samples_per_class < folds:
        print(f"Warning: Minimum samples per class ({min_samples_per_class}) is less than folds ({folds})")
        folds = min(folds, min_samples_per_class)
        print(f"Adjusting folds to {folds}")
    
    if folds < 2:
        print("Error: Not enough data for cross-validation")
        return None
    
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(file_paths, labels)):
        print(f"\nFold {fold + 1}/{folds}")
        
        train_paths = [data_paths[i] for i in train_idx]
        val_paths = [data_paths[i] for i in val_idx]
        
        train_dataset = Method1Dataset(train_paths, feature, model)
        val_dataset = Method1Dataset(val_paths, feature, model)
        
        if len(train_dataset) == 0 or len(val_dataset) == 0:
            print(f"Skipping fold {fold + 1} due to empty dataset")
            continue
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                                num_workers=num_workers, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                              num_workers=num_workers, pin_memory=True)
        
        model_module = Method1LightningModule(
            model_type=model,
            num_classes=len(GENRES),
            learning_rate=learning_rate
        )
        
        checkpoint_callback = ModelCheckpoint(
            dirpath=f'checkpoints/method1_{feature}_{model}_fold_{fold+1}',
            filename='{epoch}-{val_acc:.2f}',
            monitor='val_acc',
            mode='max',
            save_top_k=1
        )
        
        early_stopping = EarlyStopping(monitor='val_acc', mode='max', patience=50)
        logger = TensorBoardLogger('logs', name=f'method1_{feature}_{model}', version=f'fold_{fold+1}')
        
        trainer = pl.Trainer(
            max_epochs=epochs,
            callbacks=[checkpoint_callback, early_stopping],
            logger=logger,
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            devices=1,
            precision='16-mixed',
            gradient_clip_val=1.0,
            accumulate_grad_batches=2 if model == 'mert' else 1,
            enable_progress_bar=True,
            deterministic=True,
            log_every_n_steps=1
        )
        
        trainer.fit(model_module, train_loader, val_loader)
        
        # Get best metrics stored during validation
        fold_result = {
            'accuracy': model_module.best_val_metrics['accuracy'] * 100,
            'f1': model_module.best_val_metrics['f1'] * 100,
            'precision': model_module.best_val_metrics['precision'] * 100,
            'recall': model_module.best_val_metrics['recall'] * 100
        }
        
        fold_results.append(fold_result)
        
        print(f'Fold {fold + 1} Best Results:')
        print(f'  Accuracy: {fold_result["accuracy"]:.2f}%')
        print(f'  F1 Score: {fold_result["f1"]:.2f}%')
        print(f'  Precision: {fold_result["precision"]:.2f}%')
        print(f'  Recall: {fold_result["recall"]:.2f}%')
        
        del model_module, trainer, train_loader, val_loader, train_dataset, val_dataset
        torch.cuda.empty_cache()
    
    if fold_results:
        return {
            'accuracy': np.mean([r['accuracy'] for r in fold_results]),
            'f1': np.mean([r['f1'] for r in fold_results]),
            'precision': np.mean([r['precision'] for r in fold_results]),
            'recall': np.mean([r['recall'] for r in fold_results])
        }
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature', choices=['mel', 'cqt', 'audio'], default='mel')
    parser.add_argument('--model', choices=['convnext', 'simple_cnn', 'mert'], default='convnext')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--num_workers', type=int, default=4)
    
    args = parser.parse_args()
    
    if args.model == 'mert':
        args.batch_size = min(args.batch_size, 4)
    
    torch.cuda.empty_cache()
    torch.set_float32_matmul_precision('medium')
    
    data_paths = get_data_paths()
    if not data_paths:
        print("No data found for method1_30sec")
        return
    
    print(f"\n{'='*60}")
    print(f"Training {args.model} with {args.feature} features")
    print(f"{'='*60}")
    
    result = train_single_configuration(
        feature=args.feature,
        model=args.model,
        data_paths=data_paths,
        epochs=args.epochs,
        batch_size=args.batch_size,
        folds=args.folds,
        learning_rate=args.learning_rate,
        num_workers=args.num_workers
    )
    
    if result:
        print(f"\nFINAL RESULTS")
        print(f'Method: method1_30sec, Feature: {args.feature}, Model: {args.model}')
        print(f'Mean Accuracy: {result["accuracy"]:.2f}%')
        print(f'Mean F1 Score: {result["f1"]:.2f}%')
        print(f'Mean Precision: {result["precision"]:.2f}%')
        print(f'Mean Recall: {result["recall"]:.2f}%')
    else:
        print("\nTraining failed")

if __name__ == '__main__':
    main()