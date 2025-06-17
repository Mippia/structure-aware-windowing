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
import glob

GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop', 
          'jazz', 'metal', 'pop', 'reggae', 'rock']
GENRE_TO_IDX = {genre: idx for idx, genre in enumerate(GENRES)}

class GTZANSegmentDataset(Dataset):
    def __init__(self, data_paths, feature_type='mel', method='method2_5sec', model_type='convnext'):
        self.data_paths = data_paths
        self.feature_type = feature_type
        self.method = method
        self.model_type = model_type
        
        self.segments = []
        self.labels = []
        
        for base_name, genre in data_paths:
            genre_idx = GENRE_TO_IDX[genre]
            genre_dir = os.path.join('gtzan_preprocessed', method, genre)
            
            if method == 'method2_5sec':
                if model_type == 'mert':
                    target_samples = 22050 * 5
                    for i in range(6):
                        audio_path = os.path.join(genre_dir, f"{base_name}_seg{i:02d}.wav")
                        if os.path.exists(audio_path):
                            try:
                                y, sr = librosa.load(audio_path, sr=22050)
                                y = self._pad_or_crop_audio(y, target_samples)
                                self.segments.append(torch.FloatTensor(y))
                                self.labels.append(genre_idx)
                            except:
                                continue
                else:
                    for i in range(6):
                        if feature_type == 'mel':
                            seg_path = os.path.join(genre_dir, f"{base_name}_seg{i:02d}_mel.npy")
                        else:
                            seg_path = os.path.join(genre_dir, f"{base_name}_seg{i:02d}_cqt.npy")
                        
                        if os.path.exists(seg_path):
                            try:
                                features = np.load(seg_path)
                                self.segments.append(torch.FloatTensor(features).unsqueeze(0))
                                self.labels.append(genre_idx)
                            except:
                                continue

            elif method == 'method3_2bar':
                if model_type == 'mert':
                    target_samples = 22050 * 5
                    pattern = os.path.join(genre_dir, f"{base_name}_2bar*.wav")
                    seg_files = glob.glob(pattern)
                    seg_files.sort()
                    
                    for seg_file in seg_files:
                        try:
                            y, sr = librosa.load(seg_file, sr=22050)
                            y = self._pad_or_crop_audio(y, target_samples)
                            self.segments.append(torch.FloatTensor(y))
                            self.labels.append(genre_idx)
                        except:
                            continue
                else:
                    pattern = os.path.join(genre_dir, f"{base_name}_2bar*_{feature_type}.npy")
                    seg_files = glob.glob(pattern)
                    seg_files.sort()
                    
                    for seg_file in seg_files:
                        try:
                            features = np.load(seg_file)
                            features = self._pad_or_crop(features, 431)
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
    
    def _pad_or_crop(self, features, target_time_dim):
        current_time_dim = features.shape[-1]
        
        if current_time_dim < target_time_dim:
            pad_amount = target_time_dim - current_time_dim
            if features.ndim == 2:
                features = np.pad(features, ((0, 0), (0, pad_amount)), mode='constant')
            else:
                features = np.pad(features, ((0, 0), (0, 0), (0, pad_amount)), mode='constant')
        elif current_time_dim > target_time_dim:
            start_idx = (current_time_dim - target_time_dim) // 2
            if features.ndim == 2:
                features = features[:, start_idx:start_idx + target_time_dim]
            else:
                features = features[:, :, start_idx:start_idx + target_time_dim]
        
        return features
        
    def __len__(self):
        return len(self.segments)
    
    def __getitem__(self, idx):
        return self.segments[idx], self.labels[idx]

class GTZANTrackDataset(Dataset):
    def __init__(self, data_paths, feature_type='mel', method='method2_5sec', model_type='convnext'):
        self.data_paths = data_paths
        self.feature_type = feature_type
        self.method = method
        self.model_type = model_type
        
        self.tracks = []
        
        for base_name, genre in data_paths:
            genre_idx = GENRE_TO_IDX[genre]
            genre_dir = os.path.join('gtzan_preprocessed', method, genre)
            
            track_segments = []
            
            if method == 'method2_5sec':
                if model_type == 'mert':
                    target_samples = 22050 * 5
                    for i in range(6):
                        audio_path = os.path.join(genre_dir, f"{base_name}_seg{i:02d}.wav")
                        if os.path.exists(audio_path):
                            try:
                                y, sr = librosa.load(audio_path, sr=22050)
                                y = self._pad_or_crop_audio(y, target_samples)
                                track_segments.append(torch.FloatTensor(y))
                            except:
                                continue
                else:
                    for i in range(6):
                        if feature_type == 'mel':
                            seg_path = os.path.join(genre_dir, f"{base_name}_seg{i:02d}_mel.npy")
                        else:
                            seg_path = os.path.join(genre_dir, f"{base_name}_seg{i:02d}_cqt.npy")
                        
                        if os.path.exists(seg_path):
                            try:
                                features = np.load(seg_path)
                                track_segments.append(torch.FloatTensor(features).unsqueeze(0))
                            except:
                                continue
            
            elif method == 'method3_2bar':
                if model_type == 'mert':
                    target_samples = 22050 * 5
                    pattern = os.path.join(genre_dir, f"{base_name}_2bar*.wav")
                    seg_files = glob.glob(pattern)
                    seg_files.sort()
                    
                    for seg_file in seg_files:
                        try:
                            y, sr = librosa.load(seg_file, sr=22050)
                            y = self._pad_or_crop_audio(y, target_samples)
                            track_segments.append(torch.FloatTensor(y))
                        except:
                            continue
                else:
                    pattern = os.path.join(genre_dir, f"{base_name}_2bar*_{feature_type}.npy")
                    seg_files = glob.glob(pattern)
                    seg_files.sort()
                    
                    for seg_file in seg_files:
                        try:
                            features = np.load(seg_file)
                            features = self._pad_or_crop(features, 431)
                            track_segments.append(torch.FloatTensor(features).unsqueeze(0))
                        except:
                            continue
            
            if track_segments:
                self.tracks.append((track_segments, genre_idx))
    
    def _pad_or_crop_audio(self, audio, target_samples):
        if len(audio) < target_samples:
            audio = np.pad(audio, (0, target_samples - len(audio)), mode='constant')
        elif len(audio) > target_samples:
            start_idx = (len(audio) - target_samples) // 2
            audio = audio[start_idx:start_idx + target_samples]
        return audio
    
    def _pad_or_crop(self, features, target_time_dim):
        current_time_dim = features.shape[-1]
        
        if current_time_dim < target_time_dim:
            pad_amount = target_time_dim - current_time_dim
            if features.ndim == 2:
                features = np.pad(features, ((0, 0), (0, pad_amount)), mode='constant')
            else:
                features = np.pad(features, ((0, 0), (0, 0), (0, pad_amount)), mode='constant')
        elif current_time_dim > target_time_dim:
            start_idx = (current_time_dim - target_time_dim) // 2
            if features.ndim == 2:
                features = features[:, start_idx:start_idx + target_time_dim]
            else:
                features = features[:, :, start_idx:start_idx + target_time_dim]
        
        return features
        
    def __len__(self):
        return len(self.tracks)
    
    def __getitem__(self, idx):
        return self.tracks[idx]

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

class GTZANLightningModule(pl.LightningModule):
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
        
        self.train_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.train_f1 = torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average='macro')
        self.train_precision = torchmetrics.Precision(task="multiclass", num_classes=num_classes, average='macro')
        self.train_recall = torchmetrics.Recall(task="multiclass", num_classes=num_classes, average='macro')
        
        self.val_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_f1 = torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average='macro')
        self.val_precision = torchmetrics.Precision(task="multiclass", num_classes=num_classes, average='macro')
        self.val_recall = torchmetrics.Recall(task="multiclass", num_classes=num_classes, average='macro')
        
        self.validation_predictions = []
        self.validation_targets = []
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        data, target = batch
        outputs = self.model(data)
        loss = self.criterion(outputs, target)
        
        preds = torch.argmax(outputs, dim=1)
        self.train_accuracy(preds, target)
        self.train_f1(preds, target)
        self.train_precision(preds, target)
        self.train_recall(preds, target)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', self.train_accuracy, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_f1', self.train_f1, on_step=False, on_epoch=True)
        self.log('train_precision', self.train_precision, on_step=False, on_epoch=True)
        self.log('train_recall', self.train_recall, on_step=False, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        track_segments, track_target = batch
        
        segment_preds = []
        total_loss = 0.0
        
        for segment in track_segments:
            if segment.numel() == 0:
                continue
                
            with torch.no_grad():
                segment = segment.to(self.device)
                if segment.dim() == 3:
                    segment = segment.unsqueeze(0)
                elif segment.dim() == 1:
                    segment = segment.unsqueeze(0)
                
                output = self.model(segment)
                
                target_tensor = torch.tensor([track_target], device=self.device)
                segment_loss = self.criterion(output, target_tensor)
                total_loss += segment_loss.item()
                
                pred = torch.argmax(output, dim=1)
                segment_preds.append(pred.item())
        
        if not segment_preds:
            return torch.tensor(0.0, device=self.device)
        
        vote_counts = Counter(segment_preds)
        track_pred = vote_counts.most_common(1)[0][0]
        
        self.validation_predictions.append(track_pred)
        self.validation_targets.append(track_target)
        
        avg_loss = total_loss / len(segment_preds) if segment_preds else 0.0
        return torch.tensor(avg_loss, device=self.device)
    
    def on_validation_epoch_end(self):
        if self.validation_predictions and self.validation_targets:
            pred_tensor = torch.tensor(self.validation_predictions, device=self.device)
            target_tensor = torch.tensor(self.validation_targets, device=self.device)
            
            self.val_accuracy(pred_tensor, target_tensor)
            self.val_f1(pred_tensor, target_tensor)
            self.val_precision(pred_tensor, target_tensor)
            self.val_recall(pred_tensor, target_tensor)
            
            self.log('val_acc', self.val_accuracy, on_epoch=True, prog_bar=True)
            self.log('val_f1', self.val_f1, on_epoch=True, prog_bar=True)
            self.log('val_precision', self.val_precision, on_epoch=True, prog_bar=True)
            self.log('val_recall', self.val_recall, on_epoch=True, prog_bar=True)
            
            self.validation_predictions.clear()
            self.validation_targets.clear()
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_acc",
                "frequency": 1
            }
        }

def get_data_paths(data_dir, method):
    data_paths = []
    method_dir = os.path.join(data_dir, method)
    
    for genre in GENRES:
        genre_dir = os.path.join(method_dir, genre)
        if os.path.exists(genre_dir):
            track_names = set()
            all_files = os.listdir(genre_dir)
            
            for file in all_files:
                if file.endswith('.wav'):
                    if method == 'method2_5sec':
                        if '_seg' in file:
                            base_name = file.split('_seg')[0]
                            track_names.add(base_name)
                    elif method == 'method3_2bar':
                        if '_2bar' in file:
                            base_name = file.split('_2bar')[0]
                            track_names.add(base_name)
            
            for base_name in track_names:
                data_paths.append((base_name, genre))
    
    return data_paths

def track_collate_fn(batch):
    return batch[0]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='gtzan_preprocessed')
    parser.add_argument('--method', choices=['method2_5sec', 'method3_2bar'], default='method2_5sec')
    parser.add_argument('--feature', choices=['mel', 'cqt','audio'], default='mel') # use 'mel' or  'cqu' in mert, then it will use audio feature automatically, spaghetti haha
    parser.add_argument('--model', choices=['convnext', 'simple_cnn', 'mert'], default='convnext')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--num_workers', type=int, default=4)
    
    args = parser.parse_args()
    
    if args.model == 'mert':
        args.batch_size = min(args.batch_size, 4)
    
    torch.cuda.empty_cache()
    torch.set_float32_matmul_precision('medium')
    
    data_paths = get_data_paths(args.data_dir, args.method)
    file_paths = [path for path, _ in data_paths]
    labels = [GENRE_TO_IDX[genre] for _, genre in data_paths]
    
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(file_paths, labels)):
        train_paths = [data_paths[i] for i in train_idx]
        val_paths = [data_paths[i] for i in val_idx]
        
        train_dataset = GTZANSegmentDataset(train_paths, args.feature, args.method, args.model)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                                num_workers=args.num_workers, pin_memory=True)
        
        val_dataset = GTZANTrackDataset(val_paths, args.feature, args.method, args.model)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, 
                              collate_fn=track_collate_fn, num_workers=0)
        
        model = GTZANLightningModule(
            model_type=args.model,
            num_classes=len(GENRES),
            learning_rate=args.learning_rate
        )
        
        checkpoint_callback = ModelCheckpoint(
            dirpath=f'checkpoints/fold_{fold+1}',
            filename='{epoch}-{val_acc:.2f}',
            monitor='val_acc',
            mode='max',
            save_top_k=1
        )
        
        early_stopping = EarlyStopping(monitor='val_acc', mode='max', patience=10)
        logger = TensorBoardLogger('logs', name=f'{args.model}_{args.method}_{args.feature}', version=f'fold_{fold+1}')
        
        trainer = pl.Trainer(
            max_epochs=args.epochs,
            callbacks=[checkpoint_callback, early_stopping],
            logger=logger,
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            devices=1,
            gradient_clip_val=1.0,
            accumulate_grad_batches=2 if args.model == 'mert' else 1,
            deterministic=True
        )
        
        trainer.fit(model, train_loader, val_loader)
        
        best_val_acc = checkpoint_callback.best_model_score.item()
        val_f1 = trainer.logged_metrics.get('val_f1', torch.tensor(0.0))
        val_precision = trainer.logged_metrics.get('val_precision', torch.tensor(0.0))
        val_recall = trainer.logged_metrics.get('val_recall', torch.tensor(0.0))
        
        if hasattr(val_f1, 'item'):
            val_f1 = val_f1.item()
        if hasattr(val_precision, 'item'):
            val_precision = val_precision.item()
        if hasattr(val_recall, 'item'):
            val_recall = val_recall.item()
        
        fold_result = {
            'accuracy': best_val_acc * 100,
            'f1': float(val_f1) * 100,
            'precision': float(val_precision) * 100,
            'recall': float(val_recall) * 100
        }
        
        fold_results.append(fold_result)
        
        del model, trainer, train_loader, val_loader, train_dataset, val_dataset
        torch.cuda.empty_cache()
    
    accuracies = [result['accuracy'] for result in fold_results]
    f1_scores = [result['f1'] for result in fold_results]
    precisions = [result['precision'] for result in fold_results]
    recalls = [result['recall'] for result in fold_results]
    
    mean_acc = np.mean(accuracies)
    mean_f1 = np.mean(f1_scores)
    mean_precision = np.mean(precisions)
    mean_recall = np.mean(recalls)
    
    print(f"Final Results - {args.method}, {args.feature}, {args.model}")
    print(f"Accuracy: {mean_acc:.2f}%")
    print(f"F1 Score: {mean_f1:.2f}%")
    print(f"Precision: {mean_precision:.2f}%")
    print(f"Recall: {mean_recall:.2f}%")

if __name__ == '__main__':
    main()