#!/usr/bin/env python3
"""
Gray-Scott 3D CNN Autoencoder Phase 3
マルチスケール特徴融合による高度な特徴学習

Phase 2の成果: Silhouette Score 0.4671 (ResNet+Attention)
Phase 3目標: 0.4671 → 0.55+ (+15-20% 追加向上)

主要改善点:
1. マルチスケール特徴融合 (Multi-Scale Feature Fusion)
2. 高度なデータ拡張戦略 (Advanced Data Augmentation)
3. 改善された訓練ループ (Enhanced Training Loop)
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from PIL import Image
import imageio.v2 as imageio
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import time
import pickle
from typing import List, Tuple, Dict, Optional

# GPU設定
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Phase 3 Using device: {device}")

# ================================
# 1. 高度なデータ拡張システム
# ================================

class GrayScottAugmentation:
    """Gray-Scott専用データ拡張クラス"""
    
    def __init__(self, 
                 temporal_shift_prob=0.3,
                 spatial_flip_prob=0.5,
                 noise_prob=0.2,
                 intensity_prob=0.3,
                 temporal_crop_prob=0.2):
        self.temporal_shift_prob = temporal_shift_prob
        self.spatial_flip_prob = spatial_flip_prob
        self.noise_prob = noise_prob
        self.intensity_prob = intensity_prob
        self.temporal_crop_prob = temporal_crop_prob
    
    def temporal_shift(self, tensor, max_shift=3):
        """時間軸シフト"""
        if np.random.random() < self.temporal_shift_prob:
            shift = np.random.randint(-max_shift, max_shift + 1)
            if shift != 0:
                tensor = torch.roll(tensor, shift, dims=1)  # 時間軸でシフト
        return tensor
    
    def spatial_flip(self, tensor):
        """空間軸反転"""
        if np.random.random() < self.spatial_flip_prob:
            # 水平反転
            if np.random.random() < 0.5:
                tensor = torch.flip(tensor, dims=[3])  # width軸
            # 垂直反転
            if np.random.random() < 0.5:
                tensor = torch.flip(tensor, dims=[2])  # height軸
        return tensor
    
    def add_noise(self, tensor, noise_std=0.02):
        """ガウシアンノイズ追加"""
        if np.random.random() < self.noise_prob:
            noise = torch.randn_like(tensor) * noise_std
            tensor = torch.clamp(tensor + noise, 0, 1)
        return tensor
    
    def intensity_transform(self, tensor, gamma_range=(0.8, 1.2)):
        """強度変換（ガンマ補正）"""
        if np.random.random() < self.intensity_prob:
            gamma = np.random.uniform(*gamma_range)
            tensor = torch.pow(tensor, gamma)
        return tensor
    
    def temporal_crop(self, tensor, crop_ratio=0.1):
        """時間軸クロップ"""
        if np.random.random() < self.temporal_crop_prob:
            T = tensor.shape[1]
            crop_size = int(T * crop_ratio)
            start_idx = np.random.randint(0, crop_size + 1)
            end_idx = T - np.random.randint(0, crop_size + 1)
            
            # クロップした部分を補間で埋める
            cropped = tensor[:, start_idx:end_idx]
            tensor = F.interpolate(cropped.unsqueeze(0), size=(T, tensor.shape[2], tensor.shape[3]), 
                                 mode='trilinear', align_corners=False).squeeze(0)
        return tensor
    
    def __call__(self, tensor):
        """全ての拡張を適用"""
        tensor = self.temporal_shift(tensor)
        tensor = self.spatial_flip(tensor)
        tensor = self.add_noise(tensor)
        tensor = self.intensity_transform(tensor)
        tensor = self.temporal_crop(tensor)
        return tensor

# ================================
# 2. マルチスケール特徴融合モジュール
# ================================

class MultiScaleFeatureFusion(nn.Module):
    """マルチスケール特徴融合モジュール"""
    
    def __init__(self, in_channels, out_channels):
        super(MultiScaleFeatureFusion, self).__init__()
        
        # 異なるカーネルサイズでの並列処理
        self.scale1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels//4, kernel_size=(1, 1, 1), padding=0),
            nn.BatchNorm3d(out_channels//4),
            nn.ReLU(inplace=True)
        )
        
        self.scale2 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels//4, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(out_channels//4),
            nn.ReLU(inplace=True)
        )
        
        self.scale3 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels//4, kernel_size=(5, 5, 5), padding=2),
            nn.BatchNorm3d(out_channels//4),
            nn.ReLU(inplace=True)
        )
        
        # プーリング分岐
        self.pool_branch = nn.Sequential(
            nn.AvgPool3d(kernel_size=3, stride=1, padding=1),
            nn.Conv3d(in_channels, out_channels//4, kernel_size=(1, 1, 1), padding=0),
            nn.BatchNorm3d(out_channels//4),
            nn.ReLU(inplace=True)
        )
        
        # 特徴融合
        self.fusion = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=(1, 1, 1)),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 注意機構による重み付け
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(out_channels, out_channels//8, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels//8, out_channels, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # 各スケールでの特徴抽出
        feat1 = self.scale1(x)  # 1x1x1 - 点特徴
        feat2 = self.scale2(x)  # 3x3x3 - 局所特徴
        feat3 = self.scale3(x)  # 5x5x5 - 広域特徴
        feat4 = self.pool_branch(x)  # プーリング特徴
        
        # 特徴を結合
        fused = torch.cat([feat1, feat2, feat3, feat4], dim=1)
        
        # 融合処理
        fused = self.fusion(fused)
        
        # 注意機構による重み付け
        attention_weights = self.attention(fused)
        fused = fused * attention_weights
        
        return fused

# ================================
# 3. 改良された時空間注意機構
# ================================

class EnhancedSpatioTemporalAttention(nn.Module):
    """改良された時空間注意機構"""
    
    def __init__(self, channels):
        super(EnhancedSpatioTemporalAttention, self).__init__()
        
        # 分離可能な注意機構
        self.temporal_attention = nn.Sequential(
            nn.AdaptiveAvgPool3d((None, 1, 1)),
            nn.Conv3d(channels, channels//8, kernel_size=(3, 1, 1), padding=(1, 0, 0)),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels//8, channels, kernel_size=(1, 1, 1)),
            nn.Sigmoid()
        )
        
        self.spatial_attention = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, None, None)),
            nn.Conv3d(channels, channels//8, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels//8, channels, kernel_size=(1, 1, 1)),
            nn.Sigmoid()
        )
        
        # 融合層
        self.fusion = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=1),
            nn.BatchNorm3d(channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        B, C, T, H, W = x.shape
        identity = x
        
        # 時空間注意
        temp_att = self.temporal_attention(x)
        spat_att = self.spatial_attention(x)
        
        # 注意機構を適用
        x_att = x * temp_att * spat_att
        
        # 融合と残差接続
        x_fused = self.fusion(x_att)
        
        return x_fused + identity * 0.2

# ================================
# 4. Phase 3 メインアーキテクチャ
# ================================

class ResidualMultiScaleBlock3D(nn.Module):
    """残差マルチスケールブロック"""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualMultiScaleBlock3D, self).__init__()
        
        # マルチスケール特徴融合
        self.multiscale = MultiScaleFeatureFusion(in_channels, out_channels)
        
        # 改良された注意機構
        self.attention = EnhancedSpatioTemporalAttention(out_channels)
        
        # ショートカット接続
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )
        
        # 最終処理
        self.final_conv = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm3d(out_channels)
        )
        
        self.dropout = nn.Dropout3d(0.1)
        
    def forward(self, x):
        identity = x
        
        # マルチスケール特徴抽出
        out = self.multiscale(x)
        
        # 注意機構
        out = self.attention(out)
        
        # 最終畳み込み
        out = self.final_conv(out)
        out = self.dropout(out)
        
        # 残差接続
        out += self.shortcut(identity)
        
        return F.relu(out)

class Conv3DAutoencoderPhase3(nn.Module):
    """Phase 3: マルチスケール特徴融合 Autoencoder"""
    
    def __init__(self, input_channels=1, fixed_frames=30, target_size=(64, 64), latent_dim=512):
        super(Conv3DAutoencoderPhase3, self).__init__()
        
        self.latent_dim = latent_dim
        self.fixed_frames = fixed_frames
        self.target_size = target_size
        
        # 初期畳み込み
        self.initial_conv = nn.Sequential(
            nn.Conv3d(input_channels, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Dropout3d(0.05)
        )
        
        # マルチスケール残差ブロック群
        self.res_block1 = ResidualMultiScaleBlock3D(64, 128, stride=(2, 2, 2))
        self.res_block2 = ResidualMultiScaleBlock3D(128, 128)
        self.res_block3 = ResidualMultiScaleBlock3D(128, 256, stride=(2, 2, 2))
        self.res_block4 = ResidualMultiScaleBlock3D(256, 256)
        self.res_block5 = ResidualMultiScaleBlock3D(256, 512, stride=(2, 2, 2))
        self.res_block6 = ResidualMultiScaleBlock3D(512, 512)
        
        # グローバル特徴抽出
        self.global_pool = nn.AdaptiveAvgPool3d((2, 2, 2))
        self.dropout_before_latent = nn.Dropout3d(0.3)
        
        # 潜在空間射影（拡張）
        self.to_latent = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 2 * 2 * 2, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(1024, latent_dim),
            nn.BatchNorm1d(latent_dim)
        )
        
        # 復元パス
        self.from_latent = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(1024, 512 * 2 * 2 * 2),
            nn.ReLU(inplace=True)
        )
        
        # デコーダー（マルチスケール対応）
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(512, 256, kernel_size=(3, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.Dropout3d(0.2),
            
            nn.ConvTranspose3d(256, 128, kernel_size=(3, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Dropout3d(0.15),
            
            nn.ConvTranspose3d(128, 64, kernel_size=(3, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Dropout3d(0.1),
            
            nn.ConvTranspose3d(64, input_channels, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3)),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        """エンコード処理"""
        x = self.initial_conv(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.res_block4(x)
        x = self.res_block5(x)
        x = self.res_block6(x)
        x = self.global_pool(x)
        x = self.dropout_before_latent(x)
        return self.to_latent(x)
    
    def decode(self, latent):
        """デコード処理"""
        x = self.from_latent(latent)
        x = x.view(-1, 512, 2, 2, 2)
        x = self.decoder(x)
        
        # 目標サイズに調整
        target_h, target_w = self.target_size
        x = F.interpolate(x, size=(self.fixed_frames, target_h, target_w), 
                         mode='trilinear', align_corners=False)
        return x
    
    def forward(self, x):
        latent = self.encode(x)
        reconstructed = self.decode(latent)
        return reconstructed, latent

# ================================
# 5. 改良されたデータセット（拡張対応）
# ================================

class GrayScottDatasetPhase3(Dataset):
    """Phase 3対応データセット（拡張機能付き）"""
    
    def __init__(self, gif_folder, fixed_frames=30, target_size=(64, 64), 
                 use_augmentation=True, max_samples=None):
        self.gif_folder = gif_folder
        self.fixed_frames = fixed_frames
        self.target_size = target_size
        self.use_augmentation = use_augmentation
        self.max_samples = max_samples
        
        # データ拡張器
        self.augmentation = GrayScottAugmentation() if use_augmentation else None
        
        self.gif_files = []
        self.f_values = []
        self.k_values = []
        self.tensors = []
        
        self._load_data()
    
    def _parse_filename(self, filename):
        """ファイル名からf, kパラメータを抽出"""
        pattern = r'GrayScott-f([0-9.]+)-k([0-9.]+)-\d+\.gif'
        match = re.match(pattern, filename)
        if match:
            return float(match.group(1)), float(match.group(2))
        return None, None
    
    def _load_gif_as_tensor(self, gif_path):
        """GIFを3Dテンソルに変換"""
        try:
            gif = imageio.mimread(gif_path)
            frames = []
            
            for frame in gif[:self.fixed_frames]:
                if len(frame.shape) == 3:
                    frame = np.mean(frame, axis=2)
                
                pil_frame = Image.fromarray(frame.astype(np.uint8))
                pil_frame = pil_frame.resize(self.target_size)
                frame_array = np.array(pil_frame) / 255.0
                frames.append(frame_array)
            
            while len(frames) < self.fixed_frames:
                frames.append(frames[-1] if frames else np.zeros(self.target_size))
            
            tensor = torch.FloatTensor(np.array(frames[:self.fixed_frames]))
            return tensor.unsqueeze(0)
            
        except Exception as e:
            print(f"Error loading {gif_path}: {e}")
            return None
    
    def _load_data(self):
        """全GIFファイルを読み込み"""
        gif_files = [f for f in os.listdir(self.gif_folder) if f.endswith('.gif')]
        
        if self.max_samples:
            gif_files = gif_files[:self.max_samples]
        
        print(f"Loading {len(gif_files)} GIF files for Phase 3...")
        
        for i, filename in enumerate(gif_files):
            if i % 100 == 0:
                print(f"Progress: {i+1}/{len(gif_files)} ({(i+1)/len(gif_files)*100:.1f}%)")
            
            f_val, k_val = self._parse_filename(filename)
            if f_val is None or k_val is None:
                continue
            
            gif_path = os.path.join(self.gif_folder, filename)
            tensor = self._load_gif_as_tensor(gif_path)
            
            if tensor is not None:
                self.gif_files.append(filename)
                self.f_values.append(f_val)
                self.k_values.append(k_val)
                self.tensors.append(tensor)
        
        print(f"✅ Successfully loaded {len(self.tensors)} samples for Phase 3")
    
    def __len__(self):
        return len(self.tensors)
    
    def __getitem__(self, idx):
        tensor = self.tensors[idx].clone()
        
        # データ拡張適用
        if self.use_augmentation and self.augmentation is not None:
            tensor = self.augmentation(tensor)
        
        return {
            'tensor': tensor,
            'f_value': self.f_values[idx],
            'k_value': self.k_values[idx],
            'filename': self.gif_files[idx]
        }

# ================================
# 6. 改良された訓練システム
# ================================

def train_autoencoder_phase3(model, dataloader, num_epochs=60, learning_rate=1e-3, 
                           weight_decay=1e-4, warmup_epochs=5):
    """Phase 3 改良訓練システム"""
    
    print("🎯 Starting Phase 3 Training...")
    print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 損失関数（マルチタスク）
    mse_loss = nn.MSELoss()
    l1_loss = nn.L1Loss()
    
    # オプティマイザー（AdamW + スケジューラー）
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, 
                           betas=(0.9, 0.999), eps=1e-8)
    
    # 学習率スケジューラー（ウォームアップ + コサインアニーリング）
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        else:
            return 0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / (num_epochs - warmup_epochs)))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # 訓練ループ
    model.train()
    losses = []
    start_time = time.time()
    
    print("="*80)
    print("🚀 Phase 3 Multi-Scale Feature Fusion Training")
    print("="*80)
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        epoch_loss = 0.0
        
        for batch_idx, batch in enumerate(dataloader):
            tensors = batch['tensor'].to(device)
            
            optimizer.zero_grad()
            
            # フォワードパス
            reconstructed, latent = model(tensors)
            
            # マルチタスク損失
            loss_mse = mse_loss(reconstructed, tensors)
            loss_l1 = l1_loss(reconstructed, tensors)
            
            # 潜在空間正則化
            latent_reg = torch.mean(torch.norm(latent, dim=1))
            
            # 総合損失
            total_loss = loss_mse + 0.1 * loss_l1 + 0.001 * latent_reg
            
            # バックワードパス
            total_loss.backward()
            
            # 勾配クリッピング
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            epoch_loss += total_loss.item()
        
        scheduler.step()
        
        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        current_lr = scheduler.get_last_lr()[0]
        epoch_time = time.time() - epoch_start_time
        
        # 進捗表示
        progress = ((epoch + 1) / num_epochs) * 100
        if (epoch + 1) % 5 == 0 or epoch < 5:
            print(f'Epoch [{epoch+1:2d}/{num_epochs}] '
                  f'({progress:5.1f}%) | '
                  f'Loss: {avg_loss:.6f} | '
                  f'LR: {current_lr:.2e} | '
                  f'Time: {epoch_time:.1f}s')
    
    total_time = time.time() - start_time
    print("="*80)
    print(f"🎉 Phase 3 Training completed in {total_time/60:.1f} minutes")
    
    return losses

def extract_latent_vectors_phase3(model, dataloader):
    """Phase 3 潜在ベクトル抽出"""
    print("🔍 Extracting latent vectors (Phase 3)...")
    
    model.eval()
    latent_vectors = []
    f_values = []
    k_values = []
    
    with torch.no_grad():
        for batch in dataloader:
            tensors = batch['tensor'].to(device)
            _, latent = model(tensors)
            latent_vectors.append(latent.cpu().numpy())
            f_values.extend(batch['f_value'].numpy())
            k_values.extend(batch['k_value'].numpy())
    
    latent_vectors = np.vstack(latent_vectors)
    f_values = np.array(f_values)
    k_values = np.array(k_values)
    
    print(f"✅ Extracted {len(latent_vectors)} latent vectors (dim: {latent_vectors.shape[1]})")
    
    return latent_vectors, f_values, k_values

def perform_clustering_phase3(latent_vectors, n_clusters=6):
    """Phase 3 クラスタリング"""
    print(f"🎯 Performing clustering (k={n_clusters})...")
    
    # 標準化
    scaler = StandardScaler()
    latent_scaled = scaler.fit_transform(latent_vectors)
    
    # クラスタリング
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(latent_scaled)
    
    # 性能評価
    silhouette_avg = silhouette_score(latent_vectors, cluster_labels)
    calinski_score = calinski_harabasz_score(latent_vectors, cluster_labels)
    davies_bouldin = davies_bouldin_score(latent_vectors, cluster_labels)
    
    print(f"📊 Clustering Results:")
    print(f"   ⭐ Silhouette Score: {silhouette_avg:.4f}")
    print(f"   📊 Calinski-Harabasz: {calinski_score:.2f}")
    print(f"   📈 Davies-Bouldin: {davies_bouldin:.4f}")
    
    return cluster_labels, silhouette_avg, calinski_score, davies_bouldin

def main():
    """Phase 3 メイン実行関数"""
    
    print("="*80)
    print("🌐 Gray-Scott Phase 3: Multi-Scale Feature Fusion")
    print("="*80)
    
    # パラメータ設定
    gif_folder = 'data/gif'
    fixed_frames = 30
    target_size = (64, 64)
    latent_dim = 512  # Phase 2から拡張
    num_epochs = 60
    batch_size = 4  # メモリ使用量考慮
    learning_rate = 1e-3
    weight_decay = 1e-4
    n_clusters = 6
    
    # データセット作成
    print("📊 Creating Phase 3 dataset with augmentation...")
    dataset = GrayScottDatasetPhase3(gif_folder, fixed_frames, target_size, 
                                   use_augmentation=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                          num_workers=2, pin_memory=True)
    
    print(f"📊 Dataset: {len(dataset)} samples, Batch size: {batch_size}")
    
    # モデル作成
    print("🧠 Creating Phase 3 Multi-Scale model...")
    model = Conv3DAutoencoderPhase3(latent_dim=latent_dim, 
                                  fixed_frames=fixed_frames, 
                                  target_size=target_size).to(device)
    
    # 訓練実行
    losses = train_autoencoder_phase3(model, dataloader, num_epochs, 
                                    learning_rate, weight_decay)
    
    # 潜在ベクトル抽出
    latent_vectors, f_values, k_values = extract_latent_vectors_phase3(model, dataloader)
    
    # クラスタリング
    cluster_labels, silhouette_avg, calinski_score, davies_bouldin = \
        perform_clustering_phase3(latent_vectors, n_clusters)
    
    # 結果保存
    results = {
        'model_state_dict': model.state_dict(),
        'losses': losses,
        'latent_vectors': latent_vectors,
        'cluster_labels': cluster_labels,
        'f_values': f_values,
        'k_values': k_values,
        'silhouette_score': silhouette_avg,
        'calinski_score': calinski_score,
        'davies_bouldin': davies_bouldin,
        'hyperparameters': {
            'latent_dim': latent_dim,
            'num_epochs': num_epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'weight_decay': weight_decay,
            'n_clusters': n_clusters
        }
    }
    
    # 結果保存
    results_path = 'results/phase3_results.pkl'
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"💾 Results saved to: {results_path}")
    
    # 性能比較
    print("="*80)
    print("🏆 Phase 3 Results Summary:")
    print("="*80)
    print(f"🎯 Architecture: Multi-Scale Feature Fusion + Enhanced Attention")
    print(f"📊 Samples: {len(dataset)}")
    print(f"🧠 Latent Dimension: {latent_dim}")
    print(f"⚙️  Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"📉 Final Loss: {losses[-1]:.6f}")
    print(f"🎯 Clusters: {n_clusters}")
    print(f"⭐ Silhouette Score: {silhouette_avg:.4f}")
    print(f"📊 Calinski-Harabasz: {calinski_score:.2f}")
    print(f"📈 Davies-Bouldin: {davies_bouldin:.4f}")
    print("="*80)
    
    # Phase比較
    phase2_score = 0.4671
    improvement = ((silhouette_avg - phase2_score) / phase2_score) * 100
    
    print(f"📈 Performance Comparison:")
    print(f"   Phase 2: {phase2_score:.4f}")
    print(f"   Phase 3: {silhouette_avg:.4f}")
    print(f"   Improvement: {improvement:+.1f}%")
    
    if improvement >= 10:
        print("🎉 Phase 3 目標達成！ (10%以上の向上)")
    else:
        print(f"⚠️  Phase 3 目標未達 ({improvement:.1f}% < 10%)")
    
    return results

if __name__ == "__main__":
    main()
