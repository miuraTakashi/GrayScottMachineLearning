#!/usr/bin/env python3
"""
Gray-Scott 3D CNN Autoencoder Phase 2
残差接続 + 注意機構による更なる性能向上

Phase 1の成果: Silhouette Score 0.413 → 0.565 (+36.8%)
Phase 2目標: 0.565 → 0.65+ (+15-25% 追加向上)
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
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn.functional as F

# GPU設定
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ================================
# 1. データローダー（Phase 1から継承）
# ================================

class GrayScottDataset(Dataset):
    def __init__(self, gif_folder, fixed_frames=30, target_size=(64, 64), frame_range=None):
        """Gray-Scott GIFデータセット"""
        self.gif_folder = gif_folder
        self.fixed_frames = fixed_frames
        self.target_size = target_size
        self.frame_range = frame_range
        
        if frame_range is not None:
            if len(frame_range) != 2 or frame_range[0] >= frame_range[1] or frame_range[0] < 0:
                raise ValueError("frame_range must be a tuple (start, end) with start < end and start >= 0")
        
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
            f_val = float(match.group(1))
            k_val = float(match.group(2))
            return f_val, k_val
        return None, None
    
    def _load_gif_as_tensor(self, gif_path):
        """GIFを3Dテンソルに変換"""
        try:
            gif = imageio.mimread(gif_path)
            
            if self.frame_range is not None:
                start_frame, end_frame = self.frame_range
                start_frame = max(0, min(start_frame, len(gif) - 1))
                end_frame = max(start_frame + 1, min(end_frame, len(gif)))
                gif = gif[start_frame:end_frame]
            
            frames = []
            for frame in gif:
                if len(frame.shape) == 3:
                    frame = np.mean(frame, axis=2)
                
                pil_frame = Image.fromarray(frame.astype(np.uint8))
                pil_frame = pil_frame.resize(self.target_size)
                frame_array = np.array(pil_frame) / 255.0
                frames.append(frame_array)
            
            if len(frames) > self.fixed_frames:
                frames = frames[:self.fixed_frames]
            elif len(frames) < self.fixed_frames:
                if len(frames) > 0:
                    last_frame = frames[-1]
                    while len(frames) < self.fixed_frames:
                        frames.append(last_frame)
                else:
                    zero_frame = np.zeros(self.target_size)
                    frames = [zero_frame] * self.fixed_frames
            
            tensor = torch.FloatTensor(np.array(frames))
            tensor = tensor.unsqueeze(0)  # [1, fixed_frames, height, width]
            
            return tensor
            
        except Exception as e:
            print(f"Error loading {gif_path}: {e}")
            return None
    
    def _load_data(self):
        """全GIFファイルを読み込み"""
        gif_files = [f for f in os.listdir(self.gif_folder) if f.endswith('.gif')]
        
        print(f"Found {len(gif_files)} GIF files")
        if self.frame_range is not None:
            print(f"Using frame range: {self.frame_range[0]} to {self.frame_range[1]}")
        
        for i, filename in enumerate(gif_files):
            if i % 50 == 0:
                print(f"Loading {i+1}/{len(gif_files)}: {filename}")
            
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
        
        print(f"Successfully loaded {len(self.tensors)} GIF files")
    
    def __len__(self):
        return len(self.tensors)
    
    def __getitem__(self, idx):
        return {
            'tensor': self.tensors[idx],
            'f_value': self.f_values[idx],
            'k_value': self.k_values[idx],
            'filename': self.gif_files[idx]
        }

# ================================
# 2. Phase 2: 時空間注意機構
# ================================

class SpatioTemporalAttention(nn.Module):
    """時空間注意機構"""
    def __init__(self, channels):
        super(SpatioTemporalAttention, self).__init__()
        
        # 空間注意（どの場所が重要？）
        self.spatial_attention = nn.Sequential(
            nn.AdaptiveAvgPool3d((None, 1, 1)),  # 空間次元を平均化
            nn.Conv3d(channels, channels//4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels//4, channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 時間注意（どの時刻が重要？）
        self.temporal_attention = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, None, None)),  # 時間次元を平均化
            nn.Conv3d(channels, channels//4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels//4, channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        # チャンネル注意（どの特徴が重要？）
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channels, channels//4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels//4, channels, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # 元の入力を保存
        identity = x
        
        # 空間注意を適用
        spatial_weight = self.spatial_attention(x)
        x = x * spatial_weight
        
        # 時間注意を適用
        temporal_weight = self.temporal_attention(x)
        x = x * temporal_weight
        
        # チャンネル注意を適用
        channel_weight = self.channel_attention(x)
        x = x * channel_weight
        
        # 残差接続（注意機構レベルでも）
        x = x + identity * 0.1  # 小さな残差接続
        
        return x

# ================================
# 3. Phase 2: 残差ブロック + 注意機構
# ================================

class ResidualAttentionBlock3D(nn.Module):
    """残差接続 + 注意機構ブロック"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualAttentionBlock3D, self).__init__()
        
        # メイン経路の畳み込み
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels, momentum=0.1)
        
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, 
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels, momentum=0.1)
        
        # 注意機構
        self.attention = SpatioTemporalAttention(out_channels)
        
        # ショートカット接続
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm3d(out_channels, momentum=0.1)
            )
        
        # Dropout（過学習防止）
        self.dropout = nn.Dropout3d(0.1)
    
    def forward(self, x):
        identity = x
        
        # メイン経路の処理
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        
        # 注意機構を適用
        out = self.attention(out)
        
        # 残差接続
        out += self.shortcut(identity)
        out = F.relu(out)
        
        return out

# ================================
# 4. Phase 2: 改善されたAutoencoder
# ================================

class Conv3DAutoencoderPhase2(nn.Module):
    """Phase 2: 残差接続 + 注意機構 Autoencoder"""
    def __init__(self, input_channels=1, fixed_frames=30, target_size=(64, 64), latent_dim=256):
        super(Conv3DAutoencoderPhase2, self).__init__()
        
        self.latent_dim = latent_dim
        self.fixed_frames = fixed_frames
        self.target_size = target_size
        
        # Phase 2 改善エンコーダー（残差 + 注意）
        self.initial_conv = nn.Sequential(
            nn.Conv3d(input_channels, 32, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3)),
            nn.BatchNorm3d(32, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Dropout3d(0.05)
        )
        
        # 残差注意ブロック群
        self.res_block1 = ResidualAttentionBlock3D(32, 64, stride=(2, 2, 2))
        self.res_block2 = ResidualAttentionBlock3D(64, 64)
        
        self.res_block3 = ResidualAttentionBlock3D(64, 128, stride=(2, 2, 2))
        self.res_block4 = ResidualAttentionBlock3D(128, 128)
        
        self.res_block5 = ResidualAttentionBlock3D(128, 256, stride=(2, 2, 2))
        self.res_block6 = ResidualAttentionBlock3D(256, 256)
        
        # グローバル特徴抽出
        self.global_pool = nn.AdaptiveAvgPool3d((2, 2, 2))
        self.dropout_before_latent = nn.Dropout3d(0.3)
        
        # 潜在空間射影（Phase 1から改善）
        self.to_latent = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 2 * 2 * 2, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, latent_dim),
            nn.BatchNorm1d(latent_dim)
        )
        
        # 潜在空間からの復元（Phase 1から改善）
        self.from_latent = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, 256 * 2 * 2 * 2),
            nn.ReLU(inplace=True)
        )
        
        # Phase 2 改善デコーダー
        self.decoder = nn.Sequential(
            # 第1層: [256, 2, 2, 2] -> [128, 4, 4, 4]
            nn.ConvTranspose3d(256, 128, kernel_size=(3, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(128, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Dropout3d(0.2),
            
            # 第2層: [128, 4, 4, 4] -> [64, 8, 8, 8]
            nn.ConvTranspose3d(128, 64, kernel_size=(3, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(64, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Dropout3d(0.15),
            
            # 第3層: [64, 8, 8, 8] -> [32, 16, 16, 16]
            nn.ConvTranspose3d(64, 32, kernel_size=(3, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(32, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Dropout3d(0.1),
            
            # 最終層: [32, 16, 16, 16] -> [1, 30, 64, 64]
            nn.ConvTranspose3d(32, input_channels, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3)),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        """エンコード処理"""
        x = self.initial_conv(x)
        
        # 残差注意ブロック群を通過
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.res_block4(x)
        x = self.res_block5(x)
        x = self.res_block6(x)
        
        # グローバルプーリング
        x = self.global_pool(x)
        x = self.dropout_before_latent(x)
        
        # 潜在空間へ射影
        latent = self.to_latent(x)
        return latent
    
    def decode(self, latent):
        """デコード処理"""
        x = self.from_latent(latent)
        x = x.view(-1, 256, 2, 2, 2)
        x = self.decoder(x)
        
        # 適応的サイズ調整
        target_h, target_w = self.target_size
        x = F.interpolate(x, size=(self.fixed_frames, target_h, target_w), 
                         mode='trilinear', align_corners=False)
        return x
    
    def forward(self, x):
        latent = self.encode(x)
        reconstructed = self.decode(latent)
        return reconstructed, latent

# ================================
# 5. Phase 2 訓練関数
# ================================

def train_autoencoder_phase2(model, dataloader, num_epochs=50, learning_rate=1e-3, weight_decay=1e-4):
    """Phase 2改善版 Autoencoderの訓練"""
    import time
    
    model = model.to(device)
    criterion = nn.MSELoss()
    
    # Phase 2: AdamW + 改善されたスケジューラー
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    
    model.train()
    losses = []
    
    print(f"Phase 2 Training: ResNet + Attention, Latent dim={model.latent_dim}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training for {num_epochs} epochs...")
    print("=" * 70)
    
    start_time = time.time()
    epoch_times = []
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        epoch_loss = 0.0
        
        for batch_idx, batch in enumerate(dataloader):
            tensors = batch['tensor'].to(device)
            
            optimizer.zero_grad()
            
            # フォワードパス
            reconstructed, latent = model(tensors)
            
            # 損失計算
            loss = criterion(reconstructed, tensors)
            
            # バックプロパゲーション
            loss.backward()
            
            # Phase 2: 改善されたGradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            
            optimizer.step()
            epoch_loss += loss.item()
        
        scheduler.step()
        
        # エポック終了時の処理
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        epoch_times.append(epoch_duration)
        
        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        current_lr = scheduler.get_last_lr()[0]
        
        # 推定残り時間計算
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        remaining_epochs = num_epochs - (epoch + 1)
        estimated_remaining_time = avg_epoch_time * remaining_epochs
        
        # 進行状況表示（1エポックごと）
        progress_percent = ((epoch + 1) / num_epochs) * 100
        
        print(f'Epoch [{epoch+1:2d}/{num_epochs}] '
              f'({progress_percent:5.1f}%) | '
              f'Loss: {avg_loss:.6f} | '
              f'LR: {current_lr:.2e} | '
              f'Time: {epoch_duration:.1f}s | '
              f'ETA: {estimated_remaining_time/60:.1f}min')
        
        # 5エポックごとに詳細統計を表示
        if (epoch + 1) % 5 == 0:
            total_elapsed = time.time() - start_time
            print(f'    ├─ 平均エポック時間: {avg_epoch_time:.1f}s')
            print(f'    ├─ 総経過時間: {total_elapsed/60:.1f}min')
            print(f'    └─ 推定総実行時間: {(total_elapsed + estimated_remaining_time)/60:.1f}min')
            print("-" * 70)
    
    total_time = time.time() - start_time
    print("=" * 70)
    print(f"Phase 2 Training completed!")
    print(f"Total training time: {total_time/60:.1f} minutes")
    print(f"Average epoch time: {sum(epoch_times)/len(epoch_times):.1f} seconds")
    print("=" * 70)
    
    return losses

# ================================
# 6. クラスタリング関数（Phase 1から継承）
# ================================

def extract_latent_vectors(model, dataloader):
    """訓練済みモデルから潜在ベクトルを抽出"""
    model.eval()
    
    latent_vectors = []
    f_values = []
    k_values = []
    filenames = []
    
    with torch.no_grad():
        for batch in dataloader:
            tensors = batch['tensor'].to(device)
            _, latent = model(tensors)
            
            latent_vectors.append(latent.cpu().numpy())
            f_values.extend(batch['f_value'].numpy())
            k_values.extend(batch['k_value'].numpy())
            filenames.extend(batch['filename'])
    
    latent_vectors = np.vstack(latent_vectors)
    return latent_vectors, np.array(f_values), np.array(k_values), filenames

def perform_clustering(latent_vectors, n_clusters=5):
    """潜在ベクトルに対してクラスタリングを実行"""
    scaler = StandardScaler()
    latent_scaled = scaler.fit_transform(latent_vectors)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(latent_scaled)
    
    pca = PCA(n_components=2)
    latent_2d_pca = pca.fit_transform(latent_scaled)
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(latent_vectors)-1))
    latent_2d_tsne = tsne.fit_transform(latent_scaled)
    
    return cluster_labels, latent_2d_pca, latent_2d_tsne, pca, tsne

# ================================
# 7. 可視化関数
# ================================

def visualize_results_phase2(f_values, k_values, cluster_labels, latent_2d_pca, latent_2d_tsne):
    """Phase 2結果の可視化"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Phase 2: ResNet + Attention Results', fontsize=16, fontweight='bold')
    
    # 1. f-k平面でのクラスタリング結果
    scatter1 = axes[0, 0].scatter(f_values, k_values, c=cluster_labels, cmap='viridis', alpha=0.7)
    axes[0, 0].set_xlabel('f parameter')
    axes[0, 0].set_ylabel('k parameter')
    axes[0, 0].set_title('Phase 2: Clustering in f-k Space\n(ResNet + Attention)')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].invert_yaxis()
    plt.colorbar(scatter1, ax=axes[0, 0], label='Cluster')
    
    # 2. PCA可視化
    scatter2 = axes[0, 1].scatter(latent_2d_pca[:, 0], latent_2d_pca[:, 1], c=cluster_labels, cmap='viridis', alpha=0.7)
    axes[0, 1].set_xlabel('PCA Component 1')
    axes[0, 1].set_ylabel('PCA Component 2')
    axes[0, 1].set_title('Phase 2: PCA Visualization\n(256D Latent + ResNet)')
    axes[0, 1].grid(True, alpha=0.3)
    plt.colorbar(scatter2, ax=axes[0, 1], label='Cluster')
    
    # 3. t-SNE可視化
    scatter3 = axes[1, 0].scatter(latent_2d_tsne[:, 0], latent_2d_tsne[:, 1], c=cluster_labels, cmap='viridis', alpha=0.7)
    axes[1, 0].set_xlabel('t-SNE Component 1')
    axes[1, 0].set_ylabel('t-SNE Component 2')
    axes[1, 0].set_title('Phase 2: t-SNE Visualization\n(Attention Enhanced)')
    axes[1, 0].grid(True, alpha=0.3)
    plt.colorbar(scatter3, ax=axes[1, 0], label='Cluster')
    
    # 4. f-k平面のヒートマップ
    f_unique = np.unique(f_values)
    k_unique = np.unique(k_values)
    heatmap_data = np.full((len(f_unique), len(k_unique)), np.nan)
    
    for i, f in enumerate(f_values):
        k = k_values[i]
        cluster = cluster_labels[i]
        
        f_idx = np.where(f_unique == f)[0][0]
        k_idx = np.where(k_unique == k)[0][0]
        heatmap_data[f_idx, k_idx] = cluster
    
    im = axes[1, 1].imshow(heatmap_data, cmap='viridis', aspect=1, origin='upper')
    axes[1, 1].set_xticks(range(len(k_unique)))
    axes[1, 1].set_yticks(range(len(f_unique)))
    axes[1, 1].set_xticklabels([f'{k:.4f}' for k in k_unique], rotation=45)
    axes[1, 1].set_yticklabels([f'{f:.4f}' for f in f_unique])
    axes[1, 1].set_xlabel('k parameter')
    axes[1, 1].set_ylabel('f parameter')
    axes[1, 1].set_title('Phase 2: Attention-Enhanced Heatmap')
    axes[1, 1].invert_yaxis()
    
    # カラーバーとレジェンド
    unique_clusters = np.unique(cluster_labels[~np.isnan(cluster_labels)])
    cmap = plt.cm.viridis
    legend_elements = [Patch(facecolor=cmap(cluster/len(unique_clusters)), 
                           label=f'Cluster {int(cluster)}') 
                     for cluster in unique_clusters]
    axes[1, 1].legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig('../results/gray_scott_clustering_results_phase2.png', dpi=300, bbox_inches='tight')
    plt.show()

# ================================
# 8. メイン実行関数
# ================================

def main():
    """Phase 2 メイン実行関数"""
    
    # Phase 2 ハイパーパラメータ
    gif_folder = '/Users/nakashimarikuto/Gray scott/GrayScottMachineLearning/data/gif'
    fixed_frames = 30
    target_size = (64, 64)
    latent_dim = 256  # Phase 1と同じ
    num_epochs = 50
    batch_size = 4
    learning_rate = 1e-3
    weight_decay = 1e-4
    n_clusters = 5
    
    print("=" * 70)
    print("Phase 2: ResNet + Attention Gray-Scott Analysis")
    print("=" * 70)
    print(f"主要改善点:")
    print(f"✓ 残差接続（ResNet）: 深層ネットワークの安定学習")
    print(f"✓ 時空間注意機構: 重要領域・時刻への集中")
    print(f"✓ チャンネル注意: 重要特徴の選択")
    print(f"✓ 改善された正則化: Dropout, BatchNorm")
    print(f"期待改善率: Phase 1 (0.565) → Phase 2 (0.65+)")
    print("=" * 70)
    
    # データセットの作成
    print("Loading dataset...")
    dataset = GrayScottDataset(gif_folder, fixed_frames, target_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    print(f"Dataset size: {len(dataset)} samples")
    print(f"Batch size: {batch_size}")
    
    # Phase 2 モデルの作成
    print("Creating Phase 2 model (ResNet + Attention)...")
    model = Conv3DAutoencoderPhase2(latent_dim=latent_dim, fixed_frames=fixed_frames, target_size=target_size)
    
    # モデル訓練
    print("Training Phase 2 model...")
    losses = train_autoencoder_phase2(model, dataloader, num_epochs, learning_rate, weight_decay)
    
    # 訓練曲線の可視化
    plt.figure(figsize=(10, 6))
    plt.plot(losses, linewidth=2, color='purple')
    plt.title('Phase 2: Training Loss (ResNet + Attention)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.savefig('../results/training_loss_phase2.png', dpi=300)
    plt.show()
    
    # モデル保存
    print("Saving Phase 2 model...")
    os.makedirs('../models', exist_ok=True)
    torch.save(model.state_dict(), '../models/trained_autoencoder_phase2.pth')
    
    # 潜在ベクトルの抽出
    print("Extracting latent vectors...")
    latent_vectors, f_values, k_values, filenames = extract_latent_vectors(model, dataloader)
    
    print(f"Latent vectors shape: {latent_vectors.shape}")
    print(f"Latent dimension: {latent_vectors.shape[1]}")
    
    # クラスタリング
    print("Performing clustering...")
    cluster_labels, latent_2d_pca, latent_2d_tsne, pca, tsne = perform_clustering(latent_vectors, n_clusters)
    
    # 結果保存
    print("Saving results...")
    os.makedirs('../results', exist_ok=True)
    
    results = {
        'latent_vectors': latent_vectors,
        'f_values': f_values,
        'k_values': k_values,
        'filenames': filenames,
        'cluster_labels': cluster_labels,
        'latent_2d_pca': latent_2d_pca,
        'latent_2d_tsne': latent_2d_tsne,
        'pca': pca,
        'tsne': tsne,
        'losses': losses,
        'hyperparameters': {
            'latent_dim': latent_dim,
            'num_epochs': num_epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'weight_decay': weight_decay,
            'n_clusters': n_clusters,
            'optimizer': 'AdamW',
            'scheduler': 'CosineAnnealingLR',
            'architecture': 'ResNet + SpatioTemporalAttention'
        }
    }
    
    import pickle
    with open('../results/analysis_results_phase2.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    # 可視化
    print("Creating visualizations...")
    visualize_results_phase2(f_values, k_values, cluster_labels, latent_2d_pca, latent_2d_tsne)
    
    # クラスタリング品質評価
    from sklearn.metrics import silhouette_score
    silhouette_avg = silhouette_score(latent_vectors, cluster_labels)
    
    print("=" * 70)
    print("Phase 2 Results Summary:")
    print("=" * 70)
    print(f"Architecture: ResNet + SpatioTemporalAttention")
    print(f"Samples: {len(dataset)}")
    print(f"Latent Dimension: {latent_dim}")
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Final Training Loss: {losses[-1]:.6f}")
    print(f"Clusters: {n_clusters}")
    print(f"Silhouette Score: {silhouette_avg:.4f}")
    print("=" * 70)
    
    # Phase 2 改善点の効果
    print("Phase 2 Improvements Applied:")
    print("✓ ResNet architecture with skip connections")
    print("✓ SpatioTemporal attention mechanism")
    print("✓ Channel attention for feature selection")
    print("✓ Enhanced regularization (Dropout + BatchNorm)")
    print("✓ Improved gradient clipping (max_norm=0.5)")
    print("✓ Multi-level attention integration")
    print("=" * 70)

if __name__ == "__main__":
    main() 