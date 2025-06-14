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
# 1. データローダー（gif -> 3Dテンソル、f, kの抽出）
# ================================

class GrayScottDataset(Dataset):
    def __init__(self, gif_folder, fixed_frames=30, target_size=(64, 64), frame_range=None):
        """
        Gray-Scott GIFデータセット
        
        Args:
            gif_folder: GIFファイルが格納されたフォルダ
            fixed_frames: 固定フレーム数（パディングまたはトランケート）
            target_size: リサイズ後の画像サイズ
            frame_range: 使用するフレーム範囲 (start, end) のタプル。Noneの場合は全フレーム使用
        """
        self.gif_folder = gif_folder
        self.fixed_frames = fixed_frames
        self.target_size = target_size
        self.frame_range = frame_range
        
        # フレーム範囲の検証
        if frame_range is not None:
            if len(frame_range) != 2 or frame_range[0] >= frame_range[1] or frame_range[0] < 0:
                raise ValueError("frame_range must be a tuple (start, end) with start < end and start >= 0")
        
        # GIFファイルのリストとパラメータを取得
        self.gif_files = []
        self.f_values = []
        self.k_values = []
        self.tensors = []
        
        self._load_data()
    
    def _parse_filename(self, filename):
        """
        ファイル名からf, kパラメータを抽出
        例: GrayScott-f0.0580-k0.0680-00.gif -> f=0.0580, k=0.0680
        """
        pattern = r'GrayScott-f([0-9.]+)-k([0-9.]+)-\d+\.gif'
        match = re.match(pattern, filename)
        if match:
            f_val = float(match.group(1))
            k_val = float(match.group(2))
            return f_val, k_val
        return None, None
    
    def _load_gif_as_tensor(self, gif_path):
        """
        GIFを3Dテンソルに変換（フレーム範囲指定対応）
        
        Returns:
            tensor: shape [1, fixed_frames, height, width] のテンソル
        """
        try:
            # GIFを読み込み
            gif = imageio.mimread(gif_path)
            
            # フレーム範囲の適用
            if self.frame_range is not None:
                start_frame, end_frame = self.frame_range
                # 範囲をGIFの実際のフレーム数に制限
                start_frame = max(0, min(start_frame, len(gif) - 1))
                end_frame = max(start_frame + 1, min(end_frame, len(gif)))
                gif = gif[start_frame:end_frame]
            
            frames = []
            for frame in gif:
                # グレースケール変換
                if len(frame.shape) == 3:
                    frame = np.mean(frame, axis=2)
                
                # PIL Imageに変換してリサイズ
                pil_frame = Image.fromarray(frame.astype(np.uint8))
                pil_frame = pil_frame.resize(self.target_size)
                
                # 正規化 [0, 1]
                frame_array = np.array(pil_frame) / 255.0
                frames.append(frame_array)
            
            # 固定フレーム数に調整
            if len(frames) > self.fixed_frames:
                # トランケート
                frames = frames[:self.fixed_frames]
            elif len(frames) < self.fixed_frames:
                # パディング（最後のフレームを繰り返し）
                if len(frames) > 0:
                    last_frame = frames[-1]
                    while len(frames) < self.fixed_frames:
                        frames.append(last_frame)
                else:
                    # フレームが1つもない場合はゼロで埋める
                    zero_frame = np.zeros(self.target_size)
                    frames = [zero_frame] * self.fixed_frames
            
            # テンソルに変換 [fixed_frames, height, width]
            tensor = torch.FloatTensor(np.array(frames))
            
            # チャンネル次元を追加 [1, fixed_frames, height, width]
            tensor = tensor.unsqueeze(0)
            
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
# 2. Phase 1 改善版 Autoencoder（潜在次元256、強化BatchNorm+Dropout）
# ================================

class Conv3DAutoencoderPhase1(nn.Module):
    def __init__(self, input_channels=1, fixed_frames=30, target_size=(64, 64), latent_dim=256):
        super(Conv3DAutoencoderPhase1, self).__init__()
        
        self.latent_dim = latent_dim
        self.fixed_frames = fixed_frames
        self.target_size = target_size
        
        # Phase 1 改善: 強化されたEncoder
        self.encoder = nn.Sequential(
            # 第1層: [1, 30, 64, 64] -> [16, 15, 32, 32]
            nn.Conv3d(input_channels, 16, kernel_size=(3, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(16, momentum=0.1),  # 強化BatchNorm
            nn.ReLU(inplace=True),
            nn.Dropout3d(0.1),  # Dropout追加
            
            # 第2層: [16, 15, 32, 32] -> [32, 8, 16, 16] 
            nn.Conv3d(16, 32, kernel_size=(3, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(32, momentum=0.1),  # 強化BatchNorm
            nn.ReLU(inplace=True),
            nn.Dropout3d(0.1),  # Dropout追加
            
            # 第3層: [32, 8, 16, 16] -> [64, 4, 8, 8]
            nn.Conv3d(32, 64, kernel_size=(3, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(64, momentum=0.1),  # 強化BatchNorm
            nn.ReLU(inplace=True),
            nn.Dropout3d(0.15),  # より強いDropout
            
            # 第4層: [64, 4, 8, 8] -> [128, 2, 4, 4]
            nn.Conv3d(64, 128, kernel_size=(3, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(128, momentum=0.1),  # 強化BatchNorm
            nn.ReLU(inplace=True),
            nn.Dropout3d(0.15),  # より強いDropout
            
            # 第5層追加: [128, 2, 4, 4] -> [256, 1, 2, 2]
            nn.Conv3d(128, 256, kernel_size=(3, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(256, momentum=0.1),  # 強化BatchNorm
            nn.ReLU(inplace=True),
            nn.Dropout3d(0.2),  # 最強Dropout
        )
        
        # Phase 1 改善: 潜在空間への射影（256次元）
        self.to_latent = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
            nn.Linear(256, latent_dim),
            nn.BatchNorm1d(latent_dim),  # 潜在空間でもBatchNorm
            nn.Dropout(0.2)  # 潜在空間でもDropout
        )
        
        # Phase 1 改善: 潜在空間からの復元（256次元対応）
        self.from_latent = nn.Sequential(
            nn.Linear(latent_dim, 256 * 1 * 2 * 2),
            nn.BatchNorm1d(256 * 1 * 2 * 2),  # BatchNorm追加
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)  # Dropout追加
        )
        
        # Phase 1 改善: 強化されたDecoder
        self.decoder = nn.Sequential(
            # 第1層: [256, 1, 2, 2] -> [128, 2, 4, 4]
            nn.ConvTranspose3d(256, 128, kernel_size=(3, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(128, momentum=0.1),  # 強化BatchNorm
            nn.ReLU(inplace=True),
            nn.Dropout3d(0.15),  # Dropout追加
            
            # 第2層: [128, 2, 4, 4] -> [64, 4, 8, 8]
            nn.ConvTranspose3d(128, 64, kernel_size=(3, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(64, momentum=0.1),  # 強化BatchNorm
            nn.ReLU(inplace=True),
            nn.Dropout3d(0.15),  # Dropout追加
            
            # 第3層: [64, 4, 8, 8] -> [32, 8, 16, 16]
            nn.ConvTranspose3d(64, 32, kernel_size=(3, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(32, momentum=0.1),  # 強化BatchNorm
            nn.ReLU(inplace=True),
            nn.Dropout3d(0.1),  # Dropout追加
            
            # 第4層: [32, 8, 16, 16] -> [16, 15, 32, 32]
            nn.ConvTranspose3d(32, 16, kernel_size=(3, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1), output_padding=(1, 0, 0)),
            nn.BatchNorm3d(16, momentum=0.1),  # 強化BatchNorm
            nn.ReLU(inplace=True),
            nn.Dropout3d(0.1),  # Dropout追加
            
            # 第5層: [16, 15, 32, 32] -> [1, 30, 64, 64]
            nn.ConvTranspose3d(16, input_channels, kernel_size=(3, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1), output_padding=(0, 0, 0)),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        """エンコード"""
        encoded = self.encoder(x)
        latent = self.to_latent(encoded)
        return latent
    
    def decode(self, latent):
        """デコード"""
        # 潜在ベクトルから3D特徴マップに復元
        decoded = self.from_latent(latent)
        decoded = decoded.view(-1, 256, 1, 2, 2)
        
        # デコーダーで元の形状に復元
        output = self.decoder(decoded)
        
        # 最終的に正確なサイズにリサイズ（設定されたtarget_sizeに合わせる）
        target_h, target_w = self.target_size
        output = F.interpolate(output, size=(self.fixed_frames, target_h, target_w), mode='trilinear', align_corners=False)
        
        return output
    
    def forward(self, x):
        latent = self.encode(x)
        reconstructed = self.decode(latent)
        return reconstructed, latent

# Phase 1 改善: AdamW + CosineAnnealingLR を使用した訓練関数
def train_autoencoder_phase1(model, dataloader, num_epochs=100, learning_rate=1e-3, weight_decay=1e-4):
    """Phase 1改善版 Autoencoderの訓練"""
    model = model.to(device)
    criterion = nn.MSELoss()
    
    # Phase 1 改善: AdamW オプティマイザーを使用
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Phase 1 改善: CosineAnnealingLR スケジューラーを使用
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    
    model.train()
    losses = []
    
    print(f"Phase 1 Training: Latent dim={model.latent_dim}, AdamW, CosineAnnealingLR")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    for epoch in range(num_epochs):
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
            
            # Phase 1 改善: Gradient Clippingを追加
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # スケジューラーのステップ
        scheduler.step()
        
        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            current_lr = scheduler.get_last_lr()[0]
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}, LR: {current_lr:.2e}')
    
    return losses

# ================================
# 3. 潜在ベクトルの抽出とクラスタリング
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
    # 標準化
    scaler = StandardScaler()
    latent_scaled = scaler.fit_transform(latent_vectors)
    
    # K-meansクラスタリング
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(latent_scaled)
    
    # PCAで2次元に削減
    pca = PCA(n_components=2)
    latent_2d_pca = pca.fit_transform(latent_scaled)
    
    # t-SNEで2次元に削減
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(latent_vectors)-1))
    latent_2d_tsne = tsne.fit_transform(latent_scaled)
    
    return cluster_labels, latent_2d_pca, latent_2d_tsne, pca, tsne

# ================================
# 4. f-k平面へのマッピングとヒートマップ表示
# ================================

def visualize_results(f_values, k_values, cluster_labels, latent_2d_pca, latent_2d_tsne):
    """結果の可視化"""
    
    # 図の設定
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. f-k平面でのクラスタリング結果
    scatter1 = axes[0, 0].scatter(f_values, k_values, c=cluster_labels, cmap='viridis', alpha=0.7)
    axes[0, 0].set_xlabel('f parameter')
    axes[0, 0].set_ylabel('k parameter')
    axes[0, 0].set_title('Phase 1: Clustering Results in f-k Space')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].invert_yaxis()  # F軸（縦軸）を反転
    plt.colorbar(scatter1, ax=axes[0, 0], label='Cluster')
    
    # 2. PCA可視化
    scatter2 = axes[0, 1].scatter(latent_2d_pca[:, 0], latent_2d_pca[:, 1], c=cluster_labels, cmap='viridis', alpha=0.7)
    axes[0, 1].set_xlabel('PCA Component 1')
    axes[0, 1].set_ylabel('PCA Component 2')
    axes[0, 1].set_title('Phase 1: PCA Visualization (256D Latent)')
    axes[0, 1].grid(True, alpha=0.3)
    plt.colorbar(scatter2, ax=axes[0, 1], label='Cluster')
    
    # 3. t-SNE可視化
    scatter3 = axes[1, 0].scatter(latent_2d_tsne[:, 0], latent_2d_tsne[:, 1], c=cluster_labels, cmap='viridis', alpha=0.7)
    axes[1, 0].set_xlabel('t-SNE Component 1')
    axes[1, 0].set_ylabel('t-SNE Component 2')
    axes[1, 0].set_title('Phase 1: t-SNE Visualization (256D Latent)')
    axes[1, 0].grid(True, alpha=0.3)
    plt.colorbar(scatter3, ax=axes[1, 0], label='Cluster')
    
    # 4. f-k平面のヒートマップ
    # グリッドデータを作成
    f_unique = np.unique(f_values)
    k_unique = np.unique(k_values)
    
    # ヒートマップ用のデータを準備（fが縦軸、kが横軸）
    heatmap_data = np.full((len(f_unique), len(k_unique)), np.nan)
    
    for i, f in enumerate(f_values):
        k = k_values[i]
        cluster = cluster_labels[i]
        
        f_idx = np.where(f_unique == f)[0][0]
        k_idx = np.where(k_unique == k)[0][0]
        heatmap_data[f_idx, k_idx] = cluster
    
    im = axes[1, 1].imshow(heatmap_data, cmap='viridis', aspect=1, origin='upper')  # origin='upper'でf軸を反転
    axes[1, 1].set_xticks(range(len(k_unique)))
    axes[1, 1].set_yticks(range(len(f_unique)))
    axes[1, 1].set_xticklabels([f'{k:.4f}' for k in k_unique], rotation=45)
    axes[1, 1].set_yticklabels([f'{f:.4f}' for f in f_unique])
    axes[1, 1].set_xlabel('k parameter')
    axes[1, 1].set_ylabel('f parameter')
    axes[1, 1].set_title('Phase 1: Cluster Heatmap in f-k Space')
    axes[1, 1].invert_yaxis()  # F軸（縦軸）を確実に反転
    
    # ヒートマップ用のlegendを作成
    unique_clusters = np.unique(cluster_labels[~np.isnan(cluster_labels)])
    cmap = plt.cm.viridis
    legend_elements = [Patch(facecolor=cmap(cluster/len(unique_clusters)), 
                           label=f'Cluster {int(cluster)}') 
                     for cluster in unique_clusters]
    axes[1, 1].legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig('../results/gray_scott_clustering_results_phase1.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_detailed_heatmap(f_values, k_values, cluster_labels):
    """詳細なf-k平面ヒートマップの作成"""
    
    # データフレームを作成
    df = pd.DataFrame({
        'f': f_values,
        'k': k_values,
        'cluster': cluster_labels
    })
    
    # ピボットテーブルを作成（fが縦軸、kが横軸）
    pivot_table = df.pivot(index='f', columns='k', values='cluster')
    
    # ヒートマップを描画
    plt.figure(figsize=(12, 8))
    ax = sns.heatmap(pivot_table, cmap='viridis', annot=True, fmt='.0f', 
                     cbar=False, square=True)  # cbarをFalseに設定
    ax.invert_yaxis()  # F軸（縦軸）を反転
    plt.title('Phase 1: Gray-Scott Parameter Space Clustering\n(f-k plane, 256D Latent)')
    plt.xlabel('k parameter')
    plt.ylabel('f parameter')
    
    # legendを作成
    unique_clusters = np.unique(cluster_labels)
    cmap = plt.cm.viridis
    legend_elements = [Patch(facecolor=cmap(cluster/len(unique_clusters)), 
                           label=f'Cluster {int(cluster)}') 
                     for cluster in unique_clusters]
    plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig('../results/gray_scott_detailed_heatmap_phase1.png', dpi=300, bbox_inches='tight')
    plt.show()

# ================================
# メイン実行関数
# ================================

def main():
    """Phase 1 メイン実行関数"""
    
    # Phase 1 ハイパーパラメータ
    gif_folder = '../data/gif'
    fixed_frames = 30
    target_size = (64, 64)
    latent_dim = 256  # Phase 1: 64 -> 256
    num_epochs = 50
    batch_size = 4
    learning_rate = 1e-3
    weight_decay = 1e-4  # Phase 1: AdamW用
    n_clusters = 5
    
    print("=" * 60)
    print("Phase 1: Gray-Scott 3D CNN Autoencoder Analysis")
    print("=" * 60)
    print(f"Latent Dimension: {latent_dim} (4x increase from Phase 0)")
    print(f"Optimizer: AdamW with weight decay {weight_decay}")
    print(f"Scheduler: CosineAnnealingLR")
    print(f"Regularization: Enhanced BatchNorm + Dropout")
    print("=" * 60)
    
    # データセットの作成
    print("Loading dataset...")
    dataset = GrayScottDataset(gif_folder, fixed_frames, target_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    print(f"Dataset size: {len(dataset)} samples")
    print(f"Batch size: {batch_size}")
    
    # Phase 1 モデルの作成
    print("Creating Phase 1 model...")
    model = Conv3DAutoencoderPhase1(latent_dim=latent_dim, fixed_frames=fixed_frames, target_size=target_size)
    
    # モデル訓練
    print("Training Phase 1 model...")
    losses = train_autoencoder_phase1(model, dataloader, num_epochs, learning_rate, weight_decay)
    
    # 訓練曲線の可視化
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('Phase 1: Training Loss (256D Latent, AdamW)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig('../results/training_loss_phase1.png', dpi=300)
    plt.show()
    
    # モデル保存
    print("Saving Phase 1 model...")
    os.makedirs('../models', exist_ok=True)
    torch.save(model.state_dict(), '../models/trained_autoencoder_phase1.pth')
    
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
            'scheduler': 'CosineAnnealingLR'
        }
    }
    
    import pickle
    with open('../results/analysis_results_phase1.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    # 可視化
    print("Creating visualizations...")
    visualize_results(f_values, k_values, cluster_labels, latent_2d_pca, latent_2d_tsne)
    create_detailed_heatmap(f_values, k_values, cluster_labels)
    
    # クラスタリング品質評価
    from sklearn.metrics import silhouette_score
    silhouette_avg = silhouette_score(latent_vectors, cluster_labels)
    
    print("=" * 60)
    print("Phase 1 Results Summary:")
    print("=" * 60)
    print(f"Samples: {len(dataset)}")
    print(f"Latent Dimension: {latent_dim}")
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Final Training Loss: {losses[-1]:.6f}")
    print(f"Clusters: {n_clusters}")
    print(f"Silhouette Score: {silhouette_avg:.4f}")
    print("=" * 60)
    
    # 改善効果の分析
    print("Phase 1 Improvements Applied:")
    print("✓ Latent dimension: 64 → 256 (4x increase)")
    print("✓ Enhanced BatchNorm with momentum=0.1")
    print("✓ Dropout layers (0.1-0.2) for regularization")
    print("✓ AdamW optimizer with weight decay")
    print("✓ CosineAnnealingLR scheduler")
    print("✓ Gradient clipping (max_norm=1.0)")
    print("✓ Additional encoder layer for deeper representation")
    print("=" * 60)

if __name__ == "__main__":
    main() 