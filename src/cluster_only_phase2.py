#!/usr/bin/env python3
"""
Phase 2: 学習済みモデルを使ったクラスタリング・可視化専用スクリプト
"""
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from PIL import Image
import imageio.v2 as imageio
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ========== データセット定義（Phase2と同じ） ==========
class GrayScottDataset(Dataset):
    def __init__(self, gif_folder, fixed_frames=30, target_size=(64, 64), frame_range=None):
        self.gif_folder = gif_folder
        self.fixed_frames = fixed_frames
        self.target_size = target_size
        self.frame_range = frame_range
        self.gif_files = []
        self.f_values = []
        self.k_values = []
        self.tensors = []
        self._load_data()

    def _parse_filename(self, filename):
        pattern = r'GrayScott-f([0-9.]+)-k([0-9.]+)-\d+\.gif'
        match = re.match(pattern, filename)
        if match:
            f_val = float(match.group(1))
            k_val = float(match.group(2))
            return f_val, k_val
        return None, None

    def _load_gif_as_tensor(self, gif_path):
        try:
            gif = imageio.mimread(gif_path)
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
            tensor = tensor.unsqueeze(0)
            return tensor
        except Exception as e:
            print(f"Error loading {gif_path}: {e}")
            return None

    def _load_data(self):
        gif_files = [f for f in os.listdir(self.gif_folder) if f.endswith('.gif')]
        for filename in gif_files:
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
    def __len__(self):
        return len(self.tensors)
    def __getitem__(self, idx):
        return {
            'tensor': self.tensors[idx],
            'f_value': self.f_values[idx],
            'k_value': self.k_values[idx],
            'filename': self.gif_files[idx]
        }

# ========== モデル定義（Phase2と同じ構造） ==========
class Conv3DAutoencoderPhase2(nn.Module):
    def __init__(self, input_channels=1, fixed_frames=30, target_size=(64, 64), latent_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(input_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * (fixed_frames // 8) * (target_size[0] // 8) * (target_size[1] // 8), latent_dim),
            nn.ReLU()
        )
    def encode(self, x):
        return self.encoder(x)
    def forward(self, x):
        return self.encode(x)

# ========== 潜在ベクトル抽出 ==========
def extract_latent_vectors(model, dataloader, device):
    model.eval()
    latents = []
    f_values = []
    k_values = []
    filenames = []
    with torch.no_grad():
        for batch in dataloader:
            x = batch['tensor'].to(device)
            latent = model(x)
            latents.append(latent.cpu().numpy())
            f_values.extend(batch['f_value'])
            k_values.extend(batch['k_value'])
            filenames.extend(batch['filename'])
    latents = np.vstack(latents)
    return latents, np.array(f_values), np.array(k_values), filenames

# ========== クラスタリング & 可視化 ==========
def perform_clustering_and_visualize(latent_vectors, f_values, k_values, n_clusters=5, out_prefix='phase2_cluster_only'):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(latent_vectors)
    # PCA
    pca = PCA(n_components=2)
    latent_2d_pca = pca.fit_transform(latent_vectors)
    # t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    latent_2d_tsne = tsne.fit_transform(latent_vectors)
    # 可視化
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    scatter1 = axes[0].scatter(f_values, k_values, c=cluster_labels, cmap='viridis', alpha=0.7)
    axes[0].set_xlabel('f parameter')
    axes[0].set_ylabel('k parameter')
    axes[0].set_title('Clustering in f-k Space')
    axes[0].invert_yaxis()
    plt.colorbar(scatter1, ax=axes[0], label='Cluster')
    scatter2 = axes[1].scatter(latent_2d_pca[:, 0], latent_2d_pca[:, 1], c=cluster_labels, cmap='viridis', alpha=0.7)
    axes[1].set_title('PCA Visualization')
    plt.colorbar(scatter2, ax=axes[1], label='Cluster')
    scatter3 = axes[2].scatter(latent_2d_tsne[:, 0], latent_2d_tsne[:, 1], c=cluster_labels, cmap='viridis', alpha=0.7)
    axes[2].set_title('t-SNE Visualization')
    plt.colorbar(scatter3, ax=axes[2], label='Cluster')
    plt.tight_layout()
    plt.savefig(f'results/{out_prefix}_visualization.png', dpi=300)
    plt.show()
    # CSV保存
    df = pd.DataFrame({
        'f_value': f_values,
        'k_value': k_values,
        'cluster': cluster_labels
    })
    df.to_csv(f'results/{out_prefix}_clustering_results.csv', index=False)
    print(f'クラスタリング結果を results/{out_prefix}_clustering_results.csv に保存しました')

# ========== メイン ==========
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    gif_folder = '/Users/nakashimarikuto/Gray scott/GrayScottMachineLearning/data/gif'
    fixed_frames = 30
    target_size = (64, 64)
    latent_dim = 256
    batch_size = 4
    n_clusters = 5
    model_path = 'models/trained_autoencoder_phase2.pth'
    # データセット
    print('データセット読み込み中...')
    dataset = GrayScottDataset(gif_folder, fixed_frames, target_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    print(f'サンプル数: {len(dataset)}')
    # モデルロード
    print('学習済みモデルをロード中...')
    model = Conv3DAutoencoderPhase2(fixed_frames=fixed_frames, target_size=target_size, latent_dim=latent_dim)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    # 潜在ベクトル抽出
    print('潜在ベクトル抽出中...')
    latents, f_values, k_values, filenames = extract_latent_vectors(model, dataloader, device)
    print(f'潜在ベクトル shape: {latents.shape}')
    # クラスタリング＆可視化
    print('クラスタリング＆可視化...')
    perform_clustering_and_visualize(latents, f_values, k_values, n_clusters=n_clusters)

if __name__ == '__main__':
    main() 