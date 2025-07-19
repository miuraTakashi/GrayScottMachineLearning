import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib.patches import Patch
import imageio.v2 as imageio
from PIL import Image
import re

# ======== Phase2の正しいモデル・データセット定義 ========
class SpatioTemporalAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.spatial_attention = nn.Sequential(
            nn.AdaptiveAvgPool3d((None, 1, 1)),
            nn.Conv3d(channels, channels//4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels//4, channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.temporal_attention = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, None, None)),
            nn.Conv3d(channels, channels//4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels//4, channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channels, channels//4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels//4, channels, kernel_size=1),
            nn.Sigmoid()
        )
    def forward(self, x):
        identity = x
        x = x * self.spatial_attention(x)
        x = x * self.temporal_attention(x)
        x = x * self.channel_attention(x)
        x = x + identity * 0.1
        return x

class ResidualAttentionBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels, momentum=0.1)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels, momentum=0.1)
        self.attention = SpatioTemporalAttention(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels, momentum=0.1)
            )
        self.dropout = nn.Dropout3d(0.1)
    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out = self.attention(out)
        out += self.shortcut(identity)
        out = F.relu(out)
        return out

class Conv3DAutoencoderPhase2(nn.Module):
    def __init__(self, input_channels=1, fixed_frames=30, target_size=(64, 64), latent_dim=256):
        super().__init__()
        self.latent_dim = latent_dim
        self.fixed_frames = fixed_frames
        self.target_size = target_size
        self.initial_conv = nn.Sequential(
            nn.Conv3d(input_channels, 32, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3)),
            nn.BatchNorm3d(32, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Dropout3d(0.05)
        )
        self.res_block1 = ResidualAttentionBlock3D(32, 64, stride=(2, 2, 2))
        self.res_block2 = ResidualAttentionBlock3D(64, 64)
        self.res_block3 = ResidualAttentionBlock3D(64, 128, stride=(2, 2, 2))
        self.res_block4 = ResidualAttentionBlock3D(128, 128)
        self.res_block5 = ResidualAttentionBlock3D(128, 256, stride=(2, 2, 2))
        self.res_block6 = ResidualAttentionBlock3D(256, 256)
        self.global_pool = nn.AdaptiveAvgPool3d((2, 2, 2))
        self.dropout_before_latent = nn.Dropout3d(0.3)
        self.to_latent = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 2 * 2 * 2, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, latent_dim),
            nn.BatchNorm1d(latent_dim)
        )
        self.from_latent = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, 256 * 2 * 2 * 2),
            nn.ReLU(inplace=True)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(256, 128, kernel_size=(3, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(128, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Dropout3d(0.2),
            nn.ConvTranspose3d(128, 64, kernel_size=(3, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(64, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Dropout3d(0.15),
            nn.ConvTranspose3d(64, 32, kernel_size=(3, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(32, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Dropout3d(0.1),
            nn.ConvTranspose3d(32, input_channels, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3)),
            nn.Sigmoid()
        )
    def encode(self, x):
        x = self.initial_conv(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.res_block4(x)
        x = self.res_block5(x)
        x = self.res_block6(x)
        x = self.global_pool(x)
        x = self.dropout_before_latent(x)
        latent = self.to_latent(x)
        return latent
    def decode(self, latent):
        x = self.from_latent(latent)
        x = x.view(-1, 256, 2, 2, 2)
        x = self.decoder(x)
        target_h, target_w = self.target_size
        x = F.interpolate(x, size=(self.fixed_frames, target_h, target_w), mode='trilinear', align_corners=False)
        return x
    def forward(self, x):
        latent = self.encode(x)
        reconstructed = self.decode(latent)
        return reconstructed, latent

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

# ======== 潜在ベクトル抽出・クラスタリング・可視化 ========
def extract_latent_vectors(model, dataloader, device):
    model.eval()
    latents = []
    f_values = []
    k_values = []
    filenames = []
    with torch.no_grad():
        for batch in dataloader:
            x = batch['tensor'].to(device)
            reconstructed, latent = model(x)
            latents.append(latent.cpu().numpy())
            f_values.extend(batch['f_value'])
            k_values.extend(batch['k_value'])
            filenames.extend(batch['filename'])
    latents = np.vstack(latents)
    return latents, np.array(f_values), np.array(k_values), filenames

def perform_clustering(latent_vectors, n_clusters=5):
    scaler = StandardScaler()
    latent_scaled = scaler.fit_transform(latent_vectors)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(latent_scaled)
    pca = PCA(n_components=2)
    latent_2d_pca = pca.fit_transform(latent_scaled)
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(latent_vectors)-1))
    latent_2d_tsne = tsne.fit_transform(latent_scaled)
    return cluster_labels, latent_2d_pca, latent_2d_tsne, pca, tsne

def visualize_results_phase2(f_values, k_values, cluster_labels, latent_2d_pca, latent_2d_tsne):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Phase 2: ResNet + Attention Results', fontsize=16, fontweight='bold')
    scatter1 = axes[0, 0].scatter(f_values, k_values, c=cluster_labels, cmap='viridis', alpha=0.7)
    axes[0, 0].set_xlabel('f parameter')
    axes[0, 0].set_ylabel('k parameter')
    axes[0, 0].set_title('Phase 2: Clustering in f-k Space\n(ResNet + Attention)')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].invert_yaxis()
    plt.colorbar(scatter1, ax=axes[0, 0], label='Cluster')
    scatter2 = axes[0, 1].scatter(latent_2d_pca[:, 0], latent_2d_pca[:, 1], c=cluster_labels, cmap='viridis', alpha=0.7)
    axes[0, 1].set_xlabel('PCA Component 1')
    axes[0, 1].set_ylabel('PCA Component 2')
    axes[0, 1].set_title('Phase 2: PCA Visualization\n(256D Latent + ResNet)')
    axes[0, 1].grid(True, alpha=0.3)
    plt.colorbar(scatter2, ax=axes[0, 1], label='Cluster')
    scatter3 = axes[1, 0].scatter(latent_2d_tsne[:, 0], latent_2d_tsne[:, 1], c=cluster_labels, cmap='viridis', alpha=0.7)
    axes[1, 0].set_xlabel('t-SNE Component 1')
    axes[1, 0].set_ylabel('t-SNE Component 2')
    axes[1, 0].set_title('Phase 2: t-SNE Visualization\n(Attention Enhanced)')
    axes[1, 0].grid(True, alpha=0.3)
    plt.colorbar(scatter3, ax=axes[1, 0], label='Cluster')
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
    unique_clusters = np.unique(cluster_labels[~np.isnan(cluster_labels)])
    cmap = plt.cm.viridis
    legend_elements = [Patch(facecolor=cmap(cluster/len(unique_clusters)), label=f'Cluster {int(cluster)}') for cluster in unique_clusters]
    axes[1, 1].legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('results/gray_scott_clustering_results_phase2_only.png', dpi=300, bbox_inches='tight')
    plt.show()

# ======== メイン処理 ========
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
    print('データセット読み込み中...')
    dataset = GrayScottDataset(gif_folder, fixed_frames, target_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    print(f'サンプル数: {len(dataset)}')
    print('学習済みモデルをロード中...')
    model = Conv3DAutoencoderPhase2(fixed_frames=fixed_frames, target_size=target_size, latent_dim=latent_dim)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    print('潜在ベクトル抽出中...')
    latents, f_values, k_values, filenames = extract_latent_vectors(model, dataloader, device)
    print(f'潜在ベクトル shape: {latents.shape}')
    print('クラスタリング...')
    cluster_labels, latent_2d_pca, latent_2d_tsne, pca, tsne = perform_clustering(latents, n_clusters)
    print('可視化...')
    visualize_results_phase2(f_values, k_values, cluster_labels, latent_2d_pca, latent_2d_tsne)

if __name__ == '__main__':
    main() 