#!/usr/bin/env python3
"""
Gray-Scott 3D CNN Autoencoder - Phase 4: 対比学習・評価改善
目標: Phase 3 (0.5144) → Phase 4 (0.55+) への更なる向上

主要改善点:
- 対比学習（Contrastive Learning）
- 階層的クラスタリング分析
- 包括的評価指標
- 改善された訓練ループ
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import imageio.v2 as imageio
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import time
import pickle
from typing import List, Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# GPU設定
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Phase 4 Using device: {device}")

# ================================
# 1. 対比学習システム
# ================================

class ContrastiveLoss(nn.Module):
    """f-kパラメータ類似性に基づく対比学習損失"""
    
    def __init__(self, temperature=0.5, margin=1.0):
        super().__init__()
        self.temperature = temperature
        self.margin = margin
        self.cosine_similarity = nn.CosineSimilarity(dim=2)
    
    def forward(self, features, f_params, k_params):
        """
        Args:
            features: (batch_size, feature_dim) - 潜在表現
            f_params: (batch_size,) - fパラメータ
            k_params: (batch_size,) - kパラメータ
        """
        batch_size = features.size(0)
        
        # f-kパラメータ空間での類似性計算
        f_diff = torch.abs(f_params.unsqueeze(1) - f_params.unsqueeze(0))
        k_diff = torch.abs(k_params.unsqueeze(1) - k_params.unsqueeze(0))
        
        # 正規化されたパラメータ距離
        param_distance = torch.sqrt(f_diff**2 + k_diff**2)
        
        # 類似性閾値（近いパラメータは類似、遠いパラメータは非類似）
        similarity_threshold = 0.01  # f-k空間での閾値
        positive_mask = param_distance < similarity_threshold
        negative_mask = param_distance > similarity_threshold * 3
        
        # 特徴量の類似性計算
        features_norm = F.normalize(features, p=2, dim=1)
        similarity_matrix = torch.mm(features_norm, features_norm.t()) / self.temperature
        
        # 対比学習損失
        positive_loss = 0
        negative_loss = 0
        
        if positive_mask.sum() > 0:
            positive_sim = similarity_matrix[positive_mask]
            positive_loss = -torch.log(torch.exp(positive_sim).sum() / torch.exp(similarity_matrix).sum())
        
        if negative_mask.sum() > 0:
            negative_sim = similarity_matrix[negative_mask]
            negative_loss = torch.log(torch.exp(negative_sim).sum() / torch.exp(similarity_matrix).sum())
        
        contrastive_loss = positive_loss + negative_loss
        
        return contrastive_loss

class ProjectionHead(nn.Module):
    """対比学習用射影ヘッド"""
    
    def __init__(self, input_dim=512, hidden_dim=256, output_dim=128):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim)
        )
    
    def forward(self, x):
        return self.projection(x)

# ================================
# 2. 階層的クラスタリング分析
# ================================

class HierarchicalClusteringAnalysis:
    """階層的クラスタリング分析システム"""
    
    def __init__(self, method='ward', metric='euclidean'):
        self.method = method
        self.metric = metric
        self.linkage_matrix = None
        self.optimal_clusters = None
    
    def fit(self, features):
        """階層的クラスタリング実行"""
        # 特徴量の標準化
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # 階層的クラスタリング
        self.linkage_matrix = linkage(features_scaled, method=self.method, metric=self.metric)
        
        # 最適クラスタ数の決定
        self.optimal_clusters = self._find_optimal_clusters(features_scaled)
        
        return self
    
    def _find_optimal_clusters(self, features, max_clusters=20):
        """最適クラスタ数の自動決定"""
        silhouette_scores = []
        cluster_range = range(2, min(max_clusters + 1, len(features) // 2))
        
        for n_clusters in cluster_range:
            cluster_labels = fcluster(self.linkage_matrix, n_clusters, criterion='maxclust')
            
            if len(np.unique(cluster_labels)) > 1:
                silhouette_avg = silhouette_score(features, cluster_labels)
                silhouette_scores.append(silhouette_avg)
            else:
                silhouette_scores.append(-1)
        
        if silhouette_scores:
            optimal_idx = np.argmax(silhouette_scores)
            optimal_n_clusters = cluster_range[optimal_idx]
            return optimal_n_clusters
        else:
            return 2
    
    def get_cluster_labels(self, n_clusters=None):
        """クラスタラベルの取得"""
        if n_clusters is None:
            n_clusters = self.optimal_clusters
        
        return fcluster(self.linkage_matrix, n_clusters, criterion='maxclust')
    
    def plot_dendrogram(self, figsize=(12, 8)):
        """デンドログラム可視化"""
        plt.figure(figsize=figsize)
        dendrogram(self.linkage_matrix, truncate_mode='level', p=10)
        plt.title('Hierarchical Clustering Dendrogram')
        plt.xlabel('Sample Index')
        plt.ylabel('Distance')
        plt.show()

# ================================
# 3. 包括的評価指標
# ================================

class ComprehensiveEvaluationMetrics:
    """包括的評価指標システム"""
    
    def __init__(self):
        self.metrics = {}
    
    def calculate_all_metrics(self, features, labels, f_params=None, k_params=None):
        """全ての評価指標を計算"""
        
        # 基本クラスタリング指標
        self.metrics['silhouette_score'] = silhouette_score(features, labels)
        self.metrics['calinski_harabasz_score'] = calinski_harabasz_score(features, labels)
        self.metrics['davies_bouldin_score'] = davies_bouldin_score(features, labels)
        
        # 近傍一致度指標
        self.metrics['neighborhood_agreement'] = self._calculate_neighborhood_agreement(features, labels)
        
        # パラメータ空間分離度評価
        if f_params is not None and k_params is not None:
            self.metrics['parameter_separation'] = self._calculate_parameter_separation(
                features, labels, f_params, k_params
            )
        
        # クラスタ内分散・クラスタ間分散
        self.metrics['within_cluster_variance'] = self._calculate_within_cluster_variance(features, labels)
        self.metrics['between_cluster_variance'] = self._calculate_between_cluster_variance(features, labels)
        
        # 安定性指標
        self.metrics['cluster_stability'] = self._calculate_cluster_stability(features, labels)
        
        return self.metrics
    
    def _calculate_neighborhood_agreement(self, features, labels, k=10):
        """近傍一致度の計算"""
        from sklearn.neighbors import NearestNeighbors
        
        nbrs = NearestNeighbors(n_neighbors=k+1).fit(features)
        distances, indices = nbrs.kneighbors(features)
        
        agreements = []
        for i in range(len(features)):
            neighbor_labels = labels[indices[i][1:]]  # 自分以外の近傍
            same_cluster = np.sum(neighbor_labels == labels[i])
            agreement = same_cluster / k
            agreements.append(agreement)
        
        return np.mean(agreements)
    
    def _calculate_parameter_separation(self, features, labels, f_params, k_params):
        """パラメータ空間分離度の計算"""
        unique_labels = np.unique(labels)
        separations = []
        
        for label in unique_labels:
            mask = labels == label
            if np.sum(mask) > 1:
                cluster_f = f_params[mask]
                cluster_k = k_params[mask]
                
                # クラスタ内のパラメータ分散
                f_var = np.var(cluster_f)
                k_var = np.var(cluster_k)
                cluster_variance = f_var + k_var
                
                separations.append(cluster_variance)
        
        return np.mean(separations) if separations else 0
    
    def _calculate_within_cluster_variance(self, features, labels):
        """クラスタ内分散の計算"""
        unique_labels = np.unique(labels)
        within_variances = []
        
        for label in unique_labels:
            mask = labels == label
            if np.sum(mask) > 1:
                cluster_features = features[mask]
                centroid = np.mean(cluster_features, axis=0)
                variance = np.mean(np.sum((cluster_features - centroid)**2, axis=1))
                within_variances.append(variance)
        
        return np.mean(within_variances) if within_variances else 0
    
    def _calculate_between_cluster_variance(self, features, labels):
        """クラスタ間分散の計算"""
        unique_labels = np.unique(labels)
        centroids = []
        
        for label in unique_labels:
            mask = labels == label
            centroid = np.mean(features[mask], axis=0)
            centroids.append(centroid)
        
        centroids = np.array(centroids)
        overall_centroid = np.mean(centroids, axis=0)
        
        between_variance = np.mean(np.sum((centroids - overall_centroid)**2, axis=1))
        return between_variance
    
    def _calculate_cluster_stability(self, features, labels, n_bootstrap=10):
        """クラスタ安定性の計算"""
        from sklearn.utils import resample
        
        original_labels = labels
        stability_scores = []
        
        for _ in range(n_bootstrap):
            # ブートストラップサンプリング
            bootstrap_indices = resample(range(len(features)), n_samples=len(features))
            bootstrap_features = features[bootstrap_indices]
            bootstrap_labels = labels[bootstrap_indices]
            
            # クラスタリング実行
            kmeans = KMeans(n_clusters=len(np.unique(original_labels)), random_state=42)
            new_labels = kmeans.fit_predict(bootstrap_features)
            
            # ラベル一致度計算（ハンガリアンアルゴリズム簡易版）
            agreement = self._calculate_label_agreement(bootstrap_labels, new_labels)
            stability_scores.append(agreement)
        
        return np.mean(stability_scores)
    
    def _calculate_label_agreement(self, labels1, labels2):
        """ラベル一致度の計算"""
        from scipy.optimize import linear_sum_assignment
        
        unique_labels1 = np.unique(labels1)
        unique_labels2 = np.unique(labels2)
        
        # 混同行列作成
        confusion_matrix = np.zeros((len(unique_labels1), len(unique_labels2)))
        
        for i, label1 in enumerate(unique_labels1):
            for j, label2 in enumerate(unique_labels2):
                confusion_matrix[i, j] = np.sum((labels1 == label1) & (labels2 == label2))
        
        # ハンガリアンアルゴリズムで最適割り当て
        row_ind, col_ind = linear_sum_assignment(-confusion_matrix)
        
        # 一致度計算
        agreement = confusion_matrix[row_ind, col_ind].sum() / len(labels1)
        return agreement
    
    def print_metrics(self):
        """評価指標の表示"""
        print("\n🎯 Phase 4 包括的評価指標")
        print("=" * 50)
        
        # 基本指標
        print(f"Silhouette Score: {self.metrics.get('silhouette_score', 0):.4f}")
        print(f"Calinski-Harabasz: {self.metrics.get('calinski_harabasz_score', 0):.2f}")
        print(f"Davies-Bouldin: {self.metrics.get('davies_bouldin_score', 0):.4f}")
        
        # 高度な指標
        print(f"Neighborhood Agreement: {self.metrics.get('neighborhood_agreement', 0):.4f}")
        print(f"Parameter Separation: {self.metrics.get('parameter_separation', 0):.4f}")
        print(f"Within Cluster Variance: {self.metrics.get('within_cluster_variance', 0):.4f}")
        print(f"Between Cluster Variance: {self.metrics.get('between_cluster_variance', 0):.4f}")
        print(f"Cluster Stability: {self.metrics.get('cluster_stability', 0):.4f}")

# ================================
# 4. Phase 3ベースのモデル拡張
# ================================

class GrayScottAugmentation:
    """Gray-Scott専用データ拡張クラス（Phase 3から継承）"""
    
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
                tensor = torch.roll(tensor, shift, dims=1)
        return tensor
    
    def spatial_flip(self, tensor):
        """空間軸反転"""
        if np.random.random() < self.spatial_flip_prob:
            if np.random.random() < 0.5:
                tensor = torch.flip(tensor, dims=[3])
            if np.random.random() < 0.5:
                tensor = torch.flip(tensor, dims=[2])
        return tensor
    
    def add_noise(self, tensor, noise_std=0.02):
        """ガウシアンノイズ追加"""
        if np.random.random() < self.noise_prob:
            noise = torch.randn_like(tensor) * noise_std
            tensor = torch.clamp(tensor + noise, 0, 1)
        return tensor
    
    def intensity_transform(self, tensor, gamma_range=(0.8, 1.2)):
        """強度変換"""
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

class MultiScaleFeatureFusion(nn.Module):
    """マルチスケール特徴融合（Phase 3から継承）"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        # 4つの異なるスケールでの特徴抽出
        self.scale1 = nn.Conv3d(in_channels, out_channels // 4, kernel_size=1, padding=0)  # Point-wise
        self.scale2 = nn.Conv3d(in_channels, out_channels // 4, kernel_size=3, padding=1)  # Local
        self.scale3 = nn.Conv3d(in_channels, out_channels // 4, kernel_size=5, padding=2)  # Global
        
        # プーリングベースの特徴
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.scale4 = nn.Conv3d(in_channels, out_channels // 4, kernel_size=1)
        
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        # 各スケールで特徴抽出
        feat1 = self.scale1(x)
        feat2 = self.scale2(x)
        feat3 = self.scale3(x)
        
        # プーリング特徴
        pooled = self.pool(x)
        feat4 = self.scale4(pooled)
        feat4 = feat4.expand_as(feat1)
        
        # 特徴融合
        fused = torch.cat([feat1, feat2, feat3, feat4], dim=1)
        fused = self.bn(fused)
        fused = self.relu(fused)
        
        return fused

class EnhancedSpatioTemporalAttention(nn.Module):
    """改良時空間注意機構（Phase 3から継承）"""
    
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        
        # 時間注意
        self.temporal_attention = nn.Sequential(
            nn.AdaptiveAvgPool3d((None, 1, 1)),
            nn.Conv3d(in_channels, in_channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels // reduction, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 空間注意
        self.spatial_attention = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, None, None)),
            nn.Conv3d(in_channels, in_channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels // reduction, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        # チャネル注意
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(in_channels, in_channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels // reduction, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # 各注意機構を適用
        temp_att = self.temporal_attention(x)
        spat_att = self.spatial_attention(x)
        chan_att = self.channel_attention(x)
        
        # 注意重みを適用
        x = x * temp_att * spat_att * chan_att
        
        return x

class ResidualMultiScaleBlock3D(nn.Module):
    """残差マルチスケールブロック（Phase 3から継承）"""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        self.multi_scale = MultiScaleFeatureFusion(in_channels, out_channels)
        self.attention = EnhancedSpatioTemporalAttention(out_channels)
        
        # 残差接続用
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )
    
    def forward(self, x):
        residual = self.shortcut(x)
        
        out = self.multi_scale(x)
        out = self.attention(out)
        
        out += residual
        return out

class Conv3DAutoencoderPhase4(nn.Module):
    """Phase 4: 対比学習統合オートエンコーダー"""
    
    def __init__(self, latent_dim=512, input_shape=(20, 64, 64)):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.input_shape = input_shape
        
        # エンコーダー
        self.encoder = nn.Sequential(
            # 入力: (1, 20, 64, 64)
            ResidualMultiScaleBlock3D(1, 32),
            nn.MaxPool3d(2),  # (32, 10, 32, 32)
            
            ResidualMultiScaleBlock3D(32, 64),
            nn.MaxPool3d(2),  # (64, 5, 16, 16)
            
            ResidualMultiScaleBlock3D(64, 128),
            nn.MaxPool3d(2),  # (128, 2, 8, 8)
            
            ResidualMultiScaleBlock3D(128, 256),
            nn.AdaptiveAvgPool3d(1),  # (256, 1, 1, 1)
        )
        
        # 潜在空間
        self.fc_encoder = nn.Sequential(
            nn.Linear(256, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        
        # 対比学習用射影ヘッド
        self.projection_head = ProjectionHead(latent_dim, 256, 128)
        
        # デコーダー
        self.fc_decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        
        # デコーダーを個別のレイヤーとして定義（サイズ調整付き）
        self.decoder_conv1 = nn.ConvTranspose3d(256, 128, kernel_size=(3, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1))
        self.decoder_bn1 = nn.BatchNorm3d(128)
        
        self.decoder_conv2 = nn.ConvTranspose3d(128, 64, kernel_size=(3, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1))
        self.decoder_bn2 = nn.BatchNorm3d(64)
        
        self.decoder_conv3 = nn.ConvTranspose3d(64, 32, kernel_size=(3, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1))
        self.decoder_bn3 = nn.BatchNorm3d(32)
        
        self.decoder_conv4 = nn.ConvTranspose3d(32, 1, kernel_size=(3, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1))
        
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
    
    def encode(self, x):
        """エンコード"""
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        latent = self.fc_encoder(x)
        return latent
    
    def decode(self, latent):
        """デコード（サイズ調整付き）"""
        x = self.fc_decoder(latent)
        x = x.view(x.size(0), 256, 1, 1, 1)
        
        # 段階的にデコード（各ステップでサイズを調整）
        # (256, 1, 1, 1) -> (128, 2, 8, 8)
        x = self.decoder_conv1(x)
        x = self.decoder_bn1(x)
        x = self.relu(x)
        # サイズ調整
        x = F.interpolate(x, size=(2, 8, 8), mode='trilinear', align_corners=False)
        
        # (128, 2, 8, 8) -> (64, 5, 16, 16)
        x = self.decoder_conv2(x)
        x = self.decoder_bn2(x)
        x = self.relu(x)
        # サイズ調整
        x = F.interpolate(x, size=(5, 16, 16), mode='trilinear', align_corners=False)
        
        # (64, 5, 16, 16) -> (32, 10, 32, 32)
        x = self.decoder_conv3(x)
        x = self.decoder_bn3(x)
        x = self.relu(x)
        # サイズ調整
        x = F.interpolate(x, size=(10, 32, 32), mode='trilinear', align_corners=False)
        
        # (32, 10, 32, 32) -> (1, 20, 64, 64)
        x = self.decoder_conv4(x)
        # 最終サイズ調整
        x = F.interpolate(x, size=(20, 64, 64), mode='trilinear', align_corners=False)
        x = self.sigmoid(x)
        
        return x
    
    def forward(self, x):
        """フォワードパス"""
        latent = self.encode(x)
        reconstructed = self.decode(latent)
        projection = self.projection_head(latent)
        
        return reconstructed, latent, projection

# ================================
# 5. データセットクラス
# ================================

class GrayScottDataset(Dataset):
    """Gray-Scott データセット（Phase 3から継承・拡張）"""
    
    def __init__(self, gif_folder, augmentation=None, max_samples=None):
        self.gif_folder = gif_folder
        self.augmentation = augmentation
        
        # GIFファイルリスト取得
        self.gif_files = [f for f in os.listdir(gif_folder) if f.endswith('.gif')]
        
        if max_samples:
            self.gif_files = self.gif_files[:max_samples]
        
        # f-kパラメータ抽出
        self.f_params = []
        self.k_params = []
        
        for gif_file in self.gif_files:
            f_val, k_val = self.extract_parameters(gif_file)
            self.f_params.append(f_val)
            self.k_params.append(k_val)
        
        self.f_params = np.array(self.f_params)
        self.k_params = np.array(self.k_params)
        
        print(f"📊 Dataset loaded: {len(self.gif_files)} samples")
        print(f"f range: {self.f_params.min():.4f} - {self.f_params.max():.4f}")
        print(f"k range: {self.k_params.min():.4f} - {self.k_params.max():.4f}")
    
    def extract_parameters(self, filename):
        """ファイル名からf-kパラメータを抽出"""
        pattern = r'f([\d.]+)-k([\d.]+)'
        match = re.search(pattern, filename)
        
        if match:
            f_val = float(match.group(1))
            k_val = float(match.group(2))
            return f_val, k_val
        else:
            return 0.0, 0.0
    
    def __len__(self):
        return len(self.gif_files)
    
    def __getitem__(self, idx):
        gif_path = os.path.join(self.gif_folder, self.gif_files[idx])
        
        # GIF読み込み
        gif = imageio.mimread(gif_path)
        
        # 最初の20フレームを取得
        frames = gif[:20] if len(gif) >= 20 else gif
        
        # テンソルに変換
        tensor = torch.FloatTensor(frames).unsqueeze(0)  # (1, T, H, W)
        tensor = tensor / 255.0  # 正規化
        
        # データ拡張
        if self.augmentation:
            tensor = self.augmentation(tensor)
        
        return tensor, self.f_params[idx], self.k_params[idx], idx

# ================================
# 6. 改善された訓練ループ
# ================================

def train_phase4_model(model, dataloader, num_epochs=30, learning_rate=1e-4):
    """Phase 4モデルの訓練"""
    
    # 最適化器
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # 損失関数
    reconstruction_loss = nn.MSELoss()
    contrastive_loss = ContrastiveLoss(temperature=0.5)
    
    # 訓練ループ
    model.train()
    train_losses = []
    
    print("🚀 Phase 4 Training Started")
    print("=" * 50)
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_contrastive_loss = 0.0
        
        for batch_idx, (data, f_params, k_params, _) in enumerate(dataloader):
            data = data.to(device)
            f_params = f_params.to(device)
            k_params = k_params.to(device)
            
            optimizer.zero_grad()
            
            # フォワードパス
            reconstructed, latent, projection = model(data)
            
            # 損失計算
            recon_loss = reconstruction_loss(reconstructed, data)
            contrast_loss = contrastive_loss(projection, f_params, k_params)
            
            # 総損失（重み付き）
            total_loss = recon_loss + 0.1 * contrast_loss
            
            # バックプロパゲーション
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += total_loss.item()
            epoch_recon_loss += recon_loss.item()
            epoch_contrastive_loss += contrast_loss.item()
        
        scheduler.step()
        
        avg_loss = epoch_loss / len(dataloader)
        avg_recon_loss = epoch_recon_loss / len(dataloader)
        avg_contrast_loss = epoch_contrastive_loss / len(dataloader)
        
        train_losses.append(avg_loss)
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}]")
            print(f"  Total Loss: {avg_loss:.6f}")
            print(f"  Reconstruction: {avg_recon_loss:.6f}")
            print(f"  Contrastive: {avg_contrast_loss:.6f}")
            print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")
    
    print("✅ Phase 4 Training Completed!")
    return train_losses

# ================================
# 7. メイン実行関数
# ================================

def main():
    """Phase 4メイン実行"""
    
    # データフォルダ設定
    GIF_FOLDER = "path/to/gif/folder"  # 実際のパスに変更
    
    if not os.path.exists(GIF_FOLDER):
        print(f"❌ GIF folder not found: {GIF_FOLDER}")
        return
    
    # データ拡張設定
    augmentation = GrayScottAugmentation()
    
    # データセット作成
    dataset = GrayScottDataset(GIF_FOLDER, augmentation=augmentation)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)
    
    # モデル作成
    model = Conv3DAutoencoderPhase4(latent_dim=512).to(device)
    
    print(f"📊 Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 訓練実行
    train_losses = train_phase4_model(model, dataloader, num_epochs=30)
    
    # モデル保存
    torch.save(model.state_dict(), 'models/phase4_model.pth')
    print("💾 Model saved to models/phase4_model.pth")
    
    # 評価実行
    evaluate_phase4_model(model, dataloader)

def evaluate_phase4_model(model, dataloader):
    """Phase 4モデルの評価"""
    
    model.eval()
    all_latents = []
    all_f_params = []
    all_k_params = []
    
    print("🔍 Phase 4 Evaluation Started")
    
    with torch.no_grad():
        for data, f_params, k_params, _ in dataloader:
            data = data.to(device)
            _, latent, _ = model(data)
            
            all_latents.append(latent.cpu().numpy())
            all_f_params.append(f_params.numpy())
            all_k_params.append(k_params.numpy())
    
    # データ統合
    all_latents = np.vstack(all_latents)
    all_f_params = np.concatenate(all_f_params)
    all_k_params = np.concatenate(all_k_params)
    
    # 階層的クラスタリング
    hierarchical_clustering = HierarchicalClusteringAnalysis()
    hierarchical_clustering.fit(all_latents)
    
    # クラスタラベル取得
    cluster_labels = hierarchical_clustering.get_cluster_labels()
    
    # 包括的評価
    evaluator = ComprehensiveEvaluationMetrics()
    metrics = evaluator.calculate_all_metrics(
        all_latents, cluster_labels, all_f_params, all_k_params
    )
    
    # 結果表示
    evaluator.print_metrics()
    
    # 可視化
    visualize_phase4_results(all_latents, cluster_labels, all_f_params, all_k_params)
    
    return metrics

def visualize_phase4_results(latents, labels, f_params, k_params):
    """Phase 4結果の可視化"""
    
    # PCA
    pca = PCA(n_components=2)
    latents_pca = pca.fit_transform(latents)
    
    # t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(latents)//4))
    latents_tsne = tsne.fit_transform(latents)
    
    # 可視化
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # PCA可視化
    scatter = axes[0, 0].scatter(latents_pca[:, 0], latents_pca[:, 1], c=labels, cmap='tab10', s=30)
    axes[0, 0].set_title('PCA Visualization')
    axes[0, 0].set_xlabel('PC1')
    axes[0, 0].set_ylabel('PC2')
    plt.colorbar(scatter, ax=axes[0, 0])
    
    # t-SNE可視化
    scatter = axes[0, 1].scatter(latents_tsne[:, 0], latents_tsne[:, 1], c=labels, cmap='tab10', s=30)
    axes[0, 1].set_title('t-SNE Visualization')
    axes[0, 1].set_xlabel('t-SNE 1')
    axes[0, 1].set_ylabel('t-SNE 2')
    plt.colorbar(scatter, ax=axes[0, 1])
    
    # f-k空間可視化
    scatter = axes[1, 0].scatter(f_params, k_params, c=labels, cmap='tab10', s=30)
    axes[1, 0].set_title('f-k Parameter Space')
    axes[1, 0].set_xlabel('f parameter')
    axes[1, 0].set_ylabel('k parameter')
    plt.colorbar(scatter, ax=axes[1, 0])
    
    # クラスタ分布
    unique_labels, counts = np.unique(labels, return_counts=True)
    axes[1, 1].bar(unique_labels, counts)
    axes[1, 1].set_title('Cluster Distribution')
    axes[1, 1].set_xlabel('Cluster')
    axes[1, 1].set_ylabel('Count')
    
    plt.tight_layout()
    plt.savefig('results/phase4_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main() 