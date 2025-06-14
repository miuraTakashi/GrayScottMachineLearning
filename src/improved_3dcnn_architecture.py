#!/usr/bin/env python3
"""
3D CNN分離能力向上のための改善案
Gray-Scottパターン分類の性能向上を目指した包括的改善策
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# ================================
# 改善案1: 注意機構付き3D CNN
# ================================

class SpatioTemporalAttention(nn.Module):
    """時空間注意機構"""
    def __init__(self, channels):
        super(SpatioTemporalAttention, self).__init__()
        self.spatial_attention = nn.Sequential(
            nn.Conv3d(channels, channels//8, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels//8, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.temporal_attention = nn.Sequential(
            nn.AdaptiveAvgPool3d((None, 1, 1)),
            nn.Conv3d(channels, channels//8, kernel_size=(3, 1, 1), padding=(1, 0, 0)),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels//8, channels, kernel_size=(3, 1, 1), padding=(1, 0, 0)),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # 空間注意
        spatial_att = self.spatial_attention(x)
        x_spatial = x * spatial_att
        
        # 時間注意  
        temporal_att = self.temporal_attention(x)
        x_temporal = x_spatial * temporal_att
        
        return x_temporal

class ResidualBlock3D(nn.Module):
    """3D残差ブロック"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, 
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )
        
        self.attention = SpatioTemporalAttention(out_channels)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.attention(out)  # 注意機構適用
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ImprovedConv3DAutoencoder(nn.Module):
    """改善された3D CNNオートエンコーダー"""
    def __init__(self, input_channels=1, fixed_frames=30, target_size=(64, 64), latent_dim=256):
        super(ImprovedConv3DAutoencoder, self).__init__()
        
        self.latent_dim = latent_dim
        self.fixed_frames = fixed_frames
        self.target_size = target_size
        
        # 改善されたエンコーダー
        self.encoder = nn.Sequential(
            # 初期特徴抽出
            nn.Conv3d(input_channels, 32, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            
            # 残差ブロック群
            ResidualBlock3D(32, 64, stride=(2, 2, 2)),
            ResidualBlock3D(64, 64),
            ResidualBlock3D(64, 128, stride=(2, 2, 2)),
            ResidualBlock3D(128, 128),
            ResidualBlock3D(128, 256, stride=(2, 2, 2)),
            ResidualBlock3D(256, 256),
            
            # グローバル特徴抽出
            nn.AdaptiveAvgPool3d((2, 2, 2)),
            nn.Dropout3d(0.3),
        )
        
        # 改善された潜在空間射影
        self.to_latent = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 2 * 2 * 2, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, latent_dim),
            nn.BatchNorm1d(latent_dim)  # 潜在空間の正規化
        )
        
        # 改善された復元
        self.from_latent = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256 * 2 * 2 * 2),
            nn.ReLU(inplace=True)
        )
        
        # 改善されたデコーダー
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(256, 128, kernel_size=(3, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose3d(128, 64, kernel_size=(3, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose3d(64, 32, kernel_size=(3, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose3d(32, input_channels, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3)),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        encoded = self.encoder(x)
        latent = self.to_latent(encoded)
        return latent
    
    def decode(self, latent):
        decoded = self.from_latent(latent)
        decoded = decoded.view(-1, 256, 2, 2, 2)
        output = self.decoder(decoded)
        
        # 適応的サイズ調整
        target_h, target_w = self.target_size
        output = F.interpolate(output, size=(self.fixed_frames, target_h, target_w), 
                              mode='trilinear', align_corners=False)
        return output
    
    def forward(self, x):
        latent = self.encode(x)
        reconstructed = self.decode(latent)
        return reconstructed, latent

# ================================
# 改善案2: マルチスケール特徴融合
# ================================

class MultiScaleFeatureFusion(nn.Module):
    """マルチスケール特徴融合"""
    def __init__(self, input_channels=1, latent_dim=256):
        super(MultiScaleFeatureFusion, self).__init__()
        
        # 異なるスケールでの特徴抽出
        self.scale1 = nn.Sequential(  # 高解像度特徴
            nn.Conv3d(input_channels, 32, kernel_size=(3, 3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((15, 32, 32))
        )
        
        self.scale2 = nn.Sequential(  # 中解像度特徴
            nn.Conv3d(input_channels, 32, kernel_size=(5, 5, 5), padding=2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((15, 32, 32))
        )
        
        self.scale3 = nn.Sequential(  # 低解像度特徴
            nn.Conv3d(input_channels, 32, kernel_size=(7, 7, 7), padding=3),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((15, 32, 32))
        )
        
        # 特徴融合
        self.fusion = nn.Sequential(
            nn.Conv3d(96, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
            nn.Linear(64, latent_dim)
        )
    
    def forward(self, x):
        feat1 = self.scale1(x)
        feat2 = self.scale2(x)
        feat3 = self.scale3(x)
        
        # チャンネル方向で結合
        combined = torch.cat([feat1, feat2, feat3], dim=1)
        output = self.fusion(combined)
        
        return output

# ================================
# 改善案3: 対比学習による特徴学習
# ================================

class ContrastiveLoss(nn.Module):
    """対比損失"""
    def __init__(self, temperature=0.1):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
    
    def forward(self, features, labels):
        # L2正規化
        features = F.normalize(features, dim=1)
        
        # 類似度行列計算
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # 正例・負例マスク作成
        labels = labels.view(-1, 1)
        positive_mask = torch.eq(labels, labels.T).float()
        negative_mask = 1 - positive_mask
        
        # 対比損失計算
        exp_sim = torch.exp(similarity_matrix)
        positive_sum = torch.sum(exp_sim * positive_mask, dim=1)
        total_sum = torch.sum(exp_sim * negative_mask, dim=1) + positive_sum
        
        loss = -torch.log(positive_sum / total_sum)
        return torch.mean(loss)

class ContrastiveAutoencoder(nn.Module):
    """対比学習付きオートエンコーダー"""
    def __init__(self, base_encoder, projection_dim=128):
        super(ContrastiveAutoencoder, self).__init__()
        self.encoder = base_encoder
        
        # 対比学習用の射影ヘッド
        self.projection_head = nn.Sequential(
            nn.Linear(base_encoder.latent_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, projection_dim)
        )
        
        self.contrastive_loss = ContrastiveLoss()
    
    def forward(self, x, f_values=None, k_values=None):
        latent = self.encoder.encode(x)
        reconstructed = self.encoder.decode(latent)
        
        # 対比学習用特徴
        projected = self.projection_head(latent)
        
        return reconstructed, latent, projected

# ================================
# 改善案4: データ拡張戦略
# ================================

class GrayScottAugmentation:
    """Gray-Scott専用データ拡張"""
    
    @staticmethod
    def temporal_shuffle(tensor, probability=0.3):
        """時間軸のシャッフル"""
        if np.random.random() < probability:
            T = tensor.shape[2]  # フレーム次元
            indices = torch.randperm(T)
            tensor = tensor[:, :, indices, :, :]
        return tensor
    
    @staticmethod
    def temporal_crop(tensor, crop_ratio=0.8):
        """時間軸のクロップ"""
        T = tensor.shape[2]
        crop_length = int(T * crop_ratio)
        start_idx = np.random.randint(0, T - crop_length + 1)
        return tensor[:, :, start_idx:start_idx+crop_length, :, :]
    
    @staticmethod
    def spatial_rotation(tensor, max_angle=15):
        """空間回転"""
        angle = np.random.uniform(-max_angle, max_angle)
        # PyTorchの回転変換を適用
        # 実装は省略（torchvision.transformsを使用）
        return tensor
    
    @staticmethod
    def noise_injection(tensor, noise_level=0.05):
        """ノイズ注入"""
        noise = torch.randn_like(tensor) * noise_level
        return torch.clamp(tensor + noise, 0, 1)

# ================================
# 改善案5: 階層的クラスタリング
# ================================

def hierarchical_clustering_analysis(latent_vectors, f_values, k_values):
    """階層的クラスタリング分析"""
    from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
    from sklearn.metrics import silhouette_score
    
    # 階層的クラスタリング実行
    linkage_matrix = linkage(latent_vectors, method='ward')
    
    # 最適クラスター数の自動決定
    silhouette_scores = []
    cluster_range = range(2, min(51, len(latent_vectors)//10))
    
    for n_clusters in cluster_range:
        cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
        score = silhouette_score(latent_vectors, cluster_labels)
        silhouette_scores.append(score)
    
    optimal_k = cluster_range[np.argmax(silhouette_scores)]
    best_labels = fcluster(linkage_matrix, optimal_k, criterion='maxclust')
    
    return best_labels, optimal_k, linkage_matrix, silhouette_scores

# ================================
# 改善案6: 評価指標の改善
# ================================

def comprehensive_evaluation(latent_vectors, cluster_labels, f_values, k_values):
    """包括的評価指標"""
    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
    from sklearn.neighbors import NearestNeighbors
    
    results = {}
    
    # 基本クラスタリング指標
    results['silhouette_score'] = silhouette_score(latent_vectors, cluster_labels)
    results['calinski_harabasz_score'] = calinski_harabasz_score(latent_vectors, cluster_labels)
    results['davies_bouldin_score'] = davies_bouldin_score(latent_vectors, cluster_labels)
    
    # パラメータ空間での分離度
    param_vectors = np.column_stack([f_values, k_values])
    results['param_separation'] = silhouette_score(param_vectors, cluster_labels)
    
    # 近傍一致度（同じクラスターの点が潜在空間とパラメータ空間で近いか）
    nbrs_latent = NearestNeighbors(n_neighbors=5).fit(latent_vectors)
    nbrs_param = NearestNeighbors(n_neighbors=5).fit(param_vectors)
    
    concordance_scores = []
    for i in range(len(latent_vectors)):
        _, latent_neighbors = nbrs_latent.kneighbors([latent_vectors[i]])
        _, param_neighbors = nbrs_param.kneighbors([param_vectors[i]])
        
        latent_labels = cluster_labels[latent_neighbors[0]]
        param_labels = cluster_labels[param_neighbors[0]]
        
        # 近傍でのラベル一致度
        concordance = np.mean(latent_labels == param_labels)
        concordance_scores.append(concordance)
    
    results['neighborhood_concordance'] = np.mean(concordance_scores)
    
    return results

# ================================
# 訓練戦略の改善
# ================================

def improved_training_strategy():
    """改善された訓練戦略"""
    
    strategies = {
        # 1. 段階的訓練
        "progressive_training": {
            "description": "低解像度から高解像度への段階的訓練",
            "steps": [
                "32x32で基本特徴学習",
                "64x64で詳細特徴学習", 
                "元解像度でファインチューニング"
            ]
        },
        
        # 2. カリキュラム学習
        "curriculum_learning": {
            "description": "簡単なパターンから複雑なパターンへの段階的学習",
            "steps": [
                "安定パターン（低f値）から開始",
                "動的パターン（高f値）を段階的に追加",
                "全パターンでの最終調整"
            ]
        },
        
        # 3. 正則化技術
        "regularization": {
            "description": "過学習防止と汎化性能向上",
            "techniques": [
                "Dropout (0.3-0.5)",
                "Weight Decay (1e-4)",
                "Early Stopping",
                "Label Smoothing"
            ]
        },
        
        # 4. 最適化手法
        "optimization": {
            "description": "適応的学習率とオプティマイザー",
            "methods": [
                "AdamW optimizer",
                "Cosine Annealing LR",
                "Warm-up phase",
                "Gradient Clipping"
            ]
        }
    }
    
    return strategies

# ================================
# 性能評価と可視化
# ================================

def plot_improvement_analysis(original_results, improved_results):
    """改善効果の可視化"""
    
    metrics = ['silhouette_score', 'param_separation', 'neighborhood_concordance']
    original_scores = [original_results.get(m, 0) for m in metrics]
    improved_scores = [improved_results.get(m, 0) for m in metrics]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 比較棒グラフ
    x_pos = np.arange(len(metrics))
    width = 0.35
    
    ax1.bar(x_pos - width/2, original_scores, width, label='Original', alpha=0.8)
    ax1.bar(x_pos + width/2, improved_scores, width, label='Improved', alpha=0.8)
    ax1.set_xlabel('Metrics')
    ax1.set_ylabel('Score')
    ax1.set_title('Performance Comparison')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([m.replace('_', '\n') for m in metrics])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 改善率
    improvement_rates = [(imp - orig) / orig * 100 if orig > 0 else 0 
                        for orig, imp in zip(original_scores, improved_scores)]
    
    colors = ['green' if rate > 0 else 'red' for rate in improvement_rates]
    ax2.bar(x_pos, improvement_rates, color=colors, alpha=0.7)
    ax2.set_xlabel('Metrics')
    ax2.set_ylabel('Improvement Rate (%)')
    ax2.set_title('Improvement Analysis')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([m.replace('_', '\n') for m in metrics])
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    plt.tight_layout()
    return fig

# ================================
# 実装推奨事項
# ================================

def implementation_recommendations():
    """実装推奨事項"""
    
    recommendations = {
        "immediate_improvements": [
            "🎯 潜在次元を64→256に拡張（表現力向上）",
            "🔄 残差接続とスキップ接続の追加",
            "👁️ 注意機構の導入（時空間attention）",
            "📊 バッチ正規化の最適化",
            "🎲 データ拡張の実装"
        ],
        
        "medium_term_goals": [
            "🌐 マルチスケール特徴融合",
            "📚 対比学習の導入", 
            "🔄 階層的クラスタリング分析",
            "📈 包括的評価指標の実装",
            "🎓 カリキュラム学習の導入"
        ],
        
        "advanced_techniques": [
            "🤖 Vision Transformer (ViT) の適用",
            "🔬 Self-Supervised Learning",
            "🎯 Metric Learning",
            "🌊 Graph Neural Networks for pattern relationships",
            "🔄 Domain Adaptation techniques"
        ]
    }
    
    return recommendations

def main():
    """改善案のサマリー表示"""
    
    print("🚀 3D CNN分離能力向上のための改善案")
    print("=" * 60)
    
    print("\n📋 主要改善案:")
    print("1️⃣ 注意機構付き3D CNN - 時空間注意で重要な特徴を強調")
    print("2️⃣ マルチスケール特徴融合 - 異なるスケールでの特徴統合")
    print("3️⃣ 対比学習 - 類似パターンを近く、異なるパターンを遠くに配置")
    print("4️⃣ データ拡張戦略 - 時間軸・空間軸の多様な変換")
    print("5️⃣ 階層的クラスタリング - より自然なパターン分類")
    print("6️⃣ 包括的評価指標 - 多角的な性能評価")
    
    print("\n🎯 期待効果:")
    print("• シルエットスコア: 0.413 → 0.6+ (45%以上向上)")
    print("• クラスター分離度: より明確な境界形成")
    print("• パラメータ対応: f-k空間との一致度向上")
    print("• 汎化性能: 未知パターンへの対応力向上")
    
    strategies = improved_training_strategy()
    print(f"\n📚 訓練戦略:")
    for name, strategy in strategies.items():
        print(f"• {strategy['description']}")
    
    recs = implementation_recommendations()
    print(f"\n✅ 実装推奨事項:")
    for category, items in recs.items():
        print(f"\n{category.replace('_', ' ').title()}:")
        for item in items:
            print(f"  {item}")

if __name__ == "__main__":
    main() 