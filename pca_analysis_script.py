#!/usr/bin/env python3
"""
PCA詳細分析スクリプト
直線的なマッピングの原因を調査するための分析ツール
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns

def load_data():
    """データを読み込む"""
    results_path = '/Users/miura/Library/CloudStorage/GoogleDrive-miuratakashilab@gmail.com/マイドライブ/GrayScottML/phase2_results_last64_gpu.pkl'
    
    with open(results_path, 'rb') as f:
        results = pickle.load(f)
    
    return results['latent_vectors'], results['f_values'], results['k_values']

def analyze_pca_linearity(latent_vectors, f_values, k_values):
    """PCAの直線性を詳細分析"""
    
    print("PCA詳細分析:")
    print("=" * 50)
    
    # 1. 潜在ベクトルの基本統計
    print("1. 潜在ベクトルの基本統計:")
    print(f"  形状: {latent_vectors.shape}")
    print(f"  平均値: {latent_vectors.mean():.6f}")
    print(f"  標準偏差: {latent_vectors.std():.6f}")
    print(f"  最小値: {latent_vectors.min():.6f}")
    print(f"  最大値: {latent_vectors.max():.6f}")
    
    # 2. 相関行列の分析
    print("\n2. 相関行列の分析:")
    corr_matrix = np.corrcoef(latent_vectors[:, :10].T)
    print(f"  最初の10次元間の相関係数範囲: {corr_matrix.min():.3f} ~ {corr_matrix.max():.3f}")
    
    # 強い相関（|r| > 0.8）を持つペアを検出
    strong_corr_pairs = []
    for i in range(10):
        for j in range(i+1, 10):
            if abs(corr_matrix[i, j]) > 0.8:
                strong_corr_pairs.append((i, j, corr_matrix[i, j]))
    
    if strong_corr_pairs:
        print(f"  強い相関（|r| > 0.8）を持つペア数: {len(strong_corr_pairs)}")
        print("  例（最初の5つ）:")
        for i, j, corr in strong_corr_pairs[:5]:
            print(f"    次元{i}と次元{j}: r = {corr:.3f}")
    else:
        print("  強い相関を持つペアは見つかりませんでした")
    
    # 3. PCAの詳細情報
    print("\n3. PCAの詳細情報:")
    pca = PCA(n_components=2, random_state=42)
    pca_result = pca.fit_transform(latent_vectors)
    
    print(f"  説明分散比: {pca.explained_variance_ratio_}")
    print(f"  累積説明分散比: {np.cumsum(pca.explained_variance_ratio_)}")
    print(f"  第1主成分の説明分散: {pca.explained_variance_ratio_[0]:.1%}")
    print(f"  第2主成分の説明分散: {pca.explained_variance_ratio_[1]:.1%}")
    
    # 4. 主成分の重みを確認
    print("\n4. 主成分の重み分析:")
    pc1_weights = pca.components_[0]
    pc2_weights = pca.components_[1]
    
    print(f"  第1主成分の重み範囲: {pc1_weights.min():.3f} ~ {pc1_weights.max():.3f}")
    print(f"  第2主成分の重み範囲: {pc2_weights.min():.3f} ~ {pc2_weights.max():.3f}")
    
    # 重みの大きい特徴量を確認
    top_pc1_indices = np.argsort(np.abs(pc1_weights))[-10:]
    top_pc2_indices = np.argsort(np.abs(pc2_weights))[-10:]
    
    print("  第1主成分で重要な特徴量（上位10個）:")
    for i, idx in enumerate(reversed(top_pc1_indices)):
        print(f"    次元{idx}: 重み = {pc1_weights[idx]:.3f}")
    
    print("  第2主成分で重要な特徴量（上位10個）:")
    for i, idx in enumerate(reversed(top_pc2_indices)):
        print(f"    次元{idx}: 重み = {pc2_weights[idx]:.3f}")
    
    # 5. 直線性の検証
    print("\n5. 直線性の検証:")
    pc1_pc2_corr = np.corrcoef(pca_result[:, 0], pca_result[:, 1])[0, 1]
    print(f"  PC1とPC2の相関係数: {pc1_pc2_corr:.3f}")
    
    if abs(pc1_pc2_corr) > 0.3:
        print("  ⚠️  PC1とPC2に相関があります - 直線的な配置の可能性")
    else:
        print("  ✓ PC1とPC2は独立しています")
    
    # 6. より多くの主成分での分析
    print("\n6. より多くの主成分での分析:")
    pca_full = PCA(random_state=42)
    pca_full.fit(latent_vectors)
    
    cumulative_var = np.cumsum(pca_full.explained_variance_ratio_)
    n_components_90 = np.argmax(cumulative_var >= 0.9) + 1
    n_components_95 = np.argmax(cumulative_var >= 0.95) + 1
    
    print(f"  90%の分散を説明する主成分数: {n_components_90}")
    print(f"  95%の分散を説明する主成分数: {n_components_95}")
    print(f"  全256次元での累積説明分散: {cumulative_var[-1]:.3f}")
    
    # 7. 可視化
    create_visualization_plots(latent_vectors, pca_result, pca, pc1_weights, pc2_weights, cumulative_var)
    
    # 8. 結論と推奨事項
    print("\n8. 結論と推奨事項:")
    print("=" * 50)
    
    if pca.explained_variance_ratio_[0] > 0.8:
        print("  ⚠️  第1主成分が80%以上の分散を説明 - データが1次元的")
        print("      → 直線的な配置は正常な現象です")
    elif pca.explained_variance_ratio_[0] + pca.explained_variance_ratio_[1] > 0.9:
        print("  ⚠️  第1・第2主成分で90%以上の分散を説明 - データが2次元的")
        print("      → 直線的な配置は正常な現象です")
    else:
        print("  ✓ データは高次元の構造を持っています")
    
    if abs(pc1_pc2_corr) > 0.3:
        print("  ⚠️  PC1とPC2に相関があります")
        print("      → より多くの主成分での分析を推奨")
    
    print(f"\n推奨事項:")
    print(f"  1. より多くの主成分（{n_components_90}個以上）での分析を試してください")
    print(f"  2. t-SNEやUMAPなどの非線形次元削減手法を試してください")
    print(f"  3. データの前処理（正規化、標準化）を確認してください")
    
    return pca, pca_result, pca_full, cumulative_var

def create_visualization_plots(latent_vectors, pca_result, pca, pc1_weights, pc2_weights, cumulative_var):
    """可視化プロットを作成"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('PCA詳細分析', fontsize=16, fontweight='bold')
    
    # 1. 累積説明分散
    axes[0, 0].plot(range(1, 21), cumulative_var[:20], 'bo-')
    axes[0, 0].axhline(y=0.9, color='r', linestyle='--', alpha=0.7, label='90%')
    axes[0, 0].axhline(y=0.95, color='orange', linestyle='--', alpha=0.7, label='95%')
    axes[0, 0].set_xlabel('主成分数')
    axes[0, 0].set_ylabel('累積説明分散比')
    axes[0, 0].set_title('累積説明分散比')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. PC1とPC2の散布図
    scatter = axes[0, 1].scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.7, s=20)
    axes[0, 1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    axes[0, 1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    axes[0, 1].set_title('PCA結果')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 主成分の重み分布
    axes[0, 2].hist(pc1_weights, bins=30, alpha=0.7, label='PC1', color='blue')
    axes[0, 2].hist(pc2_weights, bins=30, alpha=0.7, label='PC2', color='red')
    axes[0, 2].set_xlabel('重みの値')
    axes[0, 2].set_ylabel('頻度')
    axes[0, 2].set_title('主成分の重み分布')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. 相関行列のヒートマップ（最初の20次元）
    corr_matrix_20 = np.corrcoef(latent_vectors[:, :20].T)
    im = axes[1, 0].imshow(corr_matrix_20, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    axes[1, 0].set_title('相関行列（最初の20次元）')
    axes[1, 0].set_xlabel('次元')
    axes[1, 0].set_ylabel('次元')
    plt.colorbar(im, ax=axes[1, 0])
    
    # 5. 説明分散比の棒グラフ
    axes[1, 1].bar(range(1, len(pca.explained_variance_ratio_)+1), pca.explained_variance_ratio_)
    axes[1, 1].set_xlabel('主成分')
    axes[1, 1].set_ylabel('説明分散比')
    axes[1, 1].set_title('説明分散比')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. PC1とPC2の関係性
    axes[1, 2].scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.5, s=10)
    axes[1, 2].set_xlabel('PC1')
    axes[1, 2].set_ylabel('PC2')
    axes[1, 2].set_title('PC1 vs PC2')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def analyze_alternative_dimension_reduction(latent_vectors):
    """代替的な次元削減手法の分析"""
    
    print("\n代替的な次元削減手法の分析:")
    print("=" * 50)
    
    # t-SNEの分析
    print("1. t-SNE分析:")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_result = tsne.fit_transform(latent_vectors)
    
    # t-SNE結果の可視化
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(tsne_result[:, 0], tsne_result[:, 1], alpha=0.7, s=20)
    plt.title('t-SNE結果')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.grid(True, alpha=0.3)
    
    # より多くの主成分でのPCA
    pca_more = PCA(n_components=10, random_state=42)
    pca_more_result = pca_more.fit_transform(latent_vectors)
    
    plt.subplot(1, 2, 2)
    plt.scatter(pca_more_result[:, 0], pca_more_result[:, 1], alpha=0.7, s=20)
    plt.title('PCA (10成分) - PC1 vs PC2')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"  t-SNE結果の形状: {tsne_result.shape}")
    print(f"  10成分PCAの累積説明分散: {np.cumsum(pca_more.explained_variance_ratio_)[:5]}")

def main():
    """メイン関数"""
    print("PCA直線性分析を開始します...")
    
    # データ読み込み
    latent_vectors, f_values, k_values = load_data()
    
    # PCA詳細分析
    pca, pca_result, pca_full, cumulative_var = analyze_pca_linearity(
        latent_vectors, f_values, k_values
    )
    
    # 代替手法の分析
    analyze_alternative_dimension_reduction(latent_vectors)
    
    print("\n分析完了！")

if __name__ == "__main__":
    main() 