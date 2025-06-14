#!/usr/bin/env python3
"""
1500サンプル対応 可視化スクリプト
latent_representations_frames_all.pkl から直接可視化
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import matplotlib as mpl

def load_new_data():
    """1500サンプルの新しいデータを読み込み"""
    
    print("📂 1500サンプルデータを読み込み中...")
    
    data_file = '../results/latent_representations_frames_all.pkl'
    
    if not os.path.exists(data_file):
        print(f"❌ {data_file} が見つかりません")
        return None
    
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    
    latent_vectors = data['latent_vectors']  # (1500, 128)
    filenames = data['filenames']            # 1500個
    f_values = data['f_values']              # 1500個
    k_values = data['k_values']              # 1500個
    
    print(f"✅ データ読み込み完了")
    print(f"   サンプル数: {len(filenames)}")
    print(f"   潜在次元: {latent_vectors.shape[1]}")
    print(f"   f値範囲: {f_values.min():.4f} - {f_values.max():.4f}")
    print(f"   k値範囲: {k_values.min():.4f} - {k_values.max():.4f}")
    
    return {
        'latent_vectors': latent_vectors,
        'filenames': filenames,
        'f_values': f_values,
        'k_values': k_values
    }

def perform_clustering(latent_vectors, n_clusters=20):
    """クラスタリングを実行"""
    
    print(f"\n🔍 クラスタリング実行中 (k={n_clusters})...")
    
    # K-means クラスタリング
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(latent_vectors)
    
    # シルエットスコア計算
    silhouette = silhouette_score(latent_vectors, cluster_labels)
    
    print(f"✅ クラスタリング完了")
    print(f"   シルエットスコア: {silhouette:.3f}")
    
    # クラスター分布
    unique, counts = np.unique(cluster_labels, return_counts=True)
    print(f"   クラスター分布:")
    for cluster, count in zip(unique, counts):
        print(f"     Cluster {cluster}: {count} samples ({count/len(cluster_labels)*100:.1f}%)")
    
    return cluster_labels, silhouette

def perform_dimensionality_reduction(latent_vectors):
    """次元削減を実行"""
    
    print(f"\n📉 次元削減実行中...")
    
    # PCA
    print("   PCA実行中...")
    pca = PCA(n_components=2, random_state=42)
    pca_result = pca.fit_transform(latent_vectors)
    print(f"     寄与率: PC1={pca.explained_variance_ratio_[0]:.3f}, PC2={pca.explained_variance_ratio_[1]:.3f}")
    
    # t-SNE
    print("   t-SNE実行中...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_result = tsne.fit_transform(latent_vectors)
    
    print("✅ 次元削減完了")
    
    return pca_result, tsne_result

def create_visualizations(data, cluster_labels, pca_result, tsne_result):
    """可視化を作成"""
    
    print(f"\n🎨 可視化作成中...")
    
    f_values = data['f_values']
    k_values = data['k_values']
    
    # 安全なカラーマップを選択
    try:
        colormap = 'viridis'
        test_cmap = plt.cm.get_cmap(colormap)
        print(f"✅ Using colormap: {colormap}")
    except:
        colormap = 'jet'
        print(f"⚠️  viridis not available, using fallback colormap: {colormap}")
    
    # クラスター別の色を生成
    unique_clusters = np.unique(cluster_labels)
    colors = plt.cm.get_cmap(colormap)(np.linspace(0, 1, len(unique_clusters)))
    
    # 1. f-k空間での可視化
    plt.figure(figsize=(12, 8))
    
    for i, cluster in enumerate(unique_clusters):
        mask = cluster_labels == cluster
        count = np.sum(mask)
        plt.scatter(f_values[mask], k_values[mask], 
                   c=[colors[i]], alpha=0.7, s=30, 
                   label=f'Cluster {int(cluster)} ({count})')
    
    plt.xlabel('f parameter', fontsize=12)
    plt.ylabel('k parameter', fontsize=12)
    plt.title('Gray-Scott Clustering Results (1500 samples)\nf-k Parameter Space', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    # 保存
    output_file = '../results/fk_scatter_1500samples.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✅ Saved: {output_file}")
    
    # 2. PCA可視化
    plt.figure(figsize=(10, 8))
    
    for i, cluster in enumerate(unique_clusters):
        mask = cluster_labels == cluster
        count = np.sum(mask)
        plt.scatter(pca_result[mask, 0], pca_result[mask, 1], 
                   c=[colors[i]], alpha=0.7, s=30, 
                   label=f'Cluster {int(cluster)} ({count})')
    
    plt.xlabel('PCA Component 1', fontsize=12)
    plt.ylabel('PCA Component 2', fontsize=12)
    plt.title('PCA Visualization (1500 samples)\nLatent Space', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 保存
    output_file = '../results/pca_scatter_1500samples.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✅ Saved: {output_file}")
    
    # 3. t-SNE可視化
    plt.figure(figsize=(10, 8))
    
    for i, cluster in enumerate(unique_clusters):
        mask = cluster_labels == cluster
        count = np.sum(mask)
        plt.scatter(tsne_result[mask, 0], tsne_result[mask, 1], 
                   c=[colors[i]], alpha=0.7, s=30, 
                   label=f'Cluster {int(cluster)} ({count})')
    
    plt.xlabel('t-SNE Component 1', fontsize=12)
    plt.ylabel('t-SNE Component 2', fontsize=12)
    plt.title('t-SNE Visualization (1500 samples)\nLatent Space', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 保存
    output_file = '../results/tsne_scatter_1500samples.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✅ Saved: {output_file}")

def print_cluster_statistics(data, cluster_labels):
    """クラスター統計を表示"""
    
    print(f"\n📊 クラスター統計 (1500サンプル)")
    print("-" * 60)
    
    f_values = data['f_values']
    k_values = data['k_values']
    unique_clusters = np.unique(cluster_labels)
    
    for cluster in unique_clusters:
        mask = cluster_labels == cluster
        count = np.sum(mask)
        f_mean = f_values[mask].mean()
        k_mean = k_values[mask].mean()
        f_std = f_values[mask].std()
        k_std = k_values[mask].std()
        
        print(f"Cluster {int(cluster):2d}: {count:4d} samples ({count/len(cluster_labels)*100:5.1f}%)")
        print(f"    f: {f_mean:.4f} ± {f_std:.4f}")
        print(f"    k: {k_mean:.4f} ± {k_std:.4f}")

def main():
    """メイン実行関数"""
    
    print("🎨 Gray-Scott 1500サンプル可視化")
    print("=" * 50)
    print(f"🔧 Matplotlib version: {mpl.__version__}")
    
    # データ読み込み
    data = load_new_data()
    if data is None:
        return
    
    # クラスタリング実行
    cluster_labels, silhouette = perform_clustering(data['latent_vectors'])
    
    # 次元削減実行
    pca_result, tsne_result = perform_dimensionality_reduction(data['latent_vectors'])
    
    # 可視化作成
    create_visualizations(data, cluster_labels, pca_result, tsne_result)
    
    # 統計表示
    print_cluster_statistics(data, cluster_labels)
    
    print(f"\n🎉 1500サンプル可視化完了!")
    print(f"📈 最終シルエットスコア: {silhouette:.3f}")
    print(f"📁 保存された画像:")
    print(f"   - fk_scatter_1500samples.png")
    print(f"   - pca_scatter_1500samples.png") 
    print(f"   - tsne_scatter_1500samples.png")

if __name__ == "__main__":
    main() 