#!/usr/bin/env python3
"""
1500サンプル版 統合可視化
gray_scott_clustering_results.png の1500サンプル版を作成
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from matplotlib.patches import Patch
import matplotlib as mpl

def load_and_process_1500_data():
    """1500サンプルデータを読み込んでクラスタリング実行"""
    
    print("📂 1500サンプルデータを読み込み・処理中...")
    
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
    
    print(f"✅ データ読み込み完了: {len(filenames)} サンプル")
    
    # クラスタリング実行
    print("🔍 クラスタリング実行中...")
    n_clusters = 20
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(latent_vectors)
    silhouette = silhouette_score(latent_vectors, cluster_labels)
    print(f"   シルエットスコア: {silhouette:.3f}")
    
    # 次元削減実行
    print("📉 次元削減実行中...")
    pca = PCA(n_components=2, random_state=42)
    pca_result = pca.fit_transform(latent_vectors)
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_result = tsne.fit_transform(latent_vectors)
    
    return {
        'f_values': f_values,
        'k_values': k_values,
        'cluster_labels': cluster_labels,
        'pca_result': pca_result,
        'tsne_result': tsne_result,
        'silhouette_score': silhouette,
        'n_samples': len(filenames)
    }

def create_combined_visualization_1500(results, figsize=(15, 12), dpi=300, save_name='gray_scott_clustering_results_1500samples.png'):
    """4つのプロットを含む統合可視化（1500サンプル版）"""
    
    f_values = results['f_values']
    k_values = results['k_values']
    cluster_labels = results['cluster_labels']
    latent_2d_pca = results['pca_result']
    latent_2d_tsne = results['tsne_result']
    
    print(f"🎨 統合可視化作成中 (1500サンプル)...")
    
    # 安全なカラーマップを選択
    try:
        colormap = 'viridis'
        test_cmap = plt.cm.get_cmap(colormap)
    except:
        colormap = 'jet'
        print(f"⚠️  viridis not available, using {colormap}")
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # 1. f-k空間でのクラスタリング結果
    try:
        scatter1 = axes[0, 0].scatter(f_values, k_values, c=cluster_labels, 
                                      cmap=colormap, alpha=0.7, s=20)
        plt.colorbar(scatter1, ax=axes[0, 0], label='Cluster')
    except Exception as e:
        print(f"f-k scatter plot error: {e}")
        # フォールバック: 個別の色指定
        unique_clusters = np.unique(cluster_labels)
        colors = plt.cm.get_cmap(colormap)(np.linspace(0, 1, len(unique_clusters)))
        for i, cluster in enumerate(unique_clusters):
            mask = cluster_labels == cluster
            axes[0, 0].scatter(f_values[mask], k_values[mask], 
                               c=[colors[i]], alpha=0.7, s=20, label=f'C{cluster}')
    
    axes[0, 0].set_xlabel('f parameter')
    axes[0, 0].set_ylabel('k parameter')
    axes[0, 0].set_title('Clustering Results in f-k Space\n(1500 samples)')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].invert_yaxis()
    
    # 2. PCA可視化
    try:
        scatter2 = axes[0, 1].scatter(latent_2d_pca[:, 0], latent_2d_pca[:, 1], 
                                      c=cluster_labels, cmap=colormap, alpha=0.7, s=20)
        plt.colorbar(scatter2, ax=axes[0, 1], label='Cluster')
    except Exception as e:
        print(f"PCA scatter plot error: {e}")
        unique_clusters = np.unique(cluster_labels)
        colors = plt.cm.get_cmap(colormap)(np.linspace(0, 1, len(unique_clusters)))
        for i, cluster in enumerate(unique_clusters):
            mask = cluster_labels == cluster
            axes[0, 1].scatter(latent_2d_pca[mask, 0], latent_2d_pca[mask, 1], 
                               c=[colors[i]], alpha=0.7, s=20)
    
    axes[0, 1].set_xlabel('PCA Component 1')
    axes[0, 1].set_ylabel('PCA Component 2')
    axes[0, 1].set_title('PCA Visualization of Latent Space\n(1500 samples)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. t-SNE可視化
    try:
        scatter3 = axes[1, 0].scatter(latent_2d_tsne[:, 0], latent_2d_tsne[:, 1], 
                                      c=cluster_labels, cmap=colormap, alpha=0.7, s=20)
        plt.colorbar(scatter3, ax=axes[1, 0], label='Cluster')
    except Exception as e:
        print(f"t-SNE scatter plot error: {e}")
        unique_clusters = np.unique(cluster_labels)
        colors = plt.cm.get_cmap(colormap)(np.linspace(0, 1, len(unique_clusters)))
        for i, cluster in enumerate(unique_clusters):
            mask = cluster_labels == cluster
            axes[1, 0].scatter(latent_2d_tsne[mask, 0], latent_2d_tsne[mask, 1], 
                               c=[colors[i]], alpha=0.7, s=20)
    
    axes[1, 0].set_xlabel('t-SNE Component 1')
    axes[1, 0].set_ylabel('t-SNE Component 2')
    axes[1, 0].set_title('t-SNE Visualization of Latent Space\n(1500 samples)')
    axes[1, 0].grid(True, alpha=0.3)
    
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
    
    try:
        im = axes[1, 1].imshow(heatmap_data, cmap=colormap, aspect='auto', origin='upper')
    except Exception as e:
        print(f"Heatmap error: {e}")
        im = axes[1, 1].imshow(heatmap_data, cmap='hot', aspect='auto', origin='upper')
    
    # ヒートマップの軸設定（一部のラベルのみ表示）
    step_k = max(1, len(k_unique) // 10)
    step_f = max(1, len(f_unique) // 10)
    
    axes[1, 1].set_xticks(range(0, len(k_unique), step_k))
    axes[1, 1].set_yticks(range(0, len(f_unique), step_f))
    axes[1, 1].set_xticklabels([f'{k:.4f}' for k in k_unique[::step_k]], rotation=45)
    axes[1, 1].set_yticklabels([f'{f:.4f}' for f in f_unique[::step_f]])
    axes[1, 1].set_xlabel('k parameter')
    axes[1, 1].set_ylabel('f parameter')
    axes[1, 1].set_title('Cluster Heatmap in f-k Space\n(1500 samples)')
    axes[1, 1].invert_yaxis()
    
    # ヒートマップ用の凡例（簡略化）
    unique_clusters = np.unique(cluster_labels[~np.isnan(cluster_labels)])
    try:
        cmap = plt.cm.get_cmap(colormap)
        legend_elements = [Patch(facecolor=cmap(cluster/len(unique_clusters)), 
                               label=f'C{int(cluster)}') 
                         for cluster in unique_clusters[::2]]  # 2つおきに表示
    except:
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        legend_elements = [Patch(facecolor=colors[i % len(colors)], 
                               label=f'C{int(cluster)}') 
                         for i, cluster in enumerate(unique_clusters[::2])]
    
    axes[1, 1].legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    # 全体のタイトル
    fig.suptitle(f'Gray-Scott Clustering Results (1500 samples)\nSilhouette Score: {results["silhouette_score"]:.3f}', 
                 fontsize=16, fontweight='bold', y=0.95)
    
    plt.tight_layout()
    
    # 保存パスの修正
    if not save_name.startswith('../results/'):
        save_name = f'../results/{save_name}'
    
    plt.savefig(save_name, dpi=dpi, bbox_inches='tight')
    plt.show()
    
    print(f"✅ Saved: {save_name}")

def print_comparison_stats(results):
    """375サンプルとの比較統計を表示"""
    
    print(f"\n📊 1500サンプル vs 375サンプル比較")
    print("=" * 50)
    print(f"📈 サンプル数: 375 → 1500 (4.0倍)")
    print(f"📈 潜在次元: 64 → 128 (2.0倍)")
    print(f"📈 シルエットスコア: 0.551 → {results['silhouette_score']:.3f}")
    
    # クラスター分布
    unique, counts = np.unique(results['cluster_labels'], return_counts=True)
    print(f"📈 クラスター数: {len(unique)}")
    print(f"📈 最大クラスター: {counts.max()} samples ({counts.max()/len(results['cluster_labels'])*100:.1f}%)")
    print(f"📈 最小クラスター: {counts.min()} samples ({counts.min()/len(results['cluster_labels'])*100:.1f}%)")
    
    # パラメータ範囲
    f_values = results['f_values']
    k_values = results['k_values']
    print(f"📈 f値範囲: {f_values.min():.4f} - {f_values.max():.4f}")
    print(f"📈 k値範囲: {k_values.min():.4f} - {k_values.max():.4f}")

def main():
    """メイン実行関数"""
    
    print("🎨 Gray-Scott 1500サンプル統合可視化")
    print("=" * 50)
    print(f"🔧 Matplotlib version: {mpl.__version__}")
    
    # データ処理
    results = load_and_process_1500_data()
    if results is None:
        return
    
    # 統合可視化作成
    create_combined_visualization_1500(results)
    
    # 比較統計表示
    print_comparison_stats(results)
    
    print(f"\n🎉 1500サンプル統合可視化完了!")
    print(f"📁 保存ファイル: gray_scott_clustering_results_1500samples.png")

if __name__ == "__main__":
    main() 