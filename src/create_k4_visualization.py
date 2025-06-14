#!/usr/bin/env python3
"""
k=4 最適クラスター数での1500サンプル可視化
バランス重視の推奨値での統合可視化
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

def load_and_process_k4_data():
    """1500サンプルデータをk=4でクラスタリング実行"""
    
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
    
    print(f"✅ データ読み込み完了: {len(filenames)} サンプル")
    
    # k=4でクラスタリング実行
    print("🔍 k=4でクラスタリング実行中...")
    n_clusters = 4
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(latent_vectors)
    silhouette = silhouette_score(latent_vectors, cluster_labels)
    print(f"   シルエットスコア: {silhouette:.3f}")
    
    # クラスター分布表示
    unique, counts = np.unique(cluster_labels, return_counts=True)
    print(f"   クラスター分布:")
    for cluster, count in zip(unique, counts):
        print(f"     Cluster {cluster}: {count} samples ({count/len(cluster_labels)*100:.1f}%)")
    
    # 次元削減実行
    print("📉 次元削減実行中...")
    pca = PCA(n_components=2, random_state=42)
    pca_result = pca.fit_transform(latent_vectors)
    print(f"   PCA寄与率: PC1={pca.explained_variance_ratio_[0]:.3f}, PC2={pca.explained_variance_ratio_[1]:.3f}")
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_result = tsne.fit_transform(latent_vectors)
    print("   t-SNE完了")
    
    return {
        'f_values': f_values,
        'k_values': k_values,
        'cluster_labels': cluster_labels,
        'pca_result': pca_result,
        'tsne_result': tsne_result,
        'silhouette_score': silhouette,
        'n_samples': len(filenames),
        'cluster_counts': counts
    }

def create_k4_visualization(results, figsize=(15, 12), dpi=300, save_name='gray_scott_clustering_results_k4_1500samples.png'):
    """k=4での統合可視化（1500サンプル版）"""
    
    f_values = results['f_values']
    k_values = results['k_values']
    cluster_labels = results['cluster_labels']
    latent_2d_pca = results['pca_result']
    latent_2d_tsne = results['tsne_result']
    
    print(f"🎨 k=4統合可視化作成中 (1500サンプル)...")
    
    # 安全なカラーマップを選択
    try:
        colormap = 'viridis'
        test_cmap = plt.cm.get_cmap(colormap)
    except:
        colormap = 'Set1'  # k=4なのでカテゴリカルカラーも使用可能
        print(f"⚠️  viridis not available, using {colormap}")
    
    # k=4専用の色設定
    cluster_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # 見やすい4色
    cluster_names = ['Pattern A', 'Pattern B', 'Pattern C', 'Pattern D']
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # 1. f-k空間でのクラスタリング結果
    for i, cluster in enumerate(np.unique(cluster_labels)):
        mask = cluster_labels == cluster
        count = np.sum(mask)
        axes[0, 0].scatter(f_values[mask], k_values[mask], 
                           c=cluster_colors[i], alpha=0.7, s=25,
                           label=f'{cluster_names[i]} ({count} samples)')
    
    axes[0, 0].set_xlabel('f parameter', fontsize=12)
    axes[0, 0].set_ylabel('k parameter', fontsize=12)
    axes[0, 0].set_title('Clustering Results in f-k Space\n(k=4, 1500 samples)', fontsize=14, fontweight='bold')
    axes[0, 0].legend(loc='upper right', fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].invert_yaxis()
    
    # 2. PCA可視化
    for i, cluster in enumerate(np.unique(cluster_labels)):
        mask = cluster_labels == cluster
        count = np.sum(mask)
        axes[0, 1].scatter(latent_2d_pca[mask, 0], latent_2d_pca[mask, 1], 
                           c=cluster_colors[i], alpha=0.7, s=25,
                           label=f'{cluster_names[i]} ({count})')
    
    axes[0, 1].set_xlabel('PCA Component 1', fontsize=12)
    axes[0, 1].set_ylabel('PCA Component 2', fontsize=12)
    axes[0, 1].set_title('PCA Visualization of Latent Space\n(k=4, 1500 samples)', fontsize=14, fontweight='bold')
    axes[0, 1].legend(loc='upper right', fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. t-SNE可視化
    for i, cluster in enumerate(np.unique(cluster_labels)):
        mask = cluster_labels == cluster
        count = np.sum(mask)
        axes[1, 0].scatter(latent_2d_tsne[mask, 0], latent_2d_tsne[mask, 1], 
                           c=cluster_colors[i], alpha=0.7, s=25,
                           label=f'{cluster_names[i]} ({count})')
    
    axes[1, 0].set_xlabel('t-SNE Component 1', fontsize=12)
    axes[1, 0].set_ylabel('t-SNE Component 2', fontsize=12)
    axes[1, 0].set_title('t-SNE Visualization of Latent Space\n(k=4, 1500 samples)', fontsize=14, fontweight='bold')
    axes[1, 0].legend(loc='upper right', fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. f-k平面のヒートマップ（k=4用に最適化）
    f_unique = np.unique(f_values)
    k_unique = np.unique(k_values)
    
    heatmap_data = np.full((len(f_unique), len(k_unique)), np.nan)
    
    for i, f in enumerate(f_values):
        k = k_values[i]
        cluster = cluster_labels[i]
        
        f_idx = np.where(f_unique == f)[0][0]
        k_idx = np.where(k_unique == k)[0][0]
        heatmap_data[f_idx, k_idx] = cluster
    
    # k=4用のカスタムカラーマップ
    from matplotlib.colors import ListedColormap
    custom_cmap = ListedColormap(cluster_colors)
    
    im = axes[1, 1].imshow(heatmap_data, cmap=custom_cmap, aspect='auto', origin='upper', vmin=0, vmax=3)
    
    # ヒートマップの軸設定（k=4なので詳細に表示）
    step_k = max(1, len(k_unique) // 8)
    step_f = max(1, len(f_unique) // 8)
    
    axes[1, 1].set_xticks(range(0, len(k_unique), step_k))
    axes[1, 1].set_yticks(range(0, len(f_unique), step_f))
    axes[1, 1].set_xticklabels([f'{k:.4f}' for k in k_unique[::step_k]], rotation=45, fontsize=9)
    axes[1, 1].set_yticklabels([f'{f:.4f}' for f in f_unique[::step_f]], fontsize=9)
    axes[1, 1].set_xlabel('k parameter', fontsize=12)
    axes[1, 1].set_ylabel('f parameter', fontsize=12)
    axes[1, 1].set_title('Cluster Heatmap in f-k Space\n(k=4, 1500 samples)', fontsize=14, fontweight='bold')
    axes[1, 1].invert_yaxis()
    
    # k=4用の凡例（4つなので全て表示）
    legend_elements = [Patch(facecolor=cluster_colors[i], 
                           label=cluster_names[i]) 
                     for i in range(4)]
    axes[1, 1].legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    # 全体のタイトル
    fig.suptitle(f'Gray-Scott Clustering Results (k=4, 1500 samples)\nSilhouette Score: {results["silhouette_score"]:.3f}', 
                 fontsize=16, fontweight='bold', y=0.95)
    
    plt.tight_layout()
    
    # 保存パスの修正
    if not save_name.startswith('../results/'):
        save_name = f'../results/{save_name}'
    
    plt.savefig(save_name, dpi=dpi, bbox_inches='tight')
    plt.show()
    
    print(f"✅ Saved: {save_name}")

def analyze_k4_clusters(results):
    """k=4クラスターの詳細分析"""
    
    print(f"\n📊 k=4クラスター詳細分析")
    print("=" * 50)
    
    f_values = results['f_values']
    k_values = results['k_values']
    cluster_labels = results['cluster_labels']
    cluster_counts = results['cluster_counts']
    
    cluster_names = ['Pattern A', 'Pattern B', 'Pattern C', 'Pattern D']
    
    for cluster in range(4):
        mask = cluster_labels == cluster
        count = np.sum(mask)
        
        f_mean = f_values[mask].mean()
        k_mean = k_values[mask].mean()
        f_std = f_values[mask].std()
        k_std = k_values[mask].std()
        f_range = f_values[mask].max() - f_values[mask].min()
        k_range = k_values[mask].max() - k_values[mask].min()
        
        print(f"\n🎯 {cluster_names[cluster]} (Cluster {cluster}):")
        print(f"   サンプル数: {count} ({count/len(cluster_labels)*100:.1f}%)")
        print(f"   f値: {f_mean:.4f} ± {f_std:.4f} (範囲: {f_range:.4f})")
        print(f"   k値: {k_mean:.4f} ± {k_std:.4f} (範囲: {k_range:.4f})")
        
        # パターンの特徴推定
        if f_mean < 0.025:
            pattern_type = "安定パターン (低f値)"
        elif f_mean > 0.045:
            pattern_type = "動的パターン (高f値)"
        elif k_mean < 0.050:
            pattern_type = "拡散パターン (低k値)"
        else:
            pattern_type = "複雑パターン (高k値)"
        
        print(f"   推定パターン: {pattern_type}")

def print_k4_comparison():
    """k=4と他のクラスター数との比較"""
    
    print(f"\n📈 k=4 vs 他のクラスター数比較")
    print("=" * 50)
    print(f"🎯 k=4の利点:")
    print(f"   ✅ 解釈しやすい4つの主要パターン")
    print(f"   ✅ バランスの良いクラスターサイズ")
    print(f"   ✅ 十分な統計的信頼性")
    print(f"   ✅ 可視化での色分けが見やすい")
    
    print(f"\n🔄 他の選択肢:")
    print(f"   k=2: より大まかな分類（最高シルエットスコア）")
    print(f"   k=20: 従来の375サンプル時代との比較用")
    print(f"   k=30+: 非常に細かい分類（研究用）")

def main():
    """メイン実行関数"""
    
    print("🎨 Gray-Scott k=4最適クラスター可視化")
    print("=" * 50)
    print(f"🔧 Matplotlib version: {mpl.__version__}")
    
    # データ処理
    results = load_and_process_k4_data()
    if results is None:
        return
    
    # k=4統合可視化作成
    create_k4_visualization(results)
    
    # クラスター詳細分析
    analyze_k4_clusters(results)
    
    # 比較情報表示
    print_k4_comparison()
    
    print(f"\n🎉 k=4可視化完了!")
    print(f"📁 保存ファイル: gray_scott_clustering_results_k4_1500samples.png")
    print(f"🎯 最適なバランス重視クラスター数での分析結果です")

if __name__ == "__main__":
    main() 