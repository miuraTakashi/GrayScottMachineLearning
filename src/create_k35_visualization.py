#!/usr/bin/env python3
"""
k=35 詳細クラスター数での1500サンプル可視化
細かいパターン分類での統合可視化
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
from matplotlib.colors import ListedColormap
import seaborn as sns

def load_and_process_k35_data():
    """1500サンプルデータをk=35でクラスタリング実行"""
    
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
    
    # k=35でクラスタリング実行
    print("🔍 k=35でクラスタリング実行中...")
    n_clusters = 35
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(latent_vectors)
    silhouette = silhouette_score(latent_vectors, cluster_labels)
    print(f"   シルエットスコア: {silhouette:.3f}")
    
    # クラスター分布表示
    unique, counts = np.unique(cluster_labels, return_counts=True)
    print(f"   クラスター分布統計:")
    print(f"     平均サンプル数: {counts.mean():.1f}")
    print(f"     最大クラスター: {counts.max()} samples")
    print(f"     最小クラスター: {counts.min()} samples")
    print(f"     標準偏差: {counts.std():.1f}")
    
    # サイズの大きいクラスターを表示
    large_clusters = [(i, count) for i, count in enumerate(counts) if count > counts.mean() + counts.std()]
    if large_clusters:
        print(f"   大きなクラスター:")
        for cluster_id, count in large_clusters:
            print(f"     Cluster {cluster_id}: {count} samples ({count/len(cluster_labels)*100:.1f}%)")
    
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
        'cluster_counts': counts,
        'n_clusters': n_clusters
    }

def create_k35_visualization(results, figsize=(16, 12), dpi=300, save_name='gray_scott_clustering_results_k35_1500samples.png'):
    """k=35での統合可視化（1500サンプル版）"""
    
    f_values = results['f_values']
    k_values = results['k_values']
    cluster_labels = results['cluster_labels']
    latent_2d_pca = results['pca_result']
    latent_2d_tsne = results['tsne_result']
    n_clusters = results['n_clusters']
    
    print(f"🎨 k=35統合可視化作成中 (1500サンプル)...")
    
    # k=35用の連続カラーマップを使用
    try:
        colormap = plt.cm.viridis
    except:
        try:
            colormap = plt.cm.tab20
        except:
            colormap = plt.cm.Set3
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # 1. f-k空間でのクラスタリング結果
    scatter1 = axes[0, 0].scatter(f_values, k_values, c=cluster_labels, 
                                  cmap=colormap, alpha=0.7, s=20)
    axes[0, 0].set_xlabel('f parameter', fontsize=12)
    axes[0, 0].set_ylabel('k parameter', fontsize=12)
    axes[0, 0].set_title('Clustering Results in f-k Space\n(k=35, 1500 samples)', fontsize=14, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].invert_yaxis()
    
    # カラーバー（クラスター数が多いので）
    plt.colorbar(scatter1, ax=axes[0, 0], label='Cluster ID', shrink=0.8)
    
    # 2. PCA可視化
    scatter2 = axes[0, 1].scatter(latent_2d_pca[:, 0], latent_2d_pca[:, 1], 
                                  c=cluster_labels, cmap=colormap, alpha=0.7, s=20)
    axes[0, 1].set_xlabel('PCA Component 1', fontsize=12)
    axes[0, 1].set_ylabel('PCA Component 2', fontsize=12)
    axes[0, 1].set_title('PCA Visualization of Latent Space\n(k=35, 1500 samples)', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    plt.colorbar(scatter2, ax=axes[0, 1], label='Cluster ID', shrink=0.8)
    
    # 3. t-SNE可視化
    scatter3 = axes[1, 0].scatter(latent_2d_tsne[:, 0], latent_2d_tsne[:, 1], 
                                  c=cluster_labels, cmap=colormap, alpha=0.7, s=20)
    axes[1, 0].set_xlabel('t-SNE Component 1', fontsize=12)
    axes[1, 0].set_ylabel('t-SNE Component 2', fontsize=12)
    axes[1, 0].set_title('t-SNE Visualization of Latent Space\n(k=35, 1500 samples)', fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    plt.colorbar(scatter3, ax=axes[1, 0], label='Cluster ID', shrink=0.8)
    
    # 4. f-k平面のクラスターヒートマップ（k=35専用）
    f_unique = np.unique(f_values)
    k_unique = np.unique(k_values)
    
    heatmap_data = np.full((len(f_unique), len(k_unique)), np.nan)
    
    for i, f in enumerate(f_values):
        k = k_values[i]
        cluster = cluster_labels[i]
        
        f_idx = np.where(f_unique == f)[0][0]
        k_idx = np.where(k_unique == k)[0][0]
        heatmap_data[f_idx, k_idx] = cluster
    
    # k=35用の連続カラーマップでヒートマップ表示
    im = axes[1, 1].imshow(heatmap_data, cmap=colormap, aspect='auto', origin='upper', 
                           vmin=0, vmax=n_clusters-1)
    
    # ヒートマップの軸設定
    step_k = max(1, len(k_unique) // 8)
    step_f = max(1, len(f_unique) // 8)
    
    axes[1, 1].set_xticks(range(0, len(k_unique), step_k))
    axes[1, 1].set_yticks(range(0, len(f_unique), step_f))
    axes[1, 1].set_xticklabels([f'{k:.4f}' for k in k_unique[::step_k]], rotation=45, fontsize=9)
    axes[1, 1].set_yticklabels([f'{f:.4f}' for f in f_unique[::step_f]], fontsize=9)
    axes[1, 1].set_xlabel('k parameter', fontsize=12)
    axes[1, 1].set_ylabel('f parameter', fontsize=12)
    axes[1, 1].set_title('Cluster Heatmap in f-k Space\n(k=35, 1500 samples)', fontsize=14, fontweight='bold')
    axes[1, 1].invert_yaxis()
    
    # ヒートマップ用のカラーバー
    cbar = plt.colorbar(im, ax=axes[1, 1], shrink=0.6)
    cbar.set_label('Cluster ID', fontsize=10)
    
    # 全体のタイトル
    fig.suptitle(f'Gray-Scott Clustering Results (k=35, 1500 samples)\nSilhouette Score: {results["silhouette_score"]:.3f}', 
                 fontsize=16, fontweight='bold', y=0.95)
    
    plt.tight_layout()
    
    # 保存パスの修正
    if not save_name.startswith('../results/'):
        save_name = f'../results/{save_name}'
    
    plt.savefig(save_name, dpi=dpi, bbox_inches='tight')
    plt.show()
    
    print(f"✅ Saved: {save_name}")

def analyze_k35_clusters(results):
    """k=35クラスターの詳細分析"""
    
    print(f"\n📊 k=35クラスター詳細分析")
    print("=" * 60)
    
    f_values = results['f_values']
    k_values = results['k_values']
    cluster_labels = results['cluster_labels']
    cluster_counts = results['cluster_counts']
    
    # 統計サマリー
    print(f"🔢 統計サマリー:")
    print(f"   総クラスター数: {len(cluster_counts)}")
    print(f"   平均クラスターサイズ: {cluster_counts.mean():.1f} ± {cluster_counts.std():.1f}")
    print(f"   最大クラスター: {cluster_counts.max()} samples")
    print(f"   最小クラスター: {cluster_counts.min()} samples")
    print(f"   中央値: {np.median(cluster_counts):.1f}")
    
    # 大きなクラスター（平均+標準偏差以上）の分析
    threshold = cluster_counts.mean() + cluster_counts.std()
    large_clusters = []
    
    print(f"\n🎯 主要クラスター (>{threshold:.0f} samples):")
    
    for cluster in range(len(cluster_counts)):
        if cluster_counts[cluster] > threshold:
            mask = cluster_labels == cluster
            count = np.sum(mask)
            
            f_mean = f_values[mask].mean()
            k_mean = k_values[mask].mean()
            f_std = f_values[mask].std()
            k_std = k_values[mask].std()
            
            large_clusters.append({
                'id': cluster,
                'count': count,
                'f_mean': f_mean,
                'k_mean': k_mean,
                'f_std': f_std,
                'k_std': k_std
            })
            
            print(f"   Cluster {cluster:2d}: {count:3d} samples ({count/len(cluster_labels)*100:.1f}%)")
            print(f"      f: {f_mean:.4f} ± {f_std:.4f}")
            print(f"      k: {k_mean:.4f} ± {k_std:.4f}")
    
    # f-k空間での分布分析
    print(f"\n🌍 f-k空間分布分析:")
    f_ranges = []
    k_ranges = []
    
    for cluster in range(len(cluster_counts)):
        mask = cluster_labels == cluster
        if np.sum(mask) > 5:  # 最低5サンプルのクラスターのみ
            f_range = f_values[mask].max() - f_values[mask].min()
            k_range = k_values[mask].max() - k_values[mask].min()
            f_ranges.append(f_range)
            k_ranges.append(k_range)
    
    print(f"   f値範囲の平均: {np.mean(f_ranges):.4f}")
    print(f"   k値範囲の平均: {np.mean(k_ranges):.4f}")
    print(f"   コンパクトなクラスター数: {len([r for r in f_ranges if r < 0.01])} / {len(f_ranges)}")

def create_k35_heatmap(results, figsize=(12, 8), save_name='gray_scott_k35_heatmap_1500samples.png'):
    """k=35専用のf-k空間ヒートマップ"""
    
    f_values = results['f_values']
    k_values = results['k_values']
    cluster_labels = results['cluster_labels']
    
    print(f"🗺️  k=35専用ヒートマップ作成中...")
    
    # f-k空間のグリッド作成
    f_unique = np.unique(f_values)
    k_unique = np.unique(k_values)
    
    heatmap_data = np.full((len(f_unique), len(k_unique)), np.nan)
    
    for i, f in enumerate(f_values):
        k = k_values[i]
        cluster = cluster_labels[i]
        
        f_idx = np.where(f_unique == f)[0][0]
        k_idx = np.where(k_unique == k)[0][0]
        heatmap_data[f_idx, k_idx] = cluster
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # k=35用の連続カラーマップ
    im = ax.imshow(heatmap_data, cmap='viridis', aspect='auto', origin='upper', vmin=0, vmax=34)
    
    # 軸設定
    step_k = max(1, len(k_unique) // 12)
    step_f = max(1, len(f_unique) // 12)
    
    ax.set_xticks(range(0, len(k_unique), step_k))
    ax.set_yticks(range(0, len(f_unique), step_f))
    ax.set_xticklabels([f'{k:.4f}' for k in k_unique[::step_k]], rotation=45, fontsize=10)
    ax.set_yticklabels([f'{f:.4f}' for f in f_unique[::step_f]], fontsize=10)
    ax.set_xlabel('k parameter', fontsize=12)
    ax.set_ylabel('f parameter', fontsize=12)
    ax.set_title(f'Detailed Cluster Heatmap in f-k Space\n(k=35, 1500 samples)', fontsize=14, fontweight='bold')
    
    # カラーバー
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Cluster ID', fontsize=12)
    
    plt.tight_layout()
    
    # 保存
    if not save_name.startswith('../results/'):
        save_name = f'../results/{save_name}'
    
    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✅ Heatmap saved: {save_name}")

def print_k35_comparison():
    """k=35と他のクラスター数との比較"""
    
    print(f"\n📈 k=35 vs 他のクラスター数比較")
    print("=" * 60)
    print(f"🎯 k=35の特徴:")
    print(f"   ✅ 非常に詳細なパターン分類")
    print(f"   ✅ 平均43サンプル/クラスター（統計的に有意）")
    print(f"   ✅ 細かいパターンの違いを検出")
    print(f"   ⚠️  解釈の複雑さが増加")
    
    print(f"\n🔄 他の選択肢との位置づけ:")
    print(f"   k=2:  大まかな二分類（最高シルエット）")
    print(f"   k=4:  バランス重視（エルボー法推奨）")
    print(f"   k=20: 従来の375サンプル相当")
    print(f"   k=35: 詳細研究用（現在）")
    print(f"   k=50+: 過細分化の危険性")

def main():
    """メイン実行関数"""
    
    print("🎨 Gray-Scott k=35詳細クラスター可視化")
    print("=" * 60)
    print(f"🔧 Matplotlib version: {mpl.__version__}")
    
    # データ処理
    results = load_and_process_k35_data()
    if results is None:
        return
    
    # k=35統合可視化作成
    create_k35_visualization(results)
    
    # k=35専用ヒートマップ作成
    create_k35_heatmap(results)
    
    # クラスター詳細分析
    analyze_k35_clusters(results)
    
    # 比較情報表示
    print_k35_comparison()
    
    print(f"\n🎉 k=35可視化完了!")
    print(f"📁 保存ファイル:")
    print(f"   - gray_scott_clustering_results_k35_1500samples.png")
    print(f"   - gray_scott_k35_heatmap_1500samples.png")
    print(f"🔬 詳細な35クラスター分析が完了しました")

if __name__ == "__main__":
    main() 