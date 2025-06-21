#!/usr/bin/env python3
"""
Phase 2 クラスタリング結果の可視化
Google Colab で実行されたPhase 2の結果を表示する
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import os

def load_phase2_results():
    """Phase 2の結果を読み込む"""
    results_path = 'results/phase2_results_gpu.pkl'
    
    if not os.path.exists(results_path):
        print(f"エラー: {results_path} が見つかりません")
        return None
    
    try:
        with open(results_path, 'rb') as f:
            results = pickle.load(f)
        print("Phase 2結果の読み込みが完了しました")
        return results
    except Exception as e:
        print(f"結果の読み込みエラー: {e}")
        return None

def print_results_summary(results):
    """結果の概要を表示"""
    print("\n" + "="*50)
    print("Phase 2 クラスタリング結果 概要")
    print("="*50)
    
    if 'latent_vectors' in results:
        print(f"潜在ベクトル数: {len(results['latent_vectors'])}")
        print(f"潜在次元: {results['latent_vectors'].shape[1] if len(results['latent_vectors'].shape) > 1 else 'N/A'}")
    
    if 'cluster_labels' in results:
        unique_labels = np.unique(results['cluster_labels'])
        print(f"クラスタ数: {len(unique_labels)}")
        print(f"クラスタラベル: {unique_labels}")
        
        # 各クラスタのサンプル数
        for label in unique_labels:
            count = np.sum(results['cluster_labels'] == label)
            percentage = (count / len(results['cluster_labels'])) * 100
            print(f"  クラスタ {label}: {count}サンプル ({percentage:.1f}%)")
    
    # 評価指標
    if 'metrics' in results:
        metrics = results['metrics']
        print(f"\n評価指標:")
        if 'silhouette_score' in metrics:
            print(f"  シルエットスコア: {metrics['silhouette_score']:.4f}")
        if 'calinski_harabasz_score' in metrics:
            print(f"  Calinski-Harabasz スコア: {metrics['calinski_harabasz_score']:.2f}")
        if 'davies_bouldin_score' in metrics:
            print(f"  Davies-Bouldin スコア: {metrics['davies_bouldin_score']:.4f}")
    
    # 学習情報
    if 'training_info' in results:
        info = results['training_info']
        print(f"\n学習情報:")
        if 'final_loss' in info:
            print(f"  最終損失: {info['final_loss']:.6f}")
        if 'training_time' in info:
            print(f"  学習時間: {info['training_time']:.1f}秒")
        if 'epochs' in info:
            print(f"  エポック数: {info['epochs']}")

def visualize_clustering_results(results):
    """クラスタリング結果を可視化"""
    if 'latent_vectors' not in results or 'cluster_labels' not in results:
        print("可視化に必要なデータが不足しています")
        return
    
    latent_vectors = results['latent_vectors']
    cluster_labels = results['cluster_labels']
    f_values = results.get('f_values', None)
    k_values = results.get('k_values', None)
    
    # 6つのサブプロットを作成（3x2レイアウト）
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    fig.suptitle('Phase 2 クラスタリング結果', fontsize=16, fontweight='bold')
    
    # 1. PCA 2D可視化
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(latent_vectors)
    
    scatter = axes[0, 0].scatter(pca_result[:, 0], pca_result[:, 1], 
                                c=cluster_labels, cmap='tab10', alpha=0.7)
    axes[0, 0].set_title(f'PCA 2D (分散説明率: {pca.explained_variance_ratio_.sum():.3f})')
    axes[0, 0].set_xlabel('PC1')
    axes[0, 0].set_ylabel('PC2')
    plt.colorbar(scatter, ax=axes[0, 0])
    
    # 2. t-SNE 2D可視化
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_result = tsne.fit_transform(latent_vectors)
    
    scatter = axes[0, 1].scatter(tsne_result[:, 0], tsne_result[:, 1], 
                                c=cluster_labels, cmap='tab10', alpha=0.7)
    axes[0, 1].set_title('t-SNE 2D')
    axes[0, 1].set_xlabel('t-SNE 1')
    axes[0, 1].set_ylabel('t-SNE 2')
    plt.colorbar(scatter, ax=axes[0, 1])
    
    # 3. f-k パラメータ空間での分布
    if f_values is not None and k_values is not None:
        scatter = axes[1, 0].scatter(f_values, k_values, c=cluster_labels, 
                                   cmap='tab10', alpha=0.7, s=20)
        axes[1, 0].set_title('f-k パラメータ空間でのクラスタ分布')
        axes[1, 0].set_xlabel('f (feed rate)')
        axes[1, 0].set_ylabel('k (kill rate)')
        axes[1, 0].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[1, 0])
        
        # f-k空間の範囲を表示
        f_range = f"f: {f_values.min():.4f} - {f_values.max():.4f}"
        k_range = f"k: {k_values.min():.4f} - {k_values.max():.4f}"
        axes[1, 0].text(0.02, 0.98, f"{f_range}\n{k_range}", 
                       transform=axes[1, 0].transAxes, 
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    else:
        axes[1, 0].text(0.5, 0.5, 'f-k データが利用できません', 
                       ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('f-k パラメータ空間')
    
    # 4. f-k空間のヒートマップ（クラスタ密度）
    if f_values is not None and k_values is not None:
        # 各クラスタ別の散布図
        unique_labels = np.unique(cluster_labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        axes[1, 1].set_title('f-k空間 クラスタ別分布')
        axes[1, 1].set_xlabel('f (feed rate)')
        axes[1, 1].set_ylabel('k (kill rate)')
        
        for i, label in enumerate(unique_labels):
            mask = cluster_labels == label
            if np.sum(mask) > 0:
                f_cluster = f_values[mask]
                k_cluster = k_values[mask]
                axes[1, 1].scatter(f_cluster, k_cluster, 
                                 color=colors[i], alpha=0.7, s=25,
                                 label=f'Cluster {label} ({np.sum(mask)})')
        
        axes[1, 1].legend(loc='upper right', fontsize=9)
        axes[1, 1].grid(True, alpha=0.3)
        
        # f-k空間の統計情報を追加
        stats_text = f"Total samples: {len(f_values)}\n"
        stats_text += f"f range: [{f_values.min():.4f}, {f_values.max():.4f}]\n"
        stats_text += f"k range: [{k_values.min():.4f}, {k_values.max():.4f}]"
        axes[1, 1].text(0.02, 0.02, stats_text, 
                       transform=axes[1, 1].transAxes, 
                       verticalalignment='bottom',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    else:
        axes[1, 1].text(0.5, 0.5, 'f-k データが利用できません', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('f-k空間 密度分布')
    
    # 5. クラスタサイズ分布
    unique_labels, counts = np.unique(cluster_labels, return_counts=True)
    bars = axes[2, 0].bar(unique_labels, counts, color='skyblue', alpha=0.7)
    axes[2, 0].set_title('クラスタサイズ分布')
    axes[2, 0].set_xlabel('クラスタラベル')
    axes[2, 0].set_ylabel('サンプル数')
    
    # バーの上に数値を表示
    for bar, count in zip(bars, counts):
        axes[2, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                       str(count), ha='center', va='bottom')
    
    # 6. 評価指標表示
    axes[2, 1].axis('off')
    metrics_text = "評価指標\n\n"
    
    if 'metrics' in results:
        metrics = results['metrics']
        if 'silhouette_score' in metrics:
            metrics_text += f"シルエットスコア: {metrics['silhouette_score']:.4f}\n"
        if 'calinski_harabasz_score' in metrics:
            metrics_text += f"Calinski-Harabasz: {metrics['calinski_harabasz_score']:.2f}\n"
        if 'davies_bouldin_score' in metrics:
            metrics_text += f"Davies-Bouldin: {metrics['davies_bouldin_score']:.4f}\n"
    
    # Phase 1との比較情報があれば表示
    if 'comparison' in results:
        comp = results['comparison']
        metrics_text += f"\nPhase 1との比較:\n"
        if 'silhouette_improvement' in comp:
            metrics_text += f"シルエット改善: +{comp['silhouette_improvement']:.1f}%\n"
        if 'phase1_silhouette' in comp:
            metrics_text += f"Phase 1: {comp['phase1_silhouette']:.4f}\n"
            metrics_text += f"Phase 2: {comp['phase2_silhouette']:.4f}\n"
    
    axes[2, 1].text(0.1, 0.9, metrics_text, transform=axes[2, 1].transAxes,
                   fontsize=12, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    # レイアウト調整
    plt.tight_layout()
    
    # 結果を保存
    output_path = 'results/phase2_clustering_visualization_with_fk.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n可視化結果を保存しました: {output_path}")
    
    plt.show()

def compare_with_phase1(results):
    """Phase 1との比較"""
    phase1_path = 'results/phase1_comparison_results.pkl'
    
    if not os.path.exists(phase1_path):
        print("Phase 1の結果が見つかりません。比較をスキップします。")
        return
    
    try:
        with open(phase1_path, 'rb') as f:
            phase1_results = pickle.load(f)
        
        print("\n" + "="*50)
        print("Phase 1 vs Phase 2 比較")
        print("="*50)
        
        # Phase 1の結果
        if 'phase1_metrics' in phase1_results:
            p1_metrics = phase1_results['phase1_metrics']
            print(f"Phase 1 シルエットスコア: {p1_metrics.get('silhouette_score', 'N/A')}")
        
        # Phase 2の結果
        if 'metrics' in results:
            p2_metrics = results['metrics']
            print(f"Phase 2 シルエットスコア: {p2_metrics.get('silhouette_score', 'N/A')}")
            
            # 改善率を計算
            if 'phase1_metrics' in phase1_results and 'silhouette_score' in p1_metrics and 'silhouette_score' in p2_metrics:
                p1_score = p1_metrics['silhouette_score']
                p2_score = p2_metrics['silhouette_score']
                improvement = ((p2_score - p1_score) / p1_score) * 100
                print(f"改善率: {improvement:+.1f}%")
                
                if improvement > 15:
                    print("🎉 目標の15%改善を達成しました！")
                else:
                    print("📊 更なる改善の余地があります")
    
    except Exception as e:
        print(f"Phase 1との比較エラー: {e}")

def main():
    """メイン実行関数"""
    print("Phase 2 クラスタリング結果の可視化を開始します...")
    
    # 結果を読み込み
    results = load_phase2_results()
    if results is None:
        return
    
    # 結果の概要を表示
    print_results_summary(results)
    
    # 可視化
    visualize_clustering_results(results)
    
    # Phase 1との比較
    compare_with_phase1(results)
    
    print("\n✅ 可視化が完了しました！")

if __name__ == "__main__":
    main() 