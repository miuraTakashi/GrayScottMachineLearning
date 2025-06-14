#!/usr/bin/env python3
"""
1500サンプル用 最適クラスター数解析
より多くのクラスター数での分析を実施
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib as mpl

def load_1500_data():
    """1500サンプルデータを読み込み"""
    
    print("📂 1500サンプルデータを読み込み中...")
    
    data_file = '../results/latent_representations_frames_all.pkl'
    
    if not os.path.exists(data_file):
        print(f"❌ {data_file} が見つかりません")
        return None
    
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    
    latent_vectors = data['latent_vectors']  # (1500, 128)
    f_values = data['f_values']              # 1500個
    k_values = data['k_values']              # 1500個
    
    print(f"✅ データ読み込み完了: {len(f_values)} サンプル, 潜在次元: {latent_vectors.shape[1]}")
    
    return latent_vectors, f_values, k_values

def evaluate_clustering_range(latent_vectors, min_k=2, max_k=60):
    """幅広いクラスター数でシルエット分析"""
    
    print(f"🔍 クラスター数 {min_k}-{max_k} で最適化分析中...")
    
    k_range = range(min_k, max_k + 1)
    silhouette_scores = []
    calinski_scores = []
    davies_bouldin_scores = []
    inertias = []
    
    for k in k_range:
        print(f"   k={k:2d}: ", end="", flush=True)
        
        # K-means実行
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(latent_vectors)
        
        # 各種評価指標
        sil_score = silhouette_score(latent_vectors, cluster_labels)
        cal_score = calinski_harabasz_score(latent_vectors, cluster_labels)
        db_score = davies_bouldin_score(latent_vectors, cluster_labels)
        inertia = kmeans.inertia_
        
        silhouette_scores.append(sil_score)
        calinski_scores.append(cal_score)
        davies_bouldin_scores.append(db_score)
        inertias.append(inertia)
        
        print(f"silhouette={sil_score:.3f}, calinski={cal_score:.1f}, db={db_score:.3f}")
    
    return {
        'k_range': list(k_range),
        'silhouette_scores': silhouette_scores,
        'calinski_scores': calinski_scores,
        'davies_bouldin_scores': davies_bouldin_scores,
        'inertias': inertias
    }

def plot_clustering_analysis(results):
    """クラスタリング分析結果をプロット"""
    
    print("📊 クラスタリング分析結果を可視化中...")
    
    k_range = results['k_range']
    sil_scores = results['silhouette_scores']
    cal_scores = results['calinski_scores']
    db_scores = results['davies_bouldin_scores']
    inertias = results['inertias']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. シルエットスコア
    axes[0, 0].plot(k_range, sil_scores, 'bo-', linewidth=2, markersize=6)
    axes[0, 0].set_xlabel('Number of Clusters (k)')
    axes[0, 0].set_ylabel('Silhouette Score')
    axes[0, 0].set_title('Silhouette Analysis (1500 samples)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 最高スコアをマーク
    best_sil_idx = np.argmax(sil_scores)
    best_sil_k = k_range[best_sil_idx]
    best_sil_score = sil_scores[best_sil_idx]
    axes[0, 0].plot(best_sil_k, best_sil_score, 'ro', markersize=10, 
                    label=f'Best: k={best_sil_k} (score={best_sil_score:.3f})')
    axes[0, 0].legend()
    
    # 2. Calinski-Harabasz Index
    axes[0, 1].plot(k_range, cal_scores, 'go-', linewidth=2, markersize=6)
    axes[0, 1].set_xlabel('Number of Clusters (k)')
    axes[0, 1].set_ylabel('Calinski-Harabasz Index')
    axes[0, 1].set_title('Calinski-Harabasz Analysis')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 最高スコアをマーク
    best_cal_idx = np.argmax(cal_scores)
    best_cal_k = k_range[best_cal_idx]
    best_cal_score = cal_scores[best_cal_idx]
    axes[0, 1].plot(best_cal_k, best_cal_score, 'ro', markersize=10,
                    label=f'Best: k={best_cal_k} (score={best_cal_score:.1f})')
    axes[0, 1].legend()
    
    # 3. Davies-Bouldin Index (低い方が良い)
    axes[1, 0].plot(k_range, db_scores, 'mo-', linewidth=2, markersize=6)
    axes[1, 0].set_xlabel('Number of Clusters (k)')
    axes[1, 0].set_ylabel('Davies-Bouldin Index')
    axes[1, 0].set_title('Davies-Bouldin Analysis (lower is better)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 最低スコアをマーク
    best_db_idx = np.argmin(db_scores)
    best_db_k = k_range[best_db_idx]
    best_db_score = db_scores[best_db_idx]
    axes[1, 0].plot(best_db_k, best_db_score, 'ro', markersize=10,
                    label=f'Best: k={best_db_k} (score={best_db_score:.3f})')
    axes[1, 0].legend()
    
    # 4. エルボー法
    axes[1, 1].plot(k_range, inertias, 'co-', linewidth=2, markersize=6)
    axes[1, 1].set_xlabel('Number of Clusters (k)')
    axes[1, 1].set_ylabel('Inertia (Within-cluster sum of squares)')
    axes[1, 1].set_title('Elbow Method')
    axes[1, 1].grid(True, alpha=0.3)
    
    # エルボーポイントを推定
    elbow_k = estimate_elbow_point(k_range, inertias)
    elbow_inertia = inertias[k_range.index(elbow_k)]
    axes[1, 1].plot(elbow_k, elbow_inertia, 'ro', markersize=10,
                    label=f'Elbow: k={elbow_k}')
    axes[1, 1].legend()
    
    plt.suptitle('Optimal Cluster Analysis for 1500 Samples', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # 保存
    output_file = '../results/optimal_cluster_analysis_1500samples.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✅ Saved: {output_file}")
    
    return {
        'best_silhouette': (best_sil_k, best_sil_score),
        'best_calinski': (best_cal_k, best_cal_score),
        'best_davies_bouldin': (best_db_k, best_db_score),
        'elbow_point': elbow_k
    }

def estimate_elbow_point(k_range, inertias):
    """エルボーポイントを推定"""
    
    # 二次差分を計算
    diffs = np.diff(inertias)
    diff2 = np.diff(diffs)
    
    # 最大の変化点を見つける
    elbow_idx = np.argmax(diff2) + 2  # インデックス調整
    elbow_k = k_range[elbow_idx] if elbow_idx < len(k_range) else k_range[-1]
    
    return elbow_k

def analyze_top_candidates(latent_vectors, f_values, k_values, candidate_ks):
    """上位候補のクラスター数で詳細分析"""
    
    print(f"\n📈 上位候補クラスター数の詳細分析...")
    
    results = {}
    
    for k in candidate_ks:
        print(f"\n🔍 k={k} の詳細分析:")
        
        # クラスタリング実行
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(latent_vectors)
        
        # 評価指標
        sil_score = silhouette_score(latent_vectors, cluster_labels)
        cal_score = calinski_harabasz_score(latent_vectors, cluster_labels)
        db_score = davies_bouldin_score(latent_vectors, cluster_labels)
        
        # クラスター分布
        unique, counts = np.unique(cluster_labels, return_counts=True)
        min_cluster_size = counts.min()
        max_cluster_size = counts.max()
        mean_cluster_size = counts.mean()
        
        print(f"   シルエットスコア: {sil_score:.3f}")
        print(f"   Calinski-Harabasz: {cal_score:.1f}")
        print(f"   Davies-Bouldin: {db_score:.3f}")
        print(f"   クラスターサイズ: min={min_cluster_size}, max={max_cluster_size}, mean={mean_cluster_size:.1f}")
        
        # パラメータ範囲の分析
        f_ranges = []
        k_ranges = []
        for cluster in unique:
            mask = cluster_labels == cluster
            f_range = f_values[mask].max() - f_values[mask].min()
            k_range = k_values[mask].max() - k_values[mask].min()
            f_ranges.append(f_range)
            k_ranges.append(k_range)
        
        mean_f_range = np.mean(f_ranges)
        mean_k_range = np.mean(k_ranges)
        print(f"   平均パラメータ範囲: f={mean_f_range:.4f}, k={mean_k_range:.4f}")
        
        results[k] = {
            'silhouette': sil_score,
            'calinski': cal_score,
            'davies_bouldin': db_score,
            'cluster_sizes': counts,
            'mean_f_range': mean_f_range,
            'mean_k_range': mean_k_range
        }
    
    return results

def print_recommendations(best_metrics, detailed_results):
    """推奨クラスター数を表示"""
    
    print(f"\n🎯 1500サンプル用 推奨クラスター数")
    print("=" * 50)
    
    # 各手法の結果
    sil_k, sil_score = best_metrics['best_silhouette']
    cal_k, cal_score = best_metrics['best_calinski']
    db_k, db_score = best_metrics['best_davies_bouldin']
    elbow_k = best_metrics['elbow_point']
    
    print(f"📊 各手法による推奨値:")
    print(f"   シルエット分析: k={sil_k} (スコア: {sil_score:.3f})")
    print(f"   Calinski-Harabasz: k={cal_k} (スコア: {cal_score:.1f})")
    print(f"   Davies-Bouldin: k={db_k} (スコア: {db_score:.3f})")
    print(f"   エルボー法: k={elbow_k}")
    
    # 推奨順位
    candidates = [sil_k, cal_k, db_k, elbow_k]
    unique_candidates = sorted(list(set(candidates)), reverse=True)
    
    print(f"\n🏆 総合推奨順位:")
    for i, k in enumerate(unique_candidates[:5]):
        if k in detailed_results:
            result = detailed_results[k]
            print(f"   {i+1}. k={k}: シルエット={result['silhouette']:.3f}, "
                  f"クラスター平均サイズ={result['cluster_sizes'].mean():.1f}")
    
    print(f"\n💡 375サンプル時代との比較:")
    print(f"   375サンプル: k=20 (シルエット=0.551)")
    print(f"   1500サンプル: k={sil_k} (シルエット={sil_score:.3f})")
    
    if sil_k > 20:
        print(f"   🔺 より細かい分類が可能: {sil_k-20}個多いクラスター")
    elif sil_k < 20:
        print(f"   🔻 よりシンプルな分類が最適: {20-sil_k}個少ないクラスター")
    else:
        print(f"   ➡️ 同じクラスター数が最適")

def main():
    """メイン実行関数"""
    
    print("🔍 1500サンプル用 最適クラスター数解析")
    print("=" * 50)
    print(f"🔧 Matplotlib version: {mpl.__version__}")
    
    # データ読み込み
    latent_vectors, f_values, k_values = load_1500_data()
    if latent_vectors is None:
        return
    
    # 幅広いクラスター数で評価
    results = evaluate_clustering_range(latent_vectors, min_k=2, max_k=60)
    
    # 結果をプロット
    best_metrics = plot_clustering_analysis(results)
    
    # 上位候補の詳細分析
    sil_k = best_metrics['best_silhouette'][0]
    cal_k = best_metrics['best_calinski'][0]
    db_k = best_metrics['best_davies_bouldin'][0]
    elbow_k = best_metrics['elbow_point']
    
    candidates = sorted(list(set([sil_k, cal_k, db_k, elbow_k, 20, 30, 40])))[:7]  # 375時代の20も含める
    detailed_results = analyze_top_candidates(latent_vectors, f_values, k_values, candidates)
    
    # 推奨事項表示
    print_recommendations(best_metrics, detailed_results)
    
    print(f"\n🎉 最適クラスター数解析完了!")
    print(f"📁 保存ファイル: optimal_cluster_analysis_1500samples.png")

if __name__ == "__main__":
    main() 