#!/usr/bin/env python3
"""
Gray-Scott Cluster Analysis Test Script
Jupyter Notebookと同等の分析機能をPythonスクリプトで実行
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.metrics import pairwise_distances
import warnings
warnings.filterwarnings('ignore')

def load_analysis_results():
    """保存された分析結果を読み込む"""
    with open('analysis_results.pkl', 'rb') as f:
        results = pickle.load(f)
    return results

def select_cluster_representatives(results, method='centroid', n_samples=3):
    """
    各クラスターの代表例を選択
    """
    latent_vectors = results['latent_vectors']
    cluster_labels = results['cluster_labels']
    filenames = results['filenames']
    f_values = results['f_values']
    k_values = results['k_values']
    
    representatives = {}
    
    for cluster_id in np.unique(cluster_labels):
        cluster_mask = cluster_labels == cluster_id
        cluster_latents = latent_vectors[cluster_mask]
        cluster_filenames = [filenames[i] for i in range(len(filenames)) if cluster_mask[i]]
        cluster_f = f_values[cluster_mask]
        cluster_k = k_values[cluster_mask]
        
        if method == 'centroid':
            # クラスター中心を計算
            centroid = np.mean(cluster_latents, axis=0)
            # 中心に最も近いサンプルを選択
            distances = pairwise_distances([centroid], cluster_latents)[0]
            closest_indices = np.argsort(distances)[:n_samples]
        
        representatives[int(cluster_id)] = {
            'filenames': [cluster_filenames[i] for i in closest_indices],
            'f_values': [cluster_f[i] for i in closest_indices],
            'k_values': [cluster_k[i] for i in closest_indices],
            'latent_vectors': [cluster_latents[i] for i in closest_indices]
        }
    
    return representatives

def analyze_cluster_characteristics(df):
    """各クラスターの特徴を詳細分析"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    unique_clusters = sorted(df['cluster'].unique())
    
    for i, cluster_id in enumerate(unique_clusters):
        if i >= 6:  # 最大6クラスターまで表示
            break
            
        cluster_data = df[df['cluster'] == cluster_id]
        
        # f-k散布図
        axes[i].scatter(cluster_data['f_value'], cluster_data['k_value'], 
                       alpha=0.7, s=50, color=plt.cm.viridis(cluster_id/len(unique_clusters)))
        axes[i].set_xlabel('f parameter')
        axes[i].set_ylabel('k parameter')
        axes[i].set_title(f'Cluster {cluster_id} (n={len(cluster_data)})')
        axes[i].invert_yaxis()  # F軸を反転
        axes[i].grid(True, alpha=0.3)
        
        # 統計情報を追加
        f_mean, f_std = cluster_data['f_value'].mean(), cluster_data['f_value'].std()
        k_mean, k_std = cluster_data['k_value'].mean(), cluster_data['k_value'].std()
        
        info_text = f'f: {f_mean:.4f}±{f_std:.4f}\nk: {k_mean:.4f}±{k_std:.4f}'
        axes[i].text(0.05, 0.95, info_text, transform=axes[i].transAxes, 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                    verticalalignment='top', fontsize=9)
    
    # 余ったサブプロットを非表示
    for j in range(i+1, 6):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('cluster_characteristics_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_representatives_summary(representatives):
    """代表例のサマリーテーブルを作成"""
    
    summary_data = []
    
    for cluster_id in sorted(representatives.keys()):
        cluster_data = representatives[cluster_id]
        
        for i, (filename, f_val, k_val) in enumerate(zip(
            cluster_data['filenames'],
            cluster_data['f_values'],
            cluster_data['k_values']
        )):
            summary_data.append({
                'Cluster': cluster_id,
                'Representative': f'Rep {i+1}',
                'Filename': filename,
                'f_value': f_val,
                'k_value': k_val,
                'f/k_ratio': f_val/k_val if k_val != 0 else np.inf
            })
    
    summary_df = pd.DataFrame(summary_data)
    return summary_df

def main():
    print("🔬 Gray-Scott Cluster Analysis Test Script")
    print("=" * 50)
    
    # データの読み込み
    try:
        results = load_analysis_results()
        print("✅ 分析結果の読み込みが完了しました")
        print(f"サンプル数: {len(results['f_values'])}")
        print(f"クラスター数: {results['n_clusters']}")
        print(f"潜在空間次元: {results['latent_vectors'].shape[1]}")
    except FileNotFoundError:
        print("❌ analysis_results.pkl が見つかりません。先に学習を実行してください。")
        return
    
    # CSVファイルからも読み込み
    try:
        df = pd.read_csv('clustering_results.csv')
        print(f"CSV データ: {len(df)} サンプル")
        print("\nデータの先頭5行:")
        print(df.head())
    except FileNotFoundError:
        print("❌ clustering_results.csv が見つかりません")
        return
    
    # クラスター統計情報
    print("\n" + "="*50)
    print("📊 クラスター統計情報")
    print("="*50)
    
    cluster_stats = df.groupby('cluster').agg({
        'f_value': ['count', 'mean', 'std', 'min', 'max'],
        'k_value': ['mean', 'std', 'min', 'max']
    }).round(4)
    
    print(cluster_stats)
    
    # クラスター分布の可視化
    print("\n📈 クラスター分布の可視化中...")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # クラスターサイズ
    cluster_counts = df['cluster'].value_counts().sort_index()
    axes[0].bar(cluster_counts.index, cluster_counts.values, 
               color=plt.cm.viridis(cluster_counts.index/len(cluster_counts)))
    axes[0].set_xlabel('Cluster ID')
    axes[0].set_ylabel('Number of Samples')
    axes[0].set_title('Cluster Size Distribution')
    axes[0].grid(True, alpha=0.3)
    
    # f-k パラメータ分布
    scatter = axes[1].scatter(df['f_value'], df['k_value'], c=df['cluster'], cmap='viridis', alpha=0.7)
    axes[1].set_xlabel('f parameter')
    axes[1].set_ylabel('k parameter')
    axes[1].set_title('Cluster Distribution in f-k Space')
    axes[1].invert_yaxis()  # F軸を反転
    axes[1].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[1], label='Cluster')
    
    plt.tight_layout()
    plt.savefig('cluster_overview.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 代表例の選択
    print("\n🎯 代表例の選択中...")
    representatives = select_cluster_representatives(results, method='centroid', n_samples=3)
    print(f"各クラスターから3つずつ、計{len(representatives) * 3}個の代表例を選択")
    
    # 各クラスターの特徴分析
    print("\n🔍 クラスター特徴分析中...")
    analyze_cluster_characteristics(df)
    
    # パラメータ範囲の比較
    print("\n📊 クラスター別パラメータ範囲:")
    for cluster_id in sorted(df['cluster'].unique()):
        cluster_data = df[df['cluster'] == cluster_id]
        f_range = (cluster_data['f_value'].min(), cluster_data['f_value'].max())
        k_range = (cluster_data['k_value'].min(), cluster_data['k_value'].max())
        
        print(f"Cluster {cluster_id}: f=[{f_range[0]:.4f}, {f_range[1]:.4f}], "
              f"k=[{k_range[0]:.4f}, {k_range[1]:.4f}]")
    
    # 代表例の詳細比較テーブル
    print("\n📋 代表例サマリーテーブル作成中...")
    summary_df = create_representatives_summary(representatives)
    
    # CSVファイルに保存
    summary_df.to_csv('cluster_representatives_summary.csv', index=False)
    print("✅ 代表例サマリーテーブルを 'cluster_representatives_summary.csv' に保存しました")
    
    print("\n代表例サマリー:")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    print(summary_df)
    
    print("\n🎉 クラスター分析が完了しました！")
    print("\n生成されたファイル:")
    print("  - cluster_overview.png: クラスター概要")
    print("  - cluster_characteristics_analysis.png: 詳細特徴分析")
    print("  - cluster_representatives_summary.csv: 代表例サマリー")

if __name__ == "__main__":
    main() 