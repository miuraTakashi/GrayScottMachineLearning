#!/usr/bin/env python3
"""
Gray-Scott 3D CNN Autoencoder Training Script
最適クラスター数の自動探索機能付き
"""

import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# メインモジュールからインポート
from gray_scott_autoencoder import GrayScottDataset, Conv3DAutoencoder

def find_optimal_clusters_simple(latent_vectors, max_k=20):
    """
    シンプルな最適クラスター数探索（シルエット分析ベース）
    """
    print("🔍 最適クラスター数を探索中...")
    
    silhouette_scores = []
    k_range = range(2, min(max_k + 1, len(latent_vectors) // 2))
    
    best_score = -1
    best_k = 6  # デフォルト値
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(latent_vectors)
        score = silhouette_score(latent_vectors, cluster_labels)
        silhouette_scores.append(score)
        
        if score > best_score:
            best_score = score
            best_k = k
        
        print(f"  k={k}: シルエットスコア = {score:.3f}")
    
    print(f"🎯 推奨クラスター数: k={best_k} (スコア: {best_score:.3f})")
    
    return best_k, silhouette_scores, list(k_range)

def get_user_cluster_choice(recommended_k, silhouette_scores, k_range):
    """
    ユーザーにクラスター数の選択を求める
    """
    print("\n" + "="*60)
    print("🎯 クラスター数の選択")
    print("="*60)
    
    # 上位3つのクラスター数を表示
    score_k_pairs = list(zip(silhouette_scores, k_range))
    score_k_pairs.sort(reverse=True)  # スコア順でソート
    
    print("📊 シルエットスコア上位:")
    for i, (score, k) in enumerate(score_k_pairs[:5]):
        marker = "🏆" if k == recommended_k else f"{i+1}."
        print(f"  {marker} k={k}: {score:.3f}")
    
    print(f"\n💡 推奨クラスター数: {recommended_k}")
    print(f"📈 利用可能な範囲: {min(k_range)}～{max(k_range)}")
    
    while True:
        try:
            print("\n選択してください:")
            print(f"1. 推奨値を使用 (k={recommended_k})")
            print("2. 手動でクラスター数を指定")
            print("3. シルエット分析グラフを表示")
            
            choice = input("選択 (1/2/3): ").strip()
            
            if choice == "1":
                final_k = recommended_k
                print(f"✅ 推奨値 k={final_k} を使用します")
                break
            
            elif choice == "2":
                manual_k = int(input(f"クラスター数を入力 ({min(k_range)}～{max(k_range)}): "))
                if manual_k in k_range:
                    final_k = manual_k
                    # 選択したkのシルエットスコアを表示
                    selected_score = silhouette_scores[k_range.index(manual_k)]
                    print(f"✅ k={final_k} を選択しました (シルエットスコア: {selected_score:.3f})")
                    break
                else:
                    print(f"❌ {manual_k} は範囲外です。{min(k_range)}～{max(k_range)}の値を入力してください。")
            
            elif choice == "3":
                # シルエット分析グラフを表示
                plt.figure(figsize=(10, 6))
                plt.plot(k_range, silhouette_scores, 'bo-', linewidth=2, markersize=8)
                plt.axvline(x=recommended_k, color='red', linestyle='--', 
                           label=f'Recommended k = {recommended_k}')
                plt.xlabel('Number of Clusters (k)')
                plt.ylabel('Silhouette Score')
                plt.title('Silhouette Analysis for Optimal k')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                # 各点にスコアを表示
                for i, (k, score) in enumerate(zip(k_range, silhouette_scores)):
                    plt.annotate(f'{score:.3f}', (k, score), 
                               textcoords="offset points", xytext=(0,10), ha='center')
                
                plt.tight_layout()
                plt.show()
                print("📊 グラフを表示しました。")
            
            else:
                print("❌ 1, 2, または 3 を入力してください。")
        
        except ValueError:
            print("❌ 有効な数値を入力してください。")
        except KeyboardInterrupt:
            print(f"\n⚠️  中断されました。推奨値 k={recommended_k} を使用します。")
            final_k = recommended_k
            break
    
    return final_k

def main():
    print("🔬 Gray-Scott 3D CNN Autoencoder Training")
    print("最適クラスター数の自動探索機能付き")
    print("=" * 60)
    
    # ハイパーパラメータ設定
    gif_folder = 'gif'
    batch_size = 4
    fixed_frames = 30
    target_size = (64, 64)
    latent_dim = 64
    num_epochs = 50
    learning_rate = 0.001
    
    # デバイス設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用デバイス: {device}")
    
    # データセット準備
    print("\n📂 データセット準備中...")
    dataset = GrayScottDataset(gif_folder, fixed_frames=fixed_frames, target_size=target_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    print(f"データセットサイズ: {len(dataset)}")
    
    # データ形状を取得（最初のサンプルから）
    first_sample = dataset[0]
    sample_tensor = first_sample['tensor']
    print(f"データ形状: {sample_tensor.shape}")
    
    # モデル初期化
    print("\n🏗️ モデル初期化中...")
    model = Conv3DAutoencoder(latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    print(f"モデルパラメータ数: {sum(p.numel() for p in model.parameters()):,}")
    
    # 学習
    print(f"\n🚀 学習開始 ({num_epochs} エポック)...")
    model.train()
    losses = []
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(dataloader):
            data = batch['tensor'].to(device)
            
            optimizer.zero_grad()
            reconstructed, _ = model(data)
            loss = criterion(reconstructed, data)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 20 == 0:
                print(f"  Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}, Loss: {loss.item():.4f}")
        
        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)
        print(f"✅ Epoch {epoch+1}/{num_epochs} 完了, 平均Loss: {avg_loss:.4f}")
    
    # 学習曲線を保存
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.grid(True)
    plt.savefig('training_loss.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 特徴抽出（潜在ベクトル）
    print("\n🔍 特徴抽出中...")
    model.eval()
    all_latent_vectors = []
    all_filenames = []
    all_f_values = []
    all_k_values = []
    
    with torch.no_grad():
        for batch in dataloader:
            data = batch['tensor'].to(device)
            filenames = batch['filename']
            f_vals = batch['f_value']
            k_vals = batch['k_value']
            
            _, latent = model(data)
            all_latent_vectors.append(latent.cpu().numpy())
            all_filenames.extend(filenames)
            all_f_values.extend(f_vals)
            all_k_values.extend(k_vals)
    
    # 配列に変換
    latent_vectors = np.vstack(all_latent_vectors)
    f_values = np.array(all_f_values, dtype=np.float32)
    k_values = np.array(all_k_values, dtype=np.float32)
    
    print(f"潜在ベクトル形状: {latent_vectors.shape}")
    
    # 最適クラスター数の探索
    print("\n🎯 最適クラスター数の探索...")
    optimal_k, sil_scores, k_range = find_optimal_clusters_simple(latent_vectors)
    
    # ユーザーにクラスター数の選択を求める
    final_k = get_user_cluster_choice(optimal_k, sil_scores, k_range)
    
    # シルエット分析結果の可視化（最終選択を反映）
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, sil_scores, 'bo-', linewidth=2, markersize=8)
    plt.axvline(x=optimal_k, color='red', linestyle='--', label=f'Recommended k = {optimal_k}')
    if final_k != optimal_k:
        plt.axvline(x=final_k, color='green', linestyle='-', linewidth=3, 
                   label=f'Selected k = {final_k}')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Analysis for Optimal k')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 各点にスコアを表示
    for i, (k, score) in enumerate(zip(k_range, sil_scores)):
        color = 'green' if k == final_k else 'red' if k == optimal_k else 'black'
        weight = 'bold' if k in [final_k, optimal_k] else 'normal'
        plt.annotate(f'{score:.3f}', (k, score), 
                   textcoords="offset points", xytext=(0,10), ha='center',
                   color=color, weight=weight)
    
    plt.savefig('silhouette_optimization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 選択されたクラスター数でクラスタリング
    print(f"\n📊 K-means クラスタリング (k={final_k})...")
    kmeans = KMeans(n_clusters=final_k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(latent_vectors)
    
    final_silhouette = silhouette_score(latent_vectors, cluster_labels)
    print(f"最終シルエットスコア: {final_silhouette:.3f}")
    
    # 推奨値と異なる場合は比較を表示
    if final_k != optimal_k:
        print(f"\n📊 比較:")
        recommended_score = sil_scores[k_range.index(optimal_k)]
        selected_score = sil_scores[k_range.index(final_k)]
        print(f"  推奨 k={optimal_k}: シルエットスコア = {recommended_score:.3f}")
        print(f"  選択 k={final_k}: シルエットスコア = {selected_score:.3f}")
        diff = selected_score - recommended_score
        trend = "⬆️" if diff > 0 else "⬇️" if diff < 0 else "➡️"
        print(f"  差分: {diff:+.3f} {trend}")
    
    # クラスター分布を表示
    unique, counts = np.unique(cluster_labels, return_counts=True)
    print("\nクラスター分布:")
    for cluster_id, count in zip(unique, counts):
        percentage = count / len(cluster_labels) * 100
        print(f"  Cluster {cluster_id}: {count} samples ({percentage:.1f}%)")
    
    # 次元削減（可視化用）
    print("\n📈 次元削減中...")
    
    # PCA
    pca = PCA(n_components=2, random_state=42)
    pca_result = pca.fit_transform(latent_vectors)
    
    # t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_result = tsne.fit_transform(latent_vectors)
    
    # 結果をまとめる
    results = {
        'latent_vectors': latent_vectors,
        'cluster_labels': cluster_labels,
        'filenames': all_filenames,
        'f_values': f_values,
        'k_values': k_values,
        'pca_result': pca_result,
        'tsne_result': tsne_result,
        'n_clusters': final_k,
        'silhouette_score': final_silhouette,
        'optimal_k_analysis': {
            'k_range': k_range,
            'silhouette_scores': sil_scores,
            'recommended_k': optimal_k,
            'selected_k': final_k
        }
    }
    
    # モデルと結果を保存
    print("\n💾 保存中...")
    
    # モデル保存
    model_data = {
        'model_state_dict': model.state_dict(),
        'latent_dim': latent_dim,
        'fixed_frames': fixed_frames,
        'target_size': target_size,
        'num_epochs': num_epochs,
        'final_loss': losses[-1],
        'n_clusters': final_k
    }
    torch.save(model_data, 'trained_model.pth')
    
    # 分析結果保存
    with open('analysis_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    # CSV出力
    df = pd.DataFrame({
        'filename': all_filenames,
        'f_value': f_values,
        'k_value': k_values,
        'cluster': cluster_labels
    })
    df.to_csv('clustering_results.csv', index=False)
    
    print("✅ 保存完了!")
    print(f"  - trained_model.pth: 学習済みモデル ({os.path.getsize('trained_model.pth')/1024/1024:.1f}MB)")
    print(f"  - analysis_results.pkl: 分析結果 ({os.path.getsize('analysis_results.pkl')/1024:.1f}KB)")
    print(f"  - clustering_results.csv: クラスタリング結果テーブル")
    print(f"  - training_loss.png: 学習曲線")
    print(f"  - silhouette_optimization.png: シルエット分析結果")
    
    print(f"\n🎉 学習完了!")
    print(f"📊 最適クラスター数: {final_k}")
    print(f"📈 最終シルエットスコア: {final_silhouette:.3f}")
    print(f"💾 総サンプル数: {len(dataset)}")

if __name__ == "__main__":
    main() 