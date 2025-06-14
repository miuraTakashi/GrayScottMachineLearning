#!/usr/bin/env python3
"""
Gray-Scott 3D CNN Autoencoder Training Script
オートエンコーダーの学習のみを行うスクリプト
フレーム範囲指定機能付き
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
import argparse

# メインモジュールからインポート
from gray_scott_autoencoder import GrayScottDataset, Conv3DAutoencoder

def parse_frame_range(frame_range_str):
    """
    フレーム範囲文字列をパース
    例: "10-50", "1-128", "20-80"
    """
    if frame_range_str is None:
        return None
    
    try:
        parts = frame_range_str.split('-')
        if len(parts) != 2:
            raise ValueError("Frame range must be in format 'start-end'")
        
        start = int(parts[0])
        end = int(parts[1])
        
        if start >= end or start < 0:
            raise ValueError("Invalid frame range: start must be < end and >= 0")
        
        return (start, end)
    except ValueError as e:
        raise ValueError(f"Invalid frame range '{frame_range_str}': {e}")

def main():
    # コマンドライン引数の解析
    parser = argparse.ArgumentParser(description='Gray-Scott 3D CNN Autoencoder Training with Frame Range Support')
    parser.add_argument('--gif-folder', type=str, default='../data/gif', 
                       help='Path to GIF folder (default: ../data/gif)')
    parser.add_argument('--batch-size', type=int, default=4, 
                       help='Batch size (default: 4)')
    parser.add_argument('--fixed-frames', type=int, default=30, 
                       help='Fixed number of frames (default: 30)')
    parser.add_argument('--frame-range', type=str, default=None, 
                       help='Frame range to use (format: start-end, e.g., "10-50"). If not specified, uses all frames.')
    parser.add_argument('--target-size', type=str, default='64,64', 
                       help='Target image size (format: width,height, default: 64,64)')
    parser.add_argument('--latent-dim', type=int, default=64, 
                       help='Latent dimension (default: 64)')
    parser.add_argument('--epochs', type=int, default=50, 
                       help='Number of epochs (default: 50)')
    parser.add_argument('--lr', type=float, default=0.001, 
                       help='Learning rate (default: 0.001)')
    parser.add_argument('--output-suffix', type=str, default='', 
                       help='Suffix for output files (default: empty)')
    
    args = parser.parse_args()
    
    print("🔬 Gray-Scott 3D CNN Autoencoder Training")
    print("オートエンコーダーの学習専用スクリプト（フレーム範囲指定対応）")
    print("=" * 60)
    
    # パラメータの設定
    gif_folder = args.gif_folder
    batch_size = args.batch_size
    fixed_frames = args.fixed_frames
    frame_range = parse_frame_range(args.frame_range)
    target_size = tuple(map(int, args.target_size.split(',')))
    latent_dim = args.latent_dim
    num_epochs = args.epochs
    learning_rate = args.lr
    output_suffix = args.output_suffix
    
    print("📋 ハイパーパラメータ:")
    print(f"  - GIFフォルダ: {gif_folder}")
    print(f"  - バッチサイズ: {batch_size}")
    print(f"  - 固定フレーム数: {fixed_frames}")
    if frame_range is not None:
        print(f"  - フレーム範囲: {frame_range[0]} - {frame_range[1]} (範囲: {frame_range[1] - frame_range[0]}フレーム)")
    else:
        print(f"  - フレーム範囲: 全フレーム使用")
    print(f"  - 画像サイズ: {target_size}")
    print(f"  - 潜在次元数: {latent_dim}")
    print(f"  - エポック数: {num_epochs}")
    print(f"  - 学習率: {learning_rate}")
    if output_suffix:
        print(f"  - 出力サフィックス: {output_suffix}")
    
    # デバイス設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  使用デバイス: {device}")
    
    # データセット準備
    print("\n📂 データセット準備中...")
    dataset = GrayScottDataset(gif_folder, fixed_frames=fixed_frames, target_size=target_size, frame_range=frame_range)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    print(f"📊 データセットサイズ: {len(dataset)}")
    
    # データ形状を取得（最初のサンプルから）
    first_sample = dataset[0]
    sample_tensor = first_sample['tensor']
    print(f"📐 データ形状: {sample_tensor.shape}")
    
    # モデル初期化
    print("\n🏗️ モデル初期化中...")
    model = Conv3DAutoencoder(latent_dim=latent_dim, fixed_frames=fixed_frames, target_size=target_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📈 総パラメータ数: {total_params:,}")
    print(f"📈 学習パラメータ数: {trainable_params:,}")
    
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
                print(f"  Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(dataloader)}, Loss: {loss.item():.6f}")
        
        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)
        print(f"✅ Epoch {epoch+1}/{num_epochs} 完了, 平均Loss: {avg_loss:.6f}")
        
        # 10エポックごとに中間保存
        if (epoch + 1) % 10 == 0:
            print(f"💾 中間保存 (Epoch {epoch+1})...")
    
    # ファイル名の設定
    if frame_range is not None:
        frame_suffix = f"_frames{frame_range[0]}-{frame_range[1]}"
    else:
        frame_suffix = "_frames_all"
    
    if output_suffix:
        frame_suffix += f"_{output_suffix}"
    
    # 学習曲線を保存
    print("\n📊 学習曲線を保存中...")
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(losses, 'b-', linewidth=2)
    if frame_range is not None:
        plt.title(f'Training Loss Over Time (Frames {frame_range[0]}-{frame_range[1]})', fontsize=14)
    else:
        plt.title('Training Loss Over Time (All Frames)', fontsize=14)
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.grid(True, alpha=0.3)
    
    # ログスケールでも表示
    plt.subplot(2, 1, 2)
    plt.plot(losses, 'r-', linewidth=2)
    plt.title('Training Loss (Log Scale)', fontsize=14)
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss (log scale)')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'../results/training_loss{frame_suffix}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 特徴抽出（潜在ベクトル）
    print("\n🔍 特徴抽出中...")
    model.eval()
    all_latent_vectors = []
    all_filenames = []
    all_f_values = []
    all_k_values = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            data = batch['tensor'].to(device)
            filenames = batch['filename']
            f_vals = batch['f_value']
            k_vals = batch['k_value']
            
            _, latent = model(data)
            all_latent_vectors.append(latent.cpu().numpy())
            all_filenames.extend(filenames)
            all_f_values.extend(f_vals)
            all_k_values.extend(k_vals)
            
            if (batch_idx + 1) % 20 == 0:
                print(f"  特徴抽出進捗: {batch_idx+1}/{len(dataloader)} バッチ完了")
    
    # 配列に変換
    latent_vectors = np.vstack(all_latent_vectors)
    f_values = np.array(all_f_values, dtype=np.float32)
    k_values = np.array(all_k_values, dtype=np.float32)
    
    print(f"📐 潜在ベクトル形状: {latent_vectors.shape}")
    print(f"📊 f値範囲: {f_values.min():.4f} - {f_values.max():.4f}")
    print(f"📊 k値範囲: {k_values.min():.4f} - {k_values.max():.4f}")
    
    # 潜在空間の分析
    print("\n📈 潜在空間の統計分析...")
    latent_mean = np.mean(latent_vectors, axis=0)
    latent_std = np.std(latent_vectors, axis=0)
    
    print(f"📊 潜在ベクトル統計:")
    print(f"  - 平均値の範囲: {latent_mean.min():.4f} - {latent_mean.max():.4f}")
    print(f"  - 標準偏差の範囲: {latent_std.min():.4f} - {latent_std.max():.4f}")
    print(f"  - 全体平均: {np.mean(latent_vectors):.4f}")
    print(f"  - 全体標準偏差: {np.std(latent_vectors):.4f}")
    
    # 結果をまとめる
    autoencoder_results = {
        'latent_vectors': latent_vectors,
        'filenames': all_filenames,
        'f_values': f_values,
        'k_values': k_values,
        'model_config': {
            'latent_dim': latent_dim,
            'fixed_frames': fixed_frames,
            'target_size': target_size,
            'frame_range': frame_range,
            'num_epochs': num_epochs,
            'learning_rate': learning_rate,
            'batch_size': batch_size
        },
        'training_stats': {
            'final_loss': losses[-1],
            'min_loss': min(losses),
            'loss_history': losses,
            'total_samples': len(dataset)
        }
    }
    
    # モデルと結果を保存
    print("\n💾 保存中...")
    
    # モデル保存（より詳細な情報を含む）
    model_data = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'model_config': {
            'latent_dim': latent_dim,
            'fixed_frames': fixed_frames,
            'target_size': target_size,
            'frame_range': frame_range
        },
        'training_config': {
            'num_epochs': num_epochs,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'final_loss': losses[-1],
            'loss_history': losses
        },
        'dataset_info': {
            'total_samples': len(dataset),
            'gif_folder': gif_folder
        }
    }
    torch.save(model_data, f'../models/trained_autoencoder{frame_suffix}.pth')
    
    # 潜在表現データ保存
    with open(f'../results/latent_representations{frame_suffix}.pkl', 'wb') as f:
        pickle.dump(autoencoder_results, f)
    
    # CSV出力（クラスタリング前）
    df = pd.DataFrame({
        'filename': all_filenames,
        'f_value': f_values,
        'k_value': k_values
    })
    df.to_csv(f'../results/extracted_features{frame_suffix}.csv', index=False)
    
    # 保存ファイルのサイズ表示
    model_size = os.path.getsize(f'../models/trained_autoencoder{frame_suffix}.pth') / 1024 / 1024
    data_size = os.path.getsize(f'../results/latent_representations{frame_suffix}.pkl') / 1024
    csv_size = os.path.getsize(f'../results/extracted_features{frame_suffix}.csv') / 1024
    
    print("✅ 保存完了!")
    print(f"  📦 trained_autoencoder{frame_suffix}.pth: 学習済みモデル ({model_size:.1f}MB)")
    print(f"  📦 latent_representations{frame_suffix}.pkl: 潜在表現データ ({data_size:.1f}KB)")
    print(f"  📦 extracted_features{frame_suffix}.csv: 特徴量テーブル ({csv_size:.1f}KB)")
    print(f"  📦 training_loss{frame_suffix}.png: 学習曲線")
    
    print(f"\n🎉 オートエンコーダー学習完了!")
    print("=" * 60)
    print("📋 学習結果サマリー:")
    print(f"  🎯 最終損失: {losses[-1]:.6f}")
    print(f"  📉 最小損失: {min(losses):.6f}")
    print(f"  📊 総サンプル数: {len(dataset)}")
    print(f"  🔢 潜在次元数: {latent_dim}")
    print(f"  ⏱️  エポック数: {num_epochs}")
    if frame_range is not None:
        print(f"  🎬 使用フレーム範囲: {frame_range[0]} - {frame_range[1]} ({frame_range[1] - frame_range[0]}フレーム)")
    else:
        print(f"  🎬 使用フレーム: 全フレーム")
    
    print(f"\n🔗 次のステップ:")
    if frame_range is not None:
        print(f"  1. クラスター分析を実行: python cluster_analysis.py --latent-file latent_representations{frame_suffix}.pkl")
        print(f"  2. または包括的分析: python optimal_clustering.py --latent-file latent_representations{frame_suffix}.pkl")
    else:
        print("  1. クラスター分析を実行: python cluster_analysis.py")
        print("  2. または包括的分析: python optimal_clustering.py")

if __name__ == "__main__":
    main() 