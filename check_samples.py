#!/usr/bin/env python3
"""
Check sample counts in result files
"""

import pickle
import os

def check_sample_counts():
    """結果ファイルのサンプル数をチェック"""
    
    print("🔍 結果ファイルのサンプル数チェック")
    print("=" * 50)
    
    files = [
        'results/latent_representations_frames_all.pkl',
        'results/latent_representations.pkl', 
        'results/analysis_results.pkl'
    ]
    
    for file in files:
        if os.path.exists(file):
            try:
                with open(file, 'rb') as f:
                    data = pickle.load(f)
                
                size_mb = os.path.getsize(file) / (1024*1024)
                print(f'📂 {os.path.basename(file)} ({size_mb:.1f}MB):')
                
                if 'latent_vectors' in data:
                    print(f'  📊 Samples: {len(data["latent_vectors"])}')
                    if hasattr(data["latent_vectors"], 'shape'):
                        print(f'  📐 Latent dim: {data["latent_vectors"].shape[1] if len(data["latent_vectors"]) > 0 else 0}')
                elif 'f_values' in data:
                    print(f'  📊 Samples: {len(data["f_values"])}')
                else:
                    print(f'  📊 Keys: {list(data.keys())}')
                    
                if 'f_values' in data:
                    print(f'  📈 f range: {data["f_values"].min():.4f} - {data["f_values"].max():.4f}')
                if 'k_values' in data:
                    print(f'  📈 k range: {data["k_values"].min():.4f} - {data["k_values"].max():.4f}')
                    
                print()
                
            except Exception as e:
                print(f'❌ {file}: {e}')
        else:
            print(f'❌ {file}: ファイルが存在しません')
    
    # GIFファイル数も確認
    gif_count = len([f for f in os.listdir('data/gif') if f.endswith('.gif')])
    print(f"🎬 GIFファイル数: {gif_count}")

if __name__ == "__main__":
    check_sample_counts() 