import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from gray_scott_autoencoder import GrayScottDataset, Conv3DAutoencoder

def test_data_loading():
    """データ読み込み機能のテスト"""
    print("=== Testing Data Loading ===")
    
    # テストデータセット作成
    dataset = GrayScottDataset('gif', fixed_frames=10, target_size=(32, 32))
    
    print(f"Number of samples loaded: {len(dataset)}")
    print(f"F values range: {min(dataset.f_values):.4f} - {max(dataset.f_values):.4f}")
    print(f"K values range: {min(dataset.k_values):.4f} - {max(dataset.k_values):.4f}")
    
    # サンプルデータの確認
    sample = dataset[0]
    tensor = sample['tensor']
    print(f"Tensor shape: {tensor.shape}")
    print(f"F value: {sample['f_value']}")
    print(f"K value: {sample['k_value']}")
    print(f"Filename: {sample['filename']}")
    
    # フレームサンプルの可視化
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    for i in range(5):
        frame = tensor[0, i*2, :, :].numpy()
        axes[i].imshow(frame, cmap='gray')
        axes[i].set_title(f'Frame {i*2}')
        axes[i].axis('off')
    
    plt.suptitle(f'Sample GIF: {sample["filename"]}')
    plt.tight_layout()
    plt.savefig('test_gif_frames.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Data loading test completed.")
    return True

def test_autoencoder_architecture():
    """Autoencoderのアーキテクチャテスト"""
    print("\n=== Testing Autoencoder Architecture ===")
    
    # モデル作成（テスト用サイズ）
    test_size = (32, 32)
    model = Conv3DAutoencoder(input_channels=1, fixed_frames=10, target_size=test_size, latent_dim=32)
    
    # テスト入力
    batch_size = 2
    test_input = torch.randn(batch_size, 1, 10, test_size[0], test_size[1])
    
    print(f"Input shape: {test_input.shape}")
    
    # エンコード
    latent = model.encode(test_input)
    print(f"Latent shape: {latent.shape}")
    
    # デコード
    decoded = model.decode(latent)
    print(f"Decoded shape: {decoded.shape}")
    
    # フォワードパス
    reconstructed, latent_vec = model(test_input)
    print(f"Reconstructed shape: {reconstructed.shape}")
    
    # 形状チェック
    assert test_input.shape == reconstructed.shape, f"Input and output shapes must match: {test_input.shape} vs {reconstructed.shape}"
    assert latent.shape[1] == 32, "Latent dimension must be 32"
    
    print("Autoencoder architecture test completed.")
    return True

def test_parameter_extraction():
    """パラメータ抽出機能のテスト"""
    print("\n=== Testing Parameter Extraction ===")
    
    # テストファイル名
    test_filenames = [
        "GrayScott-f0.0580-k0.0680-00.gif",
        "GrayScott-f0.0123-k0.0456-00.gif",
        "GrayScott-f0.1000-k0.0999-00.gif"
    ]
    
    dataset = GrayScottDataset('gif', fixed_frames=5, target_size=(16, 16))
    
    for filename in test_filenames:
        f_val, k_val = dataset._parse_filename(filename)
        if f_val is not None and k_val is not None:
            print(f"{filename} -> f={f_val}, k={k_val}")
        else:
            print(f"Failed to parse: {filename}")
    
    print("Parameter extraction test completed.")
    return True

def test_tensor_consistency():
    """テンソル一貫性のテスト"""
    print("\n=== Testing Tensor Consistency ===")
    
    dataset = GrayScottDataset('gif', fixed_frames=15, target_size=(48, 48))
    
    print("Checking tensor shapes...")
    shapes = []
    for i in range(min(10, len(dataset))):
        tensor = dataset[i]['tensor']
        shapes.append(tensor.shape)
    
    # すべてのテンソルが同じ形状かチェック
    first_shape = shapes[0]
    all_same = all(shape == first_shape for shape in shapes)
    
    print(f"All tensors have same shape: {all_same}")
    print(f"Expected shape: {first_shape}")
    
    if not all_same:
        print("Different shapes found:")
        for i, shape in enumerate(set(shapes)):
            print(f"  Shape {i}: {shape}")
    
    print("Tensor consistency test completed.")
    return True

def test_clustering_data_format():
    """クラスタリングデータ形式のテスト"""
    print("\n=== Testing Clustering Data Format ===")
    
    # CSVファイルが存在するかチェック
    if os.path.exists('clustering_results.csv'):
        import pandas as pd
        df = pd.read_csv('clustering_results.csv')
        
        print(f"CSV file loaded successfully: {len(df)} rows")
        print(f"Columns: {list(df.columns)}")
        print(f"Unique clusters: {sorted(df['cluster'].unique())}")
        
        # F-K値の分布確認
        print(f"F values: min={df['f_value'].min():.4f}, max={df['f_value'].max():.4f}")
        print(f"K values: min={df['k_value'].min():.4f}, max={df['k_value'].max():.4f}")
        
        # 各クラスタのサンプル数
        cluster_counts = df['cluster'].value_counts().sort_index()
        print("Samples per cluster:")
        for cluster, count in cluster_counts.items():
            print(f"  Cluster {cluster}: {count} samples")
        
    else:
        print("clustering_results.csv not found. Run main analysis first.")
        return False
    
    print("Clustering data format test completed.")
    return True

def test_output_files():
    """出力ファイルの存在確認"""
    print("\n=== Testing Output Files ===")
    
    expected_files = [
        'clustering_results.csv',
        'training_loss.png',
        'gray_scott_clustering_results.png', 
        'gray_scott_detailed_heatmap.png'
    ]
    
    for filename in expected_files:
        exists = os.path.exists(filename)
        size = os.path.getsize(filename) if exists else 0
        print(f"{filename}: {'✓' if exists else '✗'} ({size} bytes)")
    
    all_exist = all(os.path.exists(f) for f in expected_files)
    print(f"All output files present: {all_exist}")
    
    return all_exist

def run_all_tests():
    """全テストの実行"""
    print("Starting comprehensive functionality tests...\n")
    
    tests = [
        test_data_loading,
        test_autoencoder_architecture, 
        test_parameter_extraction,
        test_tensor_consistency,
        test_clustering_data_format,
        test_output_files
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"Test failed with error: {e}")
            results.append(False)
    
    print(f"\n=== Test Summary ===")
    print(f"Passed: {sum(results)}/{len(results)}")
    
    if all(results):
        print("🎉 All tests passed! The implementation meets all specifications.")
    else:
        print("❌ Some tests failed. Please check the implementation.")
    
    return all(results)

if __name__ == "__main__":
    run_all_tests() 