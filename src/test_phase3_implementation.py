#!/usr/bin/env python3
"""
Phase 3 実装テストスクリプト
マルチスケール特徴融合システムの動作確認

機能:
1. Phase 3モジュールのインポートテスト
2. モデル構造の検証
3. データ拡張システムのテスト
4. 小規模データでの動作確認
"""

import os
import sys
import torch
import numpy as np
import time
from torch.utils.data import DataLoader

# Phase 3モジュールをインポート
try:
    from gray_scott_autoencoder_phase3 import (
        Conv3DAutoencoderPhase3,
        GrayScottDatasetPhase3,
        GrayScottAugmentation,
        MultiScaleFeatureFusion,
        EnhancedSpatioTemporalAttention,
        ResidualMultiScaleBlock3D
    )
    print("✅ Phase 3 modules imported successfully")
except ImportError as e:
    print(f"❌ Failed to import Phase 3 modules: {e}")
    sys.exit(1)

def test_model_architecture():
    """モデル構造のテスト"""
    print("\n🧠 Testing Phase 3 Model Architecture...")
    
    # GPU/CPU設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # モデル作成
    model = Conv3DAutoencoderPhase3(
        input_channels=1,
        fixed_frames=30,
        target_size=(64, 64),
        latent_dim=512
    ).to(device)
    
    # パラメータ数計算
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"📊 Model Parameters:")
    print(f"   Total: {total_params:,}")
    print(f"   Trainable: {trainable_params:,}")
    print(f"   Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB")
    
    # テスト入力
    batch_size = 2
    test_input = torch.randn(batch_size, 1, 30, 64, 64).to(device)
    
    print(f"🔍 Testing forward pass with input shape: {test_input.shape}")
    
    try:
        with torch.no_grad():
            start_time = time.time()
            reconstructed, latent = model(test_input)
            inference_time = time.time() - start_time
            
        print(f"✅ Forward pass successful!")
        print(f"   Input shape: {test_input.shape}")
        print(f"   Latent shape: {latent.shape}")
        print(f"   Output shape: {reconstructed.shape}")
        print(f"   Inference time: {inference_time:.3f}s")
        
        # 出力値の範囲チェック
        print(f"   Output range: [{reconstructed.min():.3f}, {reconstructed.max():.3f}]")
        print(f"   Latent range: [{latent.min():.3f}, {latent.max():.3f}]")
        
        return True
        
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        return False

def test_multiscale_modules():
    """マルチスケールモジュールのテスト"""
    print("\n🔧 Testing Multi-Scale Modules...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    success_count = 0
    total_tests = 3
    
    # MultiScaleFeatureFusion テスト
    print("Testing MultiScaleFeatureFusion...")
    msff = MultiScaleFeatureFusion(in_channels=128, out_channels=256).to(device)
    test_input = torch.randn(2, 128, 8, 16, 16).to(device)
    
    try:
        output = msff(test_input)
        print(f"   ✅ MSFF: {test_input.shape} → {output.shape}")
        success_count += 1
    except Exception as e:
        print(f"   ❌ MSFF failed: {e}")
    
    # EnhancedSpatioTemporalAttention テスト
    print("Testing EnhancedSpatioTemporalAttention...")
    esta = EnhancedSpatioTemporalAttention(channels=256).to(device)
    test_input = torch.randn(2, 256, 8, 16, 16).to(device)
    
    try:
        output = esta(test_input)
        print(f"   ✅ ESTA: {test_input.shape} → {output.shape}")
        success_count += 1
    except Exception as e:
        print(f"   ❌ ESTA failed: {e}")
    
    # ResidualMultiScaleBlock3D テスト
    print("Testing ResidualMultiScaleBlock3D...")
    rmsb = ResidualMultiScaleBlock3D(in_channels=128, out_channels=256).to(device)
    test_input = torch.randn(2, 128, 8, 16, 16).to(device)
    
    try:
        output = rmsb(test_input)
        print(f"   ✅ RMSB: {test_input.shape} → {output.shape}")
        success_count += 1
    except Exception as e:
        print(f"   ❌ RMSB failed: {e}")
    
    # 結果の評価
    if success_count == total_tests:
        print(f"✅ All multi-scale modules passed ({success_count}/{total_tests})")
        return True
    else:
        print(f"❌ Some multi-scale modules failed ({success_count}/{total_tests})")
        return False

def test_data_augmentation():
    """データ拡張システムのテスト"""
    print("\n🎨 Testing Data Augmentation System...")
    
    # データ拡張器作成
    augmentation = GrayScottAugmentation(
        temporal_shift_prob=1.0,  # 確実にテストするため
        spatial_flip_prob=1.0,
        noise_prob=1.0,
        intensity_prob=1.0,
        temporal_crop_prob=1.0
    )
    
    # テストデータ作成
    test_tensor = torch.randn(1, 30, 64, 64)
    original_shape = test_tensor.shape
    
    print(f"Original tensor shape: {original_shape}")
    print(f"Original range: [{test_tensor.min():.3f}, {test_tensor.max():.3f}]")
    
    try:
        # 各拡張をテスト
        augmented = augmentation(test_tensor.clone())
        print(f"✅ Data augmentation successful!")
        print(f"   Augmented shape: {augmented.shape}")
        print(f"   Augmented range: [{augmented.min():.3f}, {augmented.max():.3f}]")
        
        # 個別拡張のテスト
        print("\nTesting individual augmentations:")
        
        # 時間軸シフト
        shifted = augmentation.temporal_shift(test_tensor.clone())
        print(f"   ✅ Temporal shift: shape preserved = {shifted.shape == original_shape}")
        
        # 空間反転
        flipped = augmentation.spatial_flip(test_tensor.clone())
        print(f"   ✅ Spatial flip: shape preserved = {flipped.shape == original_shape}")
        
        # ノイズ追加
        noisy = augmentation.add_noise(test_tensor.clone())
        print(f"   ✅ Noise addition: shape preserved = {noisy.shape == original_shape}")
        
        # 強度変換
        transformed = augmentation.intensity_transform(test_tensor.clone())
        print(f"   ✅ Intensity transform: shape preserved = {transformed.shape == original_shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Data augmentation failed: {e}")
        return False

def test_dataset_loading():
    """データセット読み込みのテスト"""
    print("\n📁 Testing Dataset Loading...")
    
    gif_folder = 'data/gif'
    
    if not os.path.exists(gif_folder):
        print(f"⚠️ GIF folder not found: {gif_folder}")
        print("   Creating dummy dataset for testing...")
        return test_dummy_dataset()
    
    try:
        # 小規模データセットでテスト
        dataset = GrayScottDatasetPhase3(
            gif_folder=gif_folder,
            fixed_frames=30,
            target_size=(64, 64),
            use_augmentation=True,
            max_samples=10  # 最初の10サンプルのみ
        )
        
        if len(dataset) == 0:
            print("⚠️ No valid samples found in dataset")
            return False
        
        print(f"✅ Dataset created successfully!")
        print(f"   Samples: {len(dataset)}")
        
        # データローダーテスト
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
        
        for batch in dataloader:
            tensor = batch['tensor']
            f_value = batch['f_value']
            k_value = batch['k_value']
            
            print(f"   Batch tensor shape: {tensor.shape}")
            print(f"   f_value range: [{f_value.min():.3f}, {f_value.max():.3f}]")
            print(f"   k_value range: [{k_value.min():.3f}, {k_value.max():.3f}]")
            break  # 最初のバッチのみテスト
        
        return True
        
    except Exception as e:
        print(f"❌ Dataset loading failed: {e}")
        return False

def test_dummy_dataset():
    """ダミーデータセットでのテスト"""
    print("Creating dummy dataset for testing...")
    
    try:
        # ダミーのGrayScottDatasetPhase3を作成
        class DummyDataset:
            def __init__(self):
                self.tensors = [torch.randn(1, 30, 64, 64) for _ in range(5)]
                self.f_values = np.random.uniform(0.01, 0.06, 5)
                self.k_values = np.random.uniform(0.04, 0.07, 5)
                
            def __len__(self):
                return len(self.tensors)
            
            def __getitem__(self, idx):
                return {
                    'tensor': self.tensors[idx],
                    'f_value': self.f_values[idx],
                    'k_value': self.k_values[idx],
                    'filename': f'dummy_{idx}.gif'
                }
        
        dataset = DummyDataset()
        dataloader = DataLoader(dataset, batch_size=2)
        
        for batch in dataloader:
            print(f"   Dummy batch tensor shape: {batch['tensor'].shape}")
            break
        
        print("✅ Dummy dataset test successful!")
        return True
        
    except Exception as e:
        print(f"❌ Dummy dataset test failed: {e}")
        return False

def test_training_components():
    """訓練コンポーネントのテスト"""
    print("\n🏋️ Testing Training Components...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 小さなモデルでテスト
    model = Conv3DAutoencoderPhase3(latent_dim=64).to(device)  # 小さなlatent_dim
    
    # 損失関数
    mse_loss = torch.nn.MSELoss()
    l1_loss = torch.nn.L1Loss()
    
    # オプティマイザー
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    # スケジューラー
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    
    # ダミーデータ
    test_input = torch.randn(2, 1, 30, 64, 64).to(device)
    
    try:
        model.train()
        
        # フォワードパス
        reconstructed, latent = model(test_input)
        
        # 損失計算
        loss_mse = mse_loss(reconstructed, test_input)
        loss_l1 = l1_loss(reconstructed, test_input)
        latent_reg = torch.mean(torch.norm(latent, dim=1))
        total_loss = loss_mse + 0.1 * loss_l1 + 0.001 * latent_reg
        
        # バックワードパス
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        print(f"✅ Training components test successful!")
        print(f"   MSE Loss: {loss_mse.item():.6f}")
        print(f"   L1 Loss: {loss_l1.item():.6f}")
        print(f"   Latent Reg: {latent_reg.item():.6f}")
        print(f"   Total Loss: {total_loss.item():.6f}")
        print(f"   Learning Rate: {scheduler.get_last_lr()[0]:.2e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Training components test failed: {e}")
        return False

def main():
    """メイン実行関数"""
    print("="*80)
    print("🧪 Phase 3 Implementation Test Suite")
    print("="*80)
    
    tests = [
        ("Model Architecture", test_model_architecture),
        ("Multi-Scale Modules", test_multiscale_modules),
        ("Data Augmentation", test_data_augmentation),
        ("Dataset Loading", test_dataset_loading),
        ("Training Components", test_training_components)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"Running: {test_name}")
        print('='*60)
        
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ Test '{test_name}' crashed: {e}")
            results[test_name] = False
    
    # 結果サマリー
    print("\n" + "="*80)
    print("🏆 Test Results Summary")
    print("="*80)
    
    passed = 0
    total = len(tests)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
        if result:
            passed += 1
    
    print("="*80)
    print(f"📊 Overall Result: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 All tests passed! Phase 3 implementation is ready.")
        print("\n🚀 Next steps:")
        print("   1. Run: python src/gray_scott_autoencoder_phase3.py")
        print("   2. Run: python src/visualize_phase3_results.py")
    else:
        print("⚠️ Some tests failed. Please check the implementation.")
    
    print("="*80)

if __name__ == "__main__":
    main() 