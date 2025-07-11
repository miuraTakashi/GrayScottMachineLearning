#!/usr/bin/env python3
"""
Phase 4 デバッグスクリプト
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os

# Phase 4モデルをインポート
sys.path.append('src')
from gray_scott_autoencoder_phase4 import Conv3DAutoencoderPhase4

def test_model_creation():
    """モデル作成テスト"""
    print("🔧 Testing model creation...")
    try:
        model = Conv3DAutoencoderPhase4(latent_dim=512)
        print("✅ Model created successfully")
        print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        return model
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_encoder(model):
    """エンコーダーテスト"""
    print("\n🔧 Testing encoder...")
    try:
        # テストデータ作成
        batch_size = 2
        test_input = torch.randn(batch_size, 1, 20, 64, 64)
        print(f"Input shape: {test_input.shape}")
        
        # エンコード
        with torch.no_grad():
            latent = model.encode(test_input)
            print(f"Latent shape: {latent.shape}")
            print("✅ Encoder test passed")
            return latent
    except Exception as e:
        print(f"❌ Encoder test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_decoder(model, latent):
    """デコーダーテスト"""
    print("\n🔧 Testing decoder...")
    try:
        with torch.no_grad():
            decoded = model.decode(latent)
            print(f"Decoded shape: {decoded.shape}")
            print("✅ Decoder test passed")
            return decoded
    except Exception as e:
        print(f"❌ Decoder test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_full_forward(model):
    """完全なフォワードパステスト"""
    print("\n🔧 Testing full forward pass...")
    try:
        batch_size = 2
        test_input = torch.randn(batch_size, 1, 20, 64, 64)
        print(f"Input shape: {test_input.shape}")
        
        with torch.no_grad():
            reconstructed, latent, projection = model(test_input)
            print(f"Reconstructed shape: {reconstructed.shape}")
            print(f"Latent shape: {latent.shape}")
            print(f"Projection shape: {projection.shape}")
            print("✅ Full forward pass test passed")
            return True
    except Exception as e:
        print(f"❌ Full forward pass test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_decoder_step_by_step(model):
    """デコーダーの段階的テスト"""
    print("\n🔧 Testing decoder step by step...")
    try:
        batch_size = 2
        latent = torch.randn(batch_size, 512)
        
        with torch.no_grad():
            # FC decoder
            x = model.fc_decoder(latent)
            print(f"After fc_decoder: {x.shape}")
            
            # Reshape
            x = x.view(x.size(0), 256, 1, 1, 1)
            print(f"After reshape: {x.shape}")
            
            # Step 1: (256, 1, 1, 1) -> (128, 2, 8, 8)
            x = model.decoder_conv1(x)
            print(f"After decoder_conv1: {x.shape}")
            x = model.decoder_bn1(x)
            x = model.relu(x)
            x = F.interpolate(x, size=(2, 8, 8), mode='trilinear', align_corners=False)
            print(f"After step 1: {x.shape}")
            
            # Step 2: (128, 2, 8, 8) -> (64, 5, 16, 16)
            x = model.decoder_conv2(x)
            print(f"After decoder_conv2: {x.shape}")
            x = model.decoder_bn2(x)
            x = model.relu(x)
            x = F.interpolate(x, size=(5, 16, 16), mode='trilinear', align_corners=False)
            print(f"After step 2: {x.shape}")
            
            # Step 3: (64, 5, 16, 16) -> (32, 10, 32, 32)
            x = model.decoder_conv3(x)
            print(f"After decoder_conv3: {x.shape}")
            x = model.decoder_bn3(x)
            x = model.relu(x)
            x = F.interpolate(x, size=(10, 32, 32), mode='trilinear', align_corners=False)
            print(f"After step 3: {x.shape}")
            
            # Step 4: (32, 10, 32, 32) -> (1, 20, 64, 64)
            x = model.decoder_conv4(x)
            print(f"After decoder_conv4: {x.shape}")
            x = F.interpolate(x, size=(20, 64, 64), mode='trilinear', align_corners=False)
            x = model.sigmoid(x)
            print(f"Final output: {x.shape}")
            
            print("✅ Step-by-step decoder test passed")
            return True
    except Exception as e:
        print(f"❌ Step-by-step decoder test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """メインデバッグ関数"""
    print("🚀 Phase 4 Debug Test Started")
    print("=" * 50)
    
    # 1. モデル作成テスト
    model = test_model_creation()
    if model is None:
        return
    
    # 2. エンコーダーテスト
    latent = test_encoder(model)
    if latent is None:
        return
    
    # 3. デコーダーテスト
    decoded = test_decoder(model, latent)
    if decoded is None:
        return
    
    # 4. 完全なフォワードパステスト
    success = test_full_forward(model)
    if not success:
        return
    
    # 5. 段階的デコーダーテスト
    success = test_decoder_step_by_step(model)
    if not success:
        return
    
    print("\n🎉 All tests passed! Phase 4 model is working correctly.")
    print("=" * 50)

if __name__ == "__main__":
    main() 