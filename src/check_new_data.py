#!/usr/bin/env python3
"""
新しい潜在表現データの確認
latent_representations_frames_all.pkl の内容をチェック
"""

import pickle
import numpy as np
import os

def check_new_data():
    """新しいデータファイルの内容を確認"""
    
    print("🔍 新しい潜在表現データの確認")
    print("=" * 50)
    
    # 新しいファイルを確認
    new_file = '../results/latent_representations_frames_all.pkl'
    old_file = '../results/analysis_results.pkl'
    
    if not os.path.exists(new_file):
        print(f"❌ {new_file} が見つかりません")
        return
    
    print(f"📁 Loading: {new_file}")
    
    try:
        with open(new_file, 'rb') as f:
            new_data = pickle.load(f)
        
        print("✅ データ読み込み成功")
        print(f"📊 データ型: {type(new_data)}")
        
        # データ構造の確認
        if isinstance(new_data, dict):
            print(f"📋 辞書のキー: {list(new_data.keys())}")
            
            for key, value in new_data.items():
                if isinstance(value, np.ndarray):
                    print(f"  🔢 {key}: shape={value.shape}, dtype={value.dtype}")
                elif isinstance(value, list):
                    print(f"  📝 {key}: length={len(value)}, type=list")
                else:
                    print(f"  📦 {key}: type={type(value)}")
        
        elif isinstance(new_data, np.ndarray):
            print(f"📊 配列形状: {new_data.shape}")
            print(f"📊 データ型: {new_data.dtype}")
        
        else:
            print(f"📦 その他のデータ型: {type(new_data)}")
        
        # 古いファイルとの比較
        if os.path.exists(old_file):
            print(f"\n🔄 古いファイルとの比較: {old_file}")
            
            with open(old_file, 'rb') as f:
                old_data = pickle.load(f)
            
            if isinstance(old_data, dict) and isinstance(new_data, dict):
                print("📊 サンプル数比較:")
                
                for key in ['f_values', 'k_values', 'cluster_labels', 'filenames']:
                    if key in old_data and key in new_data:
                        old_len = len(old_data[key])
                        new_len = len(new_data[key])
                        print(f"  {key}: {old_len} → {new_len} ({new_len/old_len:.1f}倍)")
        
        return new_data
        
    except Exception as e:
        print(f"❌ エラー: {e}")
        return None

def check_if_analysis_needed(data):
    """分析が必要かチェック"""
    
    print(f"\n🤔 分析状況の確認")
    print("=" * 30)
    
    if not isinstance(data, dict):
        print("❌ データが辞書形式ではありません")
        return True
    
    # クラスタリング結果があるかチェック
    clustering_keys = ['cluster_labels', 'pca_result', 'tsne_result']
    missing_keys = [key for key in clustering_keys if key not in data]
    
    if missing_keys:
        print(f"❌ 不足しているキー: {missing_keys}")
        print("💡 新しいクラスタリング分析が必要です")
        return True
    else:
        print("✅ クラスタリング結果は存在します")
        
        # サンプル数チェック
        if 'cluster_labels' in data:
            n_samples = len(data['cluster_labels'])
            n_clusters = len(np.unique(data['cluster_labels']))
            print(f"📊 サンプル数: {n_samples}, クラスター数: {n_clusters}")
        
        return False

def main():
    """メイン処理"""
    
    data = check_new_data()
    
    if data is not None:
        needs_analysis = check_if_analysis_needed(data)
        
        print(f"\n🎯 推奨アクション:")
        
        if needs_analysis:
            print("1. 🔄 新しいクラスタリング分析を実行")
            print("   python cluster_analysis.py")
            print("2. 🎨 可視化スクリプトを更新")
        else:
            print("1. 🎨 可視化スクリプトを新しいファイルに対応")
            print("2. 📊 結果の確認・比較")
    
    print(f"\n💡 ファイル情報:")
    print(f"   新しいデータ: latent_representations_frames_all.pkl")
    print(f"   古いデータ: analysis_results.pkl")

if __name__ == "__main__":
    main() 