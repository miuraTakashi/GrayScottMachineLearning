#!/usr/bin/env python3
"""
Gray-Scott Project Cleanup Tool
不要な実験ファイルを整理
"""

import os
import shutil
from pathlib import Path

def analyze_files():
    """現在のファイル構成を分析"""
    
    print("📂 Gray-Scott プロジェクト ファイル分析")
    print("=" * 50)
    
    # 必要なファイル（保持すべき）
    essential_files = {
        'src/train_autoencoder.py': '✅ 学習メインスクリプト',
        'src/cluster_analysis.py': '✅ クラスタ分析',
        'src/visualize_results.py': '✅ 可視化（修正済み）',
        'src/simple_visualize.py': '✅ 可視化（エラー回避版）',
        'src/main_workflow.py': '✅ ワークフロー統合',
        'README.md': '✅ プロジェクト説明',
        'requirements.txt': '✅ 依存関係',
    }
    
    # 実験・開発用ファイル（削除候補）
    experimental_files = {
        'quick_fix_visualize.py': '🧪 一時的な修正ツール',
        'scalable_improvements.py': '🧪 スケーラビリティ実験',
        'scalability_analysis.py': '🧪 スケーラビリティ分析',
        'improvement_phase1.py': '🧪 性能改善実験',
        'quick_analysis.py': '🧪 クイック分析',
        'analyze_current_performance.py': '🧪 性能分析実験',
        'improve_classification_accuracy.py': '🧪 精度向上実験',
        'test_frame_range.py': '🧪 フレーム範囲テスト',
        'run_analysis.sh': '🧪 分析実行スクリプト',
    }
    
    print("📋 必要なファイル:")
    for file, desc in essential_files.items():
        status = "✓" if os.path.exists(file) else "✗"
        print(f"  {status} {file} - {desc}")
    
    print("\n🧪 実験ファイル（削除候補）:")
    existing_experimental = []
    for file, desc in experimental_files.items():
        if os.path.exists(file):
            existing_experimental.append(file)
            size = os.path.getsize(file) / 1024
            print(f"  📄 {file} ({size:.1f}KB) - {desc}")
    
    return existing_experimental

def cleanup_files(file_list, create_backup=True):
    """ファイルをクリーンアップ"""
    
    if not file_list:
        print("🎉 削除対象のファイルはありません")
        return
    
    if create_backup:
        # バックアップディレクトリ作成
        backup_dir = "backup_experimental_files"
        os.makedirs(backup_dir, exist_ok=True)
        print(f"📦 バックアップディレクトリ作成: {backup_dir}/")
    
    total_size = 0
    
    for file_path in file_list:
        try:
            size = os.path.getsize(file_path)
            total_size += size
            
            if create_backup:
                # バックアップに移動
                backup_path = os.path.join(backup_dir, os.path.basename(file_path))
                shutil.move(file_path, backup_path)
                print(f"  📦 {file_path} → {backup_path}")
            else:
                # 完全削除
                os.remove(file_path)
                print(f"  🗑️  {file_path} 削除")
                
        except Exception as e:
            print(f"  ❌ {file_path} エラー: {e}")
    
    print(f"\n💾 解放された容量: {total_size/1024:.1f} KB")

def check_directory_structure():
    """ディレクトリ構造の確認"""
    
    print("\n📁 現在のディレクトリ構造:")
    print("=" * 30)
    
    required_dirs = ['src', 'data', 'models', 'results', 'tests', 'docs', 'notebooks']
    
    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            file_count = len([f for f in Path(dir_name).rglob('*') if f.is_file()])
            print(f"  ✅ {dir_name}/ ({file_count} files)")
        else:
            print(f"  ❌ {dir_name}/ (missing)")

def create_project_summary():
    """プロジェクト概要ファイルを作成"""
    
    summary_content = """# Gray-Scott Machine Learning Project

## 🎯 プロジェクト概要
Gray-Scottパターンの機械学習による分類プロジェクト

## 📁 ディレクトリ構造
```
GrayScottMachineLearning/
├── src/                    # メインコード
│   ├── train_autoencoder.py    # 学習メインスクリプト
│   ├── cluster_analysis.py     # クラスタ分析
│   ├── visualize_results.py    # 可視化（修正済み）
│   ├── simple_visualize.py     # 可視化（エラー回避版）
│   └── main_workflow.py        # ワークフロー統合
├── data/                   # データ
│   └── gif/               # GIFファイル（375個）
├── models/                # 学習済みモデル
├── results/               # 結果・画像
├── tests/                 # テストファイル
├── docs/                  # ドキュメント
├── notebooks/             # Jupyterノートブック
├── README.md              # プロジェクト説明
└── requirements.txt       # 依存関係
```

## 🚀 基本的な使用方法

### 1. 学習実行
```bash
cd src
python train_autoencoder.py
```

### 2. 可視化
```bash
python simple_visualize.py  # エラー回避版（推奨）
python visualize_results.py  # 通常版
```

### 3. 統合ワークフロー
```bash
python main_workflow.py
```

## 📊 現在の性能
- 375サンプル、20クラスター
- シルエットスコア: 0.551
- フレーム範囲指定機能あり

## 🔧 開発済み機能
- フレーム範囲指定でのGIF処理
- PCA・t-SNE可視化
- f-kパラメータ空間分析
- viridisエラー回避機能

---
Generated by cleanup tool
"""
    
    with open('PROJECT_SUMMARY.md', 'w', encoding='utf-8') as f:
        f.write(summary_content)
    
    print("📋 PROJECT_SUMMARY.md を作成しました")

def main():
    """メイン処理"""
    
    print("🧹 Gray-Scott プロジェクト クリーンアップツール")
    print("=" * 50)
    
    # ファイル分析
    experimental_files = analyze_files()
    
    # ディレクトリ構造確認
    check_directory_structure()
    
    if experimental_files:
        print(f"\n🗂️  実験ファイル {len(experimental_files)} 個が見つかりました")
        print("\n📝 推奨アクション:")
        print("1. 🔄 バックアップして削除（安全）")
        print("2. 🗑️  完全削除（容量節約）")
        print("3. ⏹️  何もしない")
        
        choice = input("\n選択してください (1/2/3): ").strip()
        
        if choice == "1":
            print("\n🔄 バックアップして削除を実行...")
            cleanup_files(experimental_files, create_backup=True)
            print("✅ 完了！実験ファイルは backup_experimental_files/ に保存されました")
            
        elif choice == "2":
            confirm = input("⚠️  完全削除しますか？ (yes/no): ").strip().lower()
            if confirm == "yes":
                print("\n🗑️  完全削除を実行...")
                cleanup_files(experimental_files, create_backup=False)
                print("✅ 完了！")
            else:
                print("❌ キャンセルしました")
                
        elif choice == "3":
            print("⏹️  何も変更しませんでした")
            
        else:
            print("❌ 無効な選択です")
    
    # プロジェクト概要作成
    create_project_summary()
    
    print("\n🎉 クリーンアップ完了！")
    print("💡 基本的な解析には src/ フォルダ内のファイルのみで十分です")

if __name__ == "__main__":
    main() 