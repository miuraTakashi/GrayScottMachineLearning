#!/usr/bin/env python3
"""
Gray-Scott インタラクティブクラスタリングのテストスクリプト

このスクリプトは新しく追加されたインタラクティブ機能をテストします:
1. train_model.py のユーザー選択機能
2. optimal_clustering.py の包括的分析と選択機能
"""

import os
import subprocess
import sys

def check_requirements():
    """必要なファイルの存在確認"""
    required_files = [
        'train_model.py',
        'optimal_clustering.py',
        'visualize_results.py'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("❌ 以下のファイルが見つかりません:")
        for file in missing_files:
            print(f"   - {file}")
        return False
    
    return True

def run_interactive_training():
    """インタラクティブ学習の実行"""
    print("🚀 インタラクティブ学習を開始します...")
    print("="*60)
    print("📝 使用方法:")
    print("1. シルエット分析結果が表示されます")
    print("2. クラスター数を選択してください:")
    print("   - 推奨値の使用")
    print("   - 手動入力")
    print("   - グラフ表示")
    print("3. 選択後、学習が完了します")
    print("="*60)
    
    try:
        # train_model.pyを実行
        result = subprocess.run([sys.executable, 'train_model.py'], 
                              capture_output=False, text=True)
        
        if result.returncode == 0:
            print("✅ インタラクティブ学習が完了しました!")
            return True
        else:
            print("❌ 学習中にエラーが発生しました")
            return False
    
    except KeyboardInterrupt:
        print("\n⚠️  学習が中断されました")
        return False
    except Exception as e:
        print(f"❌ エラー: {e}")
        return False

def run_comprehensive_analysis():
    """包括的クラスター分析の実行"""
    print("\n🔬 包括的クラスター分析を開始します...")
    print("="*60)
    print("📊 この分析では以下を実行します:")
    print("1. Elbow Method")
    print("2. Silhouette Analysis")
    print("3. Gap Statistic")
    print("4. Hierarchical Clustering")
    print("5. 統合的な推奨値の提示")
    print("6. ユーザーによる最終選択")
    print("="*60)
    
    try:
        # optimal_clustering.pyを実行
        result = subprocess.run([sys.executable, 'optimal_clustering.py'], 
                              capture_output=False, text=True)
        
        if result.returncode == 0:
            print("✅ 包括的分析が完了しました!")
            return True
        else:
            print("❌ 分析中にエラーが発生しました")
            return False
    
    except KeyboardInterrupt:
        print("\n⚠️  分析が中断されました")
        return False
    except Exception as e:
        print(f"❌ エラー: {e}")
        return False

def run_visualization():
    """結果の可視化"""
    print("\n🎨 結果の可視化...")
    
    try:
        result = subprocess.run([sys.executable, 'visualize_results.py'], 
                              capture_output=False, text=True)
        
        if result.returncode == 0:
            print("✅ 可視化が完了しました!")
            return True
        else:
            print("❌ 可視化中にエラーが発生しました")
            return False
    
    except Exception as e:
        print(f"❌ エラー: {e}")
        return False

def main():
    print("🧪 Gray-Scott インタラクティブクラスタリング テスト")
    print("="*60)
    
    # 必要なファイルのチェック
    if not check_requirements():
        return
    
    print("✅ 必要なファイルが揃っています\n")
    
    # テストメニュー
    while True:
        print("\n🎯 テストメニュー:")
        print("1. インタラクティブ学習のテスト (train_model.py)")
        print("2. 包括的クラスター分析のテスト (optimal_clustering.py)")
        print("3. 結果の可視化 (visualize_results.py)")
        print("4. 全て実行")
        print("5. 終了")
        
        choice = input("\n選択 (1-5): ").strip()
        
        if choice == "1":
            if run_interactive_training():
                print("\n💡 次に包括的分析を実行することをお勧めします (選択肢2)")
        
        elif choice == "2":
            if not os.path.exists('analysis_results.pkl'):
                print("❌ analysis_results.pkl が見つかりません")
                print("   先にインタラクティブ学習を実行してください (選択肢1)")
            else:
                run_comprehensive_analysis()
        
        elif choice == "3":
            if not os.path.exists('analysis_results.pkl'):
                print("❌ analysis_results.pkl が見つかりません")
                print("   先に学習を実行してください")
            else:
                run_visualization()
        
        elif choice == "4":
            print("🔄 全ての処理を順次実行します...")
            
            # 1. 学習
            if run_interactive_training():
                print("\n⏳ 5秒後に分析を開始します...")
                import time
                time.sleep(5)
                
                # 2. 分析
                if run_comprehensive_analysis():
                    print("\n⏳ 3秒後に可視化を開始します...")
                    time.sleep(3)
                    
                    # 3. 可視化
                    run_visualization()
                    
                    print("\n🎉 全ての処理が完了しました!")
                    
                    # 生成されたファイルを表示
                    print("\n📁 生成されたファイル:")
                    output_files = [
                        'trained_model.pth',
                        'analysis_results.pkl',
                        'silhouette_optimization.png',
                        'comprehensive_clustering_analysis.png',
                        'clustering_analysis.png'
                    ]
                    
                    for file in output_files:
                        if os.path.exists(file):
                            size = os.path.getsize(file)
                            print(f"   ✅ {file} ({size:,} bytes)")
                        else:
                            print(f"   ❌ {file} (見つかりません)")
        
        elif choice == "5":
            print("👋 テストを終了します")
            break
        
        else:
            print("❌ 1-5 の数字を入力してください")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 テストが中断されました")
    except Exception as e:
        print(f"\n❌ 予期しないエラー: {e}") 