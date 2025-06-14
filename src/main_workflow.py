#!/usr/bin/env python3
"""
Gray-Scott 分離ワークフロー統合スクリプト
オートエンコーダー学習とクラスター分析を分離した新しいワークフロー
"""

import os
import sys
import subprocess
import argparse

def check_file_exists(filename):
    """ファイルの存在確認"""
    return os.path.exists(filename)

def run_script(script_name, description):
    """スクリプトを実行"""
    print(f"\n🚀 {description}を開始...")
    print("=" * 60)
    
    try:
        result = subprocess.run([sys.executable, script_name], check=True)
        print(f"✅ {description}が完了しました")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description}中にエラーが発生しました: {e}")
        return False
    except KeyboardInterrupt:
        print(f"\n⚠️  {description}が中断されました")
        return False

def show_status():
    """現在の状態を表示"""
    print("📊 現在の状態:")
    print("=" * 40)
    
    # 学習済みモデルの確認
    if check_file_exists('../models/trained_autoencoder.pth'):
        size_mb = os.path.getsize('../models/trained_autoencoder.pth') / 1024 / 1024
        print(f"✅ オートエンコーダーモデル: trained_autoencoder.pth ({size_mb:.1f}MB)")
    else:
        print("❌ オートエンコーダーモデル: 未学習")
    
    # 潜在表現データの確認
    if check_file_exists('../results/latent_representations.pkl'):
        size_kb = os.path.getsize('../results/latent_representations.pkl') / 1024
        print(f"✅ 潜在表現データ: latent_representations.pkl ({size_kb:.1f}KB)")
    else:
        print("❌ 潜在表現データ: 未生成")
    
    # クラスタリング結果の確認
    if check_file_exists('../results/analysis_results.pkl'):
        size_kb = os.path.getsize('../results/analysis_results.pkl') / 1024
        print(f"✅ クラスタリング結果: analysis_results.pkl ({size_kb:.1f}KB)")
    else:
        print("❌ クラスタリング結果: 未実行")
    
    # 可視化ファイルの確認
    viz_files = [
        '../results/training_loss.png',
        '../results/silhouette_analysis_results.png',
        '../results/comprehensive_clustering_analysis.png',
        '../results/clustering_analysis.png'
    ]
    
    existing_viz = [f for f in viz_files if check_file_exists(f)]
    if existing_viz:
        print(f"✅ 可視化ファイル: {len(existing_viz)}/{len(viz_files)}個")
        for f in existing_viz:
            print(f"   - {f}")
    else:
        print("❌ 可視化ファイル: 未生成")

def interactive_menu():
    """インタラクティブメニュー"""
    while True:
        print("\n🎯 Gray-Scott 分離ワークフロー")
        print("=" * 50)
        
        show_status()
        
        print("\n🔧 利用可能なオプション:")
        print("1. オートエンコーダー学習 (train_autoencoder.py)")
        print("2. クラスター分析 (cluster_analysis.py)")
        print("3. 包括的最適化分析 (optimal_clustering.py)")
        print("4. 結果の可視化 (visualize_results.py)")
        print("5. HTMLギャラリー作成 (create_cluster_gallery.py)")
        print("6. 完全ワークフロー実行 (1→2→4)")
        print("7. 状態確認のみ")
        print("8. 終了")
        
        choice = input("\n選択 (1-8): ").strip()
        
        if choice == "1":
            if run_script('train_autoencoder.py', 'オートエンコーダー学習'):
                print("\n💡 次に推奨するステップ:")
                print("  - クラスター分析 (選択肢2)")
                print("  - または包括的分析 (選択肢3)")
        
        elif choice == "2":
            if not check_file_exists('../results/latent_representations.pkl'):
                print("❌ 潜在表現データが見つかりません")
                print("   先にオートエンコーダー学習を実行してください (選択肢1)")
            else:
                if run_script('cluster_analysis.py', 'クラスター分析'):
                    print("\n💡 次に推奨するステップ:")
                    print("  - 結果の可視化 (選択肢4)")
                    print("  - HTMLギャラリー作成 (選択肢5)")
        
        elif choice == "3":
            if not check_file_exists('../results/latent_representations.pkl'):
                print("❌ 潜在表現データが見つかりません")
                print("   先にオートエンコーダー学習を実行してください (選択肢1)")
            else:
                run_script('optimal_clustering.py', '包括的最適化分析')
        
        elif choice == "4":
            if not check_file_exists('../results/analysis_results.pkl'):
                print("❌ クラスタリング結果が見つかりません")
                print("   先にクラスター分析を実行してください (選択肢2)")
            else:
                run_script('visualize_results.py', '結果の可視化')
        
        elif choice == "5":
            if not check_file_exists('../results/analysis_results.pkl'):
                print("❌ クラスタリング結果が見つかりません")
                print("   先にクラスター分析を実行してください (選択肢2)")
            else:
                run_script('create_cluster_gallery.py', 'HTMLギャラリー作成')
        
        elif choice == "6":
            print("🔄 完全ワークフローを実行します...")
            
            # ステップ1: オートエンコーダー学習
            if run_script('train_autoencoder.py', 'オートエンコーダー学習'):
                print("\n⏳ 次のステップに進みます...")
                
                # ステップ2: クラスター分析
                if run_script('cluster_analysis.py', 'クラスター分析'):
                    print("\n⏳ 最終ステップに進みます...")
                    
                    # ステップ3: 可視化
                    run_script('visualize_results.py', '結果の可視化')
                    
                    print("\n🎉 完全ワークフローが完了しました!")
                    
                    # 最終状態表示
                    show_status()
        
        elif choice == "7":
            continue  # 状態確認は上部で表示されているので何もしない
        
        elif choice == "8":
            print("👋 ワークフローを終了します")
            break
        
        else:
            print("❌ 1-8 の数字を入力してください")

def main():
    parser = argparse.ArgumentParser(description='Gray-Scott 分離ワークフロー統合スクリプト')
    parser.add_argument('--train', action='store_true', 
                       help='オートエンコーダー学習のみ実行')
    parser.add_argument('--cluster', action='store_true', 
                       help='クラスター分析のみ実行')
    parser.add_argument('--optimize', action='store_true', 
                       help='包括的最適化分析のみ実行')
    parser.add_argument('--visualize', action='store_true', 
                       help='結果の可視化のみ実行')
    parser.add_argument('--gallery', action='store_true', 
                       help='HTMLギャラリー作成のみ実行')
    parser.add_argument('--full', action='store_true', 
                       help='完全ワークフロー実行 (train→cluster→visualize)')
    parser.add_argument('--status', action='store_true', 
                       help='現在の状態確認のみ')
    
    args = parser.parse_args()
    
    print("🔬 Gray-Scott 分離ワークフロー統合スクリプト")
    print("=" * 60)
    print("📝 概要: オートエンコーダー学習とクラスター分析を分離した新しいワークフロー")
    print("🎯 利点: 学習とクラスタリングの独立実行、高速な反復実験")
    
    # コマンドライン引数による実行
    if args.status:
        show_status()
        return
    
    elif args.train:
        run_script('train_autoencoder.py', 'オートエンコーダー学習')
    
    elif args.cluster:
        if check_file_exists('../results/latent_representations.pkl'):
            run_script('cluster_analysis.py', 'クラスター分析')
        else:
            print("❌ 潜在表現データが見つかりません。先にオートエンコーダー学習を実行してください。")
    
    elif args.optimize:
        if check_file_exists('../results/latent_representations.pkl'):
            run_script('optimal_clustering.py', '包括的最適化分析')
        else:
            print("❌ 潜在表現データが見つかりません。先にオートエンコーダー学習を実行してください。")
    
    elif args.visualize:
        if check_file_exists('analysis_results.pkl'):
            run_script('visualize_results.py', '結果の可視化')
        else:
            print("❌ クラスタリング結果が見つかりません。先にクラスター分析を実行してください。")
    
    elif args.gallery:
        if check_file_exists('analysis_results.pkl'):
            run_script('create_cluster_gallery.py', 'HTMLギャラリー作成')
        else:
            print("❌ クラスタリング結果が見つかりません。先にクラスター分析を実行してください。")
    
    elif args.full:
        print("🔄 完全ワークフローを実行します...")
        
        if run_script('train_autoencoder.py', 'オートエンコーダー学習'):
            if run_script('cluster_analysis.py', 'クラスター分析'):
                run_script('visualize_results.py', '結果の可視化')
                print("\n🎉 完全ワークフローが完了しました!")
    
    else:
        # 引数がない場合はインタラクティブメニュー
        interactive_menu()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 プログラムが中断されました")
    except Exception as e:
        print(f"\n❌ 予期しないエラー: {e}")
        import traceback
        traceback.print_exc() 