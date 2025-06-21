# Gray-Scott Machine Learning Analysis

Gray-Scottモデルの時系列データに対する3D CNN Autoencoderを用いた機械学習分析プロジェクト

**現在の規模**: 1500サンプル、256次元潜在空間、3段階進化アーキテクチャ（Phase 2完了）

## 🎯 Quick Start

### 最高性能システム (推奨)
```bash
python src/gray_scott_autoencoder_phase1.py  # シルエットスコア 0.565
```

### 物理解析システム (最新)
```bash
# Google Colab (GPU推奨)
# GrayScott_Phase2_Colab.ipynb を実行

# ローカル可視化
python src/visualize_phase2_results.py  # f-k空間マッピング付き
```

## 📊 Performance Summary

| システム | シルエットスコア | 改善率 | 特徴 |
|----------|------------------|--------|------|
| ベースライン | 0.413 | - | 基本3D CNN |
| **Phase 1** | **0.565** | **+36.8%** | **最高性能** |
| Phase 2 | 0.467 | +13.1% | 物理的意味 |

## プロジェクト構造

```
├── src/                    # ソースコード
│   ├── gray_scott_autoencoder.py         # メインのオートエンコーダー実装
│   ├── gray_scott_autoencoder_phase1.py  # Phase 1改善版オートエンコーダー
│   ├── train_autoencoder.py              # オートエンコーダー学習専用
│   ├── train_model.py                    # モデル訓練スクリプト
│   ├── main_workflow.py                  # 統合ワークフロー
│   ├── visualize_1500_samples.py         # 1500サンプル基本可視化
│   ├── create_1500_combined_visualization.py # 統合可視化（4プロット）
│   ├── optimal_cluster_analysis_1500.py  # 最適クラスタ数分析
│   ├── create_k4_visualization.py        # k=4クラスタリング専用
│   ├── create_k35_visualization.py       # k=35クラスタリング専用
│   ├── improved_3dcnn_architecture.py    # 改善アーキテクチャ設計
│   ├── implementation_roadmap.py         # 実装ロードマップ
│   └── check_new_data.py                 # データ検証ツール
├── results/                # 結果ファイル
│   ├── *_1500samples.png  # 1500サンプル可視化画像
│   ├── training_loss_*.png # 学習曲線
│   ├── latent_representations_frames_all.pkl # 潜在表現データ
│   ├── extracted_features_frames_all.csv     # 特徴データ
│   └── 3dcnn_improvement_roadmap.png         # 改善ロードマップ図
├── models/                 # 訓練済みモデル
│   └── *.pth              # PyTorchモデルファイル
├── data/                   # データファイル
│   └── gif/               # 1500個のGIFファイル
├── tests/                  # テストファイル
│   ├── test_functionality.py        # 機能テスト
│   ├── test_cluster_analysis.py     # クラスタリングテスト
│   └── test_interactive_clustering.py # インタラクティブテスト
├── notebooks/              # Jupyter notebooks
│   ├── cluster_analysis_notebook.ipynb    # クラスタ分析ノートブック
│   └── cluster_analysis_simple.ipynb     # シンプル分析ノートブック
├── docs/                   # ドキュメント
│   ├── latent_vectors_readme.md      # 潜在ベクトル説明
│   └── fix_visualization_readme.md   # 可視化修正説明
├── requirements.txt        # 依存関係
├── PROJECT_HISTORY.md      # プロジェクト履歴
├── TECHNICAL_SUMMARY.md    # 技術サマリー
├── PROJECT_SUMMARY.md      # プロジェクト概要
└── README.md              # このファイル
```

## 機能概要

### 1. データ処理
- GIFファイルからf/kパラメータを自動抽出
- 3D tensor形式 [C, D, H, W] への変換
- 固定フレーム数での正規化

### 2. 3D CNN Autoencoder
- 時系列データの特徴抽出
- 潜在空間での表現学習
- 教師なし学習による次元削減

### 3. クラスタリング分析
- K-meansクラスタリング
- シルエット分析による最適クラスター数決定
- PCA/t-SNEによる次元削減可視化

### 4. 可視化
- f-k平面でのクラスター分布
- ヒートマップ表示
- 潜在空間の可視化

## 使用方法

### 基本的な実行方法

```bash
# プロジェクトディレクトリに移動
cd src

# 1. 基本システム（Phase 0）
python gray_scott_autoencoder.py

# 2. 改善版システム（Phase 1）
python gray_scott_autoencoder_phase1.py

# 3. 個別学習専用
python train_autoencoder.py

# 4. 1500サンプル可視化
python visualize_1500_samples.py
```

### 統合ワークフロー

```bash
# インタラクティブメニュー
python main_workflow.py

# コマンドライン実行
python main_workflow.py --full          # 完全ワークフロー
python main_workflow.py --train         # 学習のみ
python main_workflow.py --cluster       # クラスター分析のみ
python main_workflow.py --visualize     # 可視化のみ
python main_workflow.py --status        # 状態確認
```

### 高度な分析

```bash
# 包括的最適化分析（1500サンプル）
python optimal_cluster_analysis_1500.py

# 特定クラスタ数での詳細分析
python create_k4_visualization.py   # k=4クラスタリング
python create_k35_visualization.py  # k=35クラスタリング

# 統合可視化（4プロット表示）
python create_1500_combined_visualization.py

# データ検証
python check_new_data.py
```

## 出力ファイル

### モデルファイル (`models/`)
- `trained_autoencoder.pth`: 基本学習済みオートエンコーダー
- `trained_autoencoder_phase1.pth`: Phase 1改善版モデル

### 結果ファイル (`results/`)
- `latent_representations_frames_all.pkl`: 1500サンプル潜在表現データ（814KB）
- `extracted_features_frames_all.csv`: 特徴データ（66KB）
- `training_loss_frames_all.png`: 学習曲線
- `gray_scott_clustering_results_1500samples.png`: 統合可視化（1500サンプル）
- `gray_scott_clustering_results_k4_1500samples.png`: k=4クラスタリング結果
- `gray_scott_clustering_results_k35_1500samples.png`: k=35クラスタリング結果
- `gray_scott_k35_heatmap_1500samples.png`: k=35詳細ヒートマップ
- `optimal_cluster_analysis_1500samples.png`: 最適クラスタ数分析
- `pca_scatter_1500samples.png`: PCA散布図
- `tsne_scatter_1500samples.png`: t-SNE散布図
- `fk_scatter_1500samples.png`: f-k平面散布図
- `3dcnn_improvement_roadmap.png`: 改善ロードマップ図

## 依存関係

```bash
pip install -r requirements.txt
```

主要パッケージ:
- PyTorch
- NumPy
- Matplotlib
- Scikit-learn
- Pandas
- Pillow
- Seaborn

## データ形式

GIFファイル名は以下の形式である必要があります：
```
GrayScott-f{f_value}-k{k_value}-{sequence}.gif
```

例: `GrayScott-f0.0580-k0.0680-00.gif`

## 設定可能パラメータ

### オートエンコーダー
- `latent_dim`: 潜在次元数 (デフォルト: 64)
- `fixed_frames`: 固定フレーム数 (デフォルト: 30)
- `target_size`: 画像サイズ (デフォルト: (64, 64))
- `num_epochs`: エポック数 (デフォルト: 50)
- `batch_size`: バッチサイズ (デフォルト: 4)

### クラスタリング
- `max_k`: 最大クラスター数 (デフォルト: 20)
- `random_state`: 乱数シード (デフォルト: 42)

## トラブルシューティング

### よくある問題

1. **メモリ不足**
   - `batch_size`を小さくする
   - `target_size`を小さくする

2. **学習が収束しない**
   - `learning_rate`を調整
   - `num_epochs`を増やす

3. **クラスタリング結果が不安定**
   - `random_state`を固定
   - より多くのエポックで学習

### ログとデバッグ

各スクリプトは詳細な進捗情報を出力します。エラーが発生した場合は、出力メッセージを確認してください。

## 開発者向け情報

### ファイル構成の詳細

#### 主要システム
- `gray_scott_autoencoder.py`: 基本システム（Phase 0）
- `gray_scott_autoencoder_phase1.py`: Phase 1改善版システム
- `train_autoencoder.py`: 学習専用スクリプト
- `train_model.py`: モデル訓練スクリプト
- `main_workflow.py`: 統合ワークフロー管理

#### 可視化・分析ツール
- `visualize_1500_samples.py`: 基本可視化（1500サンプル）
- `create_1500_combined_visualization.py`: 統合4プロット表示
- `optimal_cluster_analysis_1500.py`: 最適クラスタ数分析
- `create_k4_visualization.py`: k=4専用分析
- `create_k35_visualization.py`: k=35専用分析

#### 設計・ユーティリティ
- `improved_3dcnn_architecture.py`: 改善アーキテクチャ設計
- `implementation_roadmap.py`: 実装ロードマップ
- `check_new_data.py`: データ検証ツール

### 拡張方法

1. **新しい可視化の追加**: `visualize_1500_samples.py`ベースで新機能作成
2. **新しいクラスタリング手法**: `create_k4_visualization.py`を参考に拡張
3. **モデルアーキテクチャの変更**: `gray_scott_autoencoder_phase1.py`をベースに改良
4. **Phase 2以降の実装**: `implementation_roadmap.py`の計画に従って段階的実装

## ライセンス

このプロジェクトはMITライセンスの下で公開されています。

## 更新履歴

- v1.0: 初期リリース（375サンプル、64次元潜在空間）
- v1.1: 分離ワークフロー対応
- v1.2: 可視化機能強化
- v1.3: ディレクトリ構造整理
- v1.4: 1500サンプル対応（4倍データ拡張、128次元潜在空間）
- v1.5: クラスタリング分析最適化（k=4, k=35専用分析）
- v1.6: Phase 1改善実装（256次元潜在空間、AdamW、強化正則化）
- v1.7: プロジェクト構造最適化とREADME更新 