# Gray-Scott Machine Learning Analysis

Gray-Scottモデルの時系列データに対する3D CNN Autoencoderを用いた機械学習分析プロジェクト

## プロジェクト構造

```
├── src/                    # ソースコード
│   ├── gray_scott_autoencoder.py    # メインのオートエンコーダー実装
│   ├── train_autoencoder.py         # オートエンコーダー学習専用
│   ├── cluster_analysis.py          # クラスター分析専用
│   ├── visualize_results.py         # 結果可視化専用
│   ├── optimal_clustering.py        # 包括的最適化分析
│   ├── create_cluster_gallery.py    # HTMLギャラリー作成
│   ├── main_workflow.py             # 統合ワークフロー
│   └── run_analysis.py              # メイン実行スクリプト
├── results/                # 結果ファイル
│   ├── *.png              # 可視化画像
│   ├── *.pkl              # 分析結果データ
│   ├── *.csv              # 結果テーブル
│   └── *.html             # HTMLギャラリー
├── models/                 # 訓練済みモデル
│   └── *.pth              # PyTorchモデルファイル
├── data/                   # データファイル
│   └── gif/               # GIFファイル
├── tests/                  # テストファイル
├── notebooks/              # Jupyter notebooks
├── docs/                   # ドキュメント
├── requirements.txt        # 依存関係
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

# 1. オートエンコーダー学習
python train_autoencoder.py

# 2. クラスター分析
python cluster_analysis.py

# 3. 結果可視化
python visualize_results.py
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
# 包括的最適化分析
python optimal_clustering.py

# HTMLギャラリー作成
python create_cluster_gallery.py
```

## 出力ファイル

### モデルファイル (`models/`)
- `trained_autoencoder.pth`: 学習済みオートエンコーダー

### 結果ファイル (`results/`)
- `analysis_results.pkl`: 可視化用データ
- `latent_representations.pkl`: 潜在表現データ
- `clustering_results.csv`: 結果テーブル
- `training_loss.png`: 学習曲線
- `gray_scott_clustering_results.png`: 統合可視化
- `gray_scott_detailed_heatmap.png`: 詳細ヒートマップ
- `silhouette_analysis_results.png`: シルエット分析

### HTMLギャラリー (`results/`)
- `cluster_gallery.html`: クラスターギャラリー
- `cluster_gallery_diverse.html`: 多様性重視ギャラリー

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

- `gray_scott_autoencoder.py`: コアクラスとモデル定義
- `train_autoencoder.py`: 学習専用スクリプト（高速反復実験用）
- `cluster_analysis.py`: クラスタリング専用スクリプト
- `visualize_results.py`: 可視化専用スクリプト
- `optimal_clustering.py`: 包括的分析（時間がかかる）

### 拡張方法

1. **新しい可視化の追加**: `visualize_results.py`に関数を追加
2. **新しいクラスタリング手法**: `cluster_analysis.py`を拡張
3. **モデルアーキテクチャの変更**: `gray_scott_autoencoder.py`のモデル定義を修正

## ライセンス

このプロジェクトはMITライセンスの下で公開されています。

## 更新履歴

- v1.0: 初期リリース
- v1.1: 分離ワークフロー対応
- v1.2: 可視化機能強化
- v1.3: ディレクトリ構造整理 