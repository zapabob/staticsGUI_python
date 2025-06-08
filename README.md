# Python GUI Statistical Analysis Software

## 概要

このソフトウェアは、Python で開発された直感的なGUIベースの統計解析ツールです。データサイエンティストや研究者が簡単に統計解析と機械学習を実行できるよう設計されています。

## 主な機能

### 📊 統計解析機能
- **記述統計**: 平均、中央値、標準偏差、分散などの基本統計量
- **推定統計**: 信頼区間、仮説検定、回帰分析
- **多変量解析**: 主成分分析、因子分析、クラスター分析
- **時系列解析**: トレンド分析、季節性分解、予測モデル

### 🤖 機械学習機能
- **教師あり学習**: 回帰、分類、決定木、ランダムフォレスト
- **教師なし学習**: クラスタリング、次元削減
- **深層学習**: ニューラルネットワーク、CNN、RNN
- **モデル評価**: 交差検証、性能メトリクス、ROC曲線

### 📈 可視化機能
- インタラクティブなグラフ作成
- カスタマイズ可能なプロット
- 統計チャート（ヒストグラム、散布図、箱ひげ図）
- 3D可視化対応

### 🛡️ 高度な機能
- **電源断保護**: 自動チェックポイント保存システム
- **GPU加速**: CUDA対応による高速計算
- **セッション管理**: 作業内容の自動保存・復旧
- **データ整合性保護**: 異常終了時の自動データ保護

## システム要件

### 推奨環境
- **OS**: Windows 11 / Windows 10
- **Python**: 3.8以上
- **GPU**: NVIDIA RTX 3080以上（CUDA計算用）
- **RAM**: 16GB以上
- **ストレージ**: 10GB以上の空き容量

### 必要なライブラリ
```bash
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
tensorflow>=2.8.0
torch>=1.11.0
tqdm>=4.62.0
tkinter
plotly>=5.0.0
```

## インストール方法

### 1. リポジトリのクローン
```bash
git clone https://github.com/zapabob/python-gui-statistics.git
cd python-gui-statistics
```

### 2. 仮想環境の作成
```bash
python -m venv venv
venv\Scripts\activate  # Windows環境
```

### 3. 依存関係のインストール
```bash
py -3 -m pip install -r requirements.txt
```

### 4. CUDA環境の設定
```bash
py -3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## 使用方法

### 基本的な起動方法
```bash
py -3 main.py
```

### GUI操作
1. **データ読み込み**: CSV、Excel、JSON形式のファイルをサポート
2. **解析手法選択**: ドロップダウンメニューから統計手法を選択
3. **パラメータ設定**: GUI上で解析パラメータを直感的に設定
4. **実行・結果表示**: ワンクリックで解析実行、結果を自動可視化

### 電源断保護機能
- **自動保存**: 5分間隔での定期的なセッション保存
- **緊急保存**: Ctrl+C や異常終了時の自動保存
- **復旧機能**: 前回セッションからの自動復旧
- **バックアップ管理**: 最大10個のバックアップ自動ローテーション

## 特徴

### 🚀 高性能計算
- NVIDIA RTX 3080 GPU対応
- CUDAによる並列計算最適化
- メモリ効率的なデータ処理

### 🔒 データ保護
- 異常終了検出システム
- JSON + Pickle複合保存形式
- セッション固有ID管理
- データ整合性チェック

### 🎨 ユーザーフレンドリー
- 直感的なGUIインターフェース
- リアルタイム進捗表示（tqdm）
- 英語キャプション対応グラフ
- カスタマイズ可能な出力形式

## プロジェクト構成

```
project/
├── main.py              # メインアプリケーション
├── gui/                 # GUI関連モジュール
├── statistics/          # 統計解析モジュール
├── ml/                  # 機械学習モジュール
├── visualization/       # 可視化モジュール
├── utils/              # ユーティリティ
├── data/               # サンプルデータ
├── checkpoints/        # チェックポイントファイル
├── backup/             # バックアップファイル
└── docs/               # ドキュメント
```

## 貢献方法

1. このリポジトリをフォーク
2. 機能ブランチを作成 (`git checkout -b feature/AmazingFeature`)
3. 変更をコミット (`git commit -m 'Add some AmazingFeature'`)
4. ブランチにプッシュ (`git push origin feature/AmazingFeature`)
5. プルリクエストを作成

## トラブルシューティング

### よくある問題

**Q: CUDA エラーが発生する**
A: NVIDIA ドライバーとCUDA Toolkitが正しくインストールされているか確認してください。

**Q: セッションが復旧できない**
A: `checkpoints/` ディレクトリ内のファイルを確認し、最新のバックアップから復旧してください。

**Q: メモリ不足エラー**
A: バッチサイズを小さくするか、データを分割して処理してください。

## ライセンス

このプロジェクトは MIT ライセンスの下で公開されています。詳細は [LICENSE](LICENSE) ファイルをご覧ください。

## 作者

- **開発者**: [Your Name]
- **メール**: [your.email@example.com]
- **GitHub**: [@yourusername](https://github.com/yourusername)

## 謝辞

- NumPy、Pandas、Scikit-learn開発チーム
- TensorFlow、PyTorch開発コミュニティ
- すべての貢献者とベータテスター

---

⚡ **Powered by NVIDIA RTX 3080** | 🛡️ **電源断保護システム搭載** | 📊 **Professional Statistical Analysis** 