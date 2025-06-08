# 🔬  Professional Statistical Analysis Software

**世界最高水準のプロフェッショナル統計解析プラットフォーム**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Mac%20M2%20%7C%20Linux-lightgrey.svg)]()
[![GPU](https://img.shields.io/badge/GPU-CUDA%20Enabled-orange.svg)]()

## 🎯 **概要**

Professional Statistical Analysis Softwareは、研究者・データサイエンティスト・統計専門家のための**次世代統計解析プラットフォーム**です。高度な統計手法、機械学習、ベイズ解析、生存解析を統合し、プロフェッショナルレベルの解析を実現します。

### ✨ **主要特徴**

- 🧠 **高度統計解析**: 多変量解析、時系列解析、包括的EDA
- ⏱️ **生存解析**: Kaplan-Meier、Cox回帰、パラメトリック生存解析
- 🎲 **ベイズ統計解析**: 階層モデル、MCMC、モデル比較
- 🤖 **機械学習パイプライン自動化**: AutoML、ハイパーパラメータ最適化
- 🌐 **多言語Webダッシュボード**: 日本語・英語対応、Mac M2最適化
- ⚡ **高性能処理**: GPU加速、並列処理、Numba JIT最適化
- 🛡️ **エンタープライズセキュリティ**: 自動バックアップ、セッション管理
- 📊 **プロフェッショナルレポート**: 論文品質の出力

## 🚀 **新機能ハイライト**

### 🔬 **完全統合された解析機能**

#### **1. 生存解析 (`survival_analysis.py`)**
```python
# Kaplan-Meier生存解析
results = complete_survival_analyzer.kaplan_meier_analysis(
    data, duration_col='time', event_col='event', group_col='treatment'
)

# Cox比例ハザード回帰
cox_results = complete_survival_analyzer.cox_regression_analysis(
    data, duration_col='time', event_col='event', 
    covariate_cols=['age', 'gender', 'treatment']
)
```

#### **2. ベイズ統計解析 (`bayesian_analysis.py`)**
```python
# ベイズ線形回帰
bayesian_results = deep_bayesian_analyzer.bayesian_linear_regression(
    data, target_col='outcome', predictor_cols=['x1', 'x2', 'x3']
)

# 階層ベイズモデリング
hierarchical_results = deep_bayesian_analyzer.hierarchical_modeling(
    data, target_col='outcome', predictor_cols=['x1', 'x2'], 
    group_col='cluster'
)
```

#### **3. 機械学習パイプライン自動化 (`ml_pipeline_automation.py`)**
```python
# 完全自動化機械学習パイプライン
ml_results = ml_pipeline_automator.complete_ml_pipeline(
    data, target_col='target', task_type='auto', 
    optimize_hyperparams=True
)
```

#### **4. 多言語Webダッシュボード (`web_dashboard.py`)**
```bash
# Webダッシュボード起動
py -3 run_web_dashboard.py
```

## 🛠️ **インストール**

### **必要条件**
- Python 3.9 以上
- Windows 10/11, macOS (Intel/M2/M3), Linux
- メモリ: 8GB以上推奨
- GPU: CUDA対応GPU（オプション）

### **1. リポジトリクローン**
```bash
git clone https://github.com/your-repo/HAD_backups.git
cd HAD_backups
```

### **2. 依存関係インストール**
```bash
# 基本インストール
pip install -r requirements.txt

# Mac M2最適化インストール（Mac使用者）
pip install --upgrade tensorflow-macos tensorflow-metal

# GPU加速（NVIDIA GPU使用者）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### **3. オプション: 高度機能**
```bash
# ベイズ統計解析（PyMC）
pip install pymc arviz

# 分散処理（大規模データ）
pip install dask[complete]
```

## 🎮 **使用方法**

### **デスクトップアプリケーション**
```bash
py -3 main.py
```
![Desktop Application](docs/images/desktop_app.png)

### **Webダッシュボード**
```bash
py -3 run_web_dashboard.py
```
- ブラウザで `http://localhost:8501` にアクセス
- 日本語・英語リアルタイム切り替え
- インタラクティブ解析・可視化

![Web Dashboard](docs/images/web_dashboard.png)

## 📊 **解析機能一覧**

### **基本統計解析**
- ✅ 記述統計・探索的データ解析
- ✅ 仮説検定（t検定、カイ二乗検定、ANOVA等）
- ✅ 相関分析・回帰分析
- ✅ データ品質チェック

### **高度統計解析**
- 🧠 **多変量解析**: PCA、因子分析、クラスタリング
- 📈 **時系列解析**: ARIMA、季節分解、定常性検定
- 📊 **包括的EDA**: 欠損値分析、外れ値検出、分布分析

### **生存解析**
- ⏱️ **Kaplan-Meier推定**: 生存曲線、信頼区間、リスクテーブル
- 📉 **Cox比例ハザード回帰**: ハザード比、比例ハザード仮定検定
- 📊 **パラメトリック生存解析**: Weibull、指数分布、対数正規分布
- 🔬 **統計検定**: Log-rank検定、多群比較
- 📈 **高度機能**: Nelson-Aalen、条件付き生存確率、RMST

### **ベイズ統計解析**
- 🎲 **ベイズ回帰**: 線形・ロジスティック回帰、不確実性定量化
- 🏗️ **階層ベイズモデリング**: グループ効果、分散成分
- 📈 **ベイズ時系列**: 状態空間モデル、予測
- ⚖️ **モデル比較**: WAIC、LOO、ベイズファクター
- 🔍 **診断**: トレースプロット、収束診断

### **機械学習**
- 🤖 **AutoML**: 自動前処理、特徴選択、モデル選択
- ⚡ **ハイパーパラメータ最適化**: Optuna、ベイズ最適化
- 🎯 **分類・回帰**: RF、XGBoost、LightGBM、ニューラルネット
- 📊 **評価**: クロスバリデーション、特徴重要度、混同行列
- 🔄 **パイプライン**: 完全自動化、モデル保存・読み込み

### **深層学習**
- 🧠 **ニューラルネット**: 分類・回帰、自動構築
- 🔁 **LSTM**: 時系列予測、sequence-to-sequence
- 🖼️ **CNN**: 画像解析、畳み込み層
- 🔀 **オートエンコーダ**: 次元削減、異常検知

## 🌟 **プロフェッショナル機能**

### **🛡️ セッション管理・復旧**
- 自動保存（5分間隔）
- 異常終了時の緊急保存
- バックアップローテーション（最大10ファイル）
- セッション復旧機能

### **📋 プロフェッショナルレポート**
- 論文品質のHTML/PDF出力
- 統計結果の自動整理
- グラフ・テーブルの統合
- カスタマイズ可能テンプレート

### **⚡ 高性能最適化**
- Numba JIT最適化
- 並列処理（マルチコア対応）
- GPU加速（CUDA）
- メモリ効率化

### **🤖 AI統合**
- OpenAI GPT統合
- Google Gemini対応
- 自然言語による解析指示
- 画像データ分析

## 🎨 **可視化機能**

- 📊 **統計グラフ**: ヒストグラム、散布図、箱ひげ図
- 📈 **高度な可視化**: PCA biplot、クラスター樹形図
- ⏱️ **生存曲線**: Kaplan-Meier、累積ハザード
- 🎲 **ベイズ診断**: 事後分布、トレースプロット
- 🤖 **ML評価**: ROC曲線、学習曲線、特徴重要度
- 🌐 **インタラクティブ**: Plotly、動的フィルタリング

## 🔧 **設定・カスタマイズ**

### **言語設定**
```python
# 日本語表示
app.set_language('ja')

# 英語表示  
app.set_language('en')
```

### **GPU設定**
```python
# CUDA利用設定
app.enable_gpu_acceleration(device='cuda:0')

# CPU並列処理
app.set_parallel_jobs(n_jobs=8)
```

### **Mac M2最適化**
```bash
# M2最適化環境変数
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
```

## 📁 **プロジェクト構造**

```
HAD_backups/
├── main.py                      # メインアプリケーション
├── web_dashboard.py             # Webダッシュボード
├── run_web_dashboard.py         # Web起動スクリプト
├── advanced_statistics.py       # 高度統計解析
├── survival_analysis.py         # 生存解析
├── bayesian_analysis.py         # ベイズ統計解析
├── ml_pipeline_automation.py    # ML自動化
├── parallel_optimization.py     # 並列処理最適化
├── professional_utils.py        # プロフェッショナル機能
├── professional_reports.py      # レポート生成
├── ai_integration.py            # AI統合
├── config.py                    # 設定管理
├── requirements.txt             # 依存関係
├── README.md                    # このファイル
├── LICENSE                      # ライセンス
├── backup/                      # 自動バックアップ
├── checkpoints/                 # チェックポイント
├── config/                      # 設定ファイル
├── data/                        # サンプルデータ
├── logs/                        # ログファイル
├── reports/                     # 生成レポート
└── templates/                   # レポートテンプレート
```

## 🎓 **使用例**

### **例1: 臨床試験の生存解析**
```python
import pandas as pd
from survival_analysis import complete_survival_analyzer

# データ読み込み
data = pd.read_csv('clinical_trial.csv')

# Kaplan-Meier解析
km_results = complete_survival_analyzer.kaplan_meier_analysis(
    data, 
    duration_col='survival_time',
    event_col='death_event',
    group_col='treatment_group'
)

# Cox回帰解析
cox_results = complete_survival_analyzer.cox_regression_analysis(
    data,
    duration_col='survival_time',
    event_col='death_event',
    covariate_cols=['age', 'gender', 'stage', 'treatment']
)

print(f"中央生存時間: {km_results['overall']['median_survival']}")
print(f"治療効果のハザード比: {cox_results['model_summary']['hazard_ratios']['treatment']}")
```

### **例2: ベイズA/Bテスト**
```python
from bayesian_analysis import deep_bayesian_analyzer

# A/Bテストデータ
ab_data = pd.read_csv('ab_test.csv')

# ベイズロジスティック回帰
bayesian_results = deep_bayesian_analyzer.bayesian_logistic_regression(
    ab_data,
    target_col='conversion',
    predictor_cols=['variant', 'age', 'device_type']
)

# 変換率向上の確率
variant_effect = bayesian_results['odds_ratios']['variant']
print(f"variant Bの改善確率: {variant_effect['probability_beneficial']:.2%}")
```

### **例3: 完全自動化機械学習**
```python
from ml_pipeline_automation import ml_pipeline_automator

# 自動化パイプライン実行
ml_results = ml_pipeline_automator.complete_ml_pipeline(
    data=sales_data,
    target_col='revenue',
    task_type='regression',
    optimize_hyperparams=True
)

# 結果確認
print(f"最良モデル: {ml_results['model_selection']['best_model']['name']}")
print(f"R²スコア: {ml_results['final_evaluation']['r2_score']:.4f}")
```

## 🔬 **研究・論文での使用**

### **統計手法の引用**
本ソフトウェアを研究で使用する際の引用例：

```
Statistical analyses were performed using HAD Professional Statistical Analysis Software v2.0 
(Minegishi, 2024), which implements advanced statistical methods including Kaplan-Meier survival 
analysis, Bayesian hierarchical modeling, and automated machine learning pipelines.
```

### **対応する統計手法**
- 生存解析: Kaplan & Meier (1958), Cox (1972)
- ベイズ統計: Gelman et al. (2013), McElreath (2020)
- 機械学習: Hastie et al. (2009), Bishop (2006)

## 🤝 **サポート・コミュニティ**

### **技術サポート**
- 📧 Email: support@had-statistics.com
- 💬 Discord: [HAD Community](https://discord.gg/had-stats)
- 📚 Documentation: [docs.had-statistics.com](https://docs.had-statistics.com)

### **貢献**
プルリクエスト、バグレポート、機能提案を歓迎します！

1. このリポジトリをフォーク
2. 機能ブランチを作成 (`git checkout -b feature/AmazingFeature`)
3. 変更をコミット (`git commit -m 'Add some AmazingFeature'`)
4. ブランチにプッシュ (`git push origin feature/AmazingFeature`)
5. プルリクエストを開く

## 📄 **ライセンス**

このプロジェクトはMITライセンスの下で配布されています。詳細は [LICENSE](LICENSE) ファイルを参照してください。

## 🙏 **謝辞**

- Python科学計算コミュニティ
- scikit-learn、PyMC、lifelines開発チーム
- 統計解析手法の研究者の皆様

---

## 🚀 **今すぐ始める**

```bash
# 1. リポジトリクローン
git clone https://github.com/your-repo/HAD_backups.git
cd HAD_backups

# 2. 依存関係インストール
pip install -r requirements.txt

# 3. アプリケーション起動
py -3 main.py

# または Webダッシュボード
py -3 run_web_dashboard.py
```

**世界最高水準の統計解析を、今すぐあなたの手に！** 🎉

---

*Made with ❤️ by Ryo Minegishi | © 2024 HAD Professional Statistical Analysis Software*
