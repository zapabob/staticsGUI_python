#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
サンプルデータ生成スクリプト
プロフェッショナル統計ソフトウェアのテスト用
"""
import pandas as pd
import numpy as np
from pathlib import Path

def create_sample_datasets():
    """サンプルデータセット作成"""
    print("📊 サンプルデータセット作成中...")
    
    # データディレクトリ作成
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # 1. 基本統計用データ (正規分布・非正規分布混在)
    np.random.seed(42)
    n = 1000
    
    basic_data = pd.DataFrame({
        'normal_data': np.random.normal(100, 15, n),
        'skewed_data': np.random.exponential(2, n),
        'uniform_data': np.random.uniform(0, 100, n),
        'bimodal_data': np.concatenate([
            np.random.normal(80, 10, n//2),
            np.random.normal(120, 10, n//2)
        ]),
        'categorical': np.random.choice(['A', 'B', 'C'], n, p=[0.5, 0.3, 0.2]),
        'binary': np.random.choice([0, 1], n),
        'age': np.random.randint(18, 80, n),
        'income': np.random.lognormal(10.5, 0.5, n)
    })
    
    # 欠損値を意図的に作成
    basic_data.loc[np.random.choice(basic_data.index, 50), 'income'] = np.nan
    basic_data.loc[np.random.choice(basic_data.index, 30), 'age'] = np.nan
    
    basic_data.to_csv(data_dir / "basic_statistics_sample.csv", index=False)
    print(f"✅ 基本統計サンプル: {len(basic_data)} rows")
    
    # 2. 回帰分析用データ
    x1 = np.random.normal(0, 1, n)
    x2 = np.random.normal(0, 1, n)
    x3 = np.random.normal(0, 1, n)
    
    # 線形関係 + ノイズ
    y_linear = 2.5 + 1.5*x1 + 0.8*x2 - 0.5*x3 + np.random.normal(0, 0.5, n)
    
    # 非線形関係
    y_nonlinear = 10 + 2*x1**2 + np.sin(x2*2) + np.exp(x3*0.1) + np.random.normal(0, 1, n)
    
    regression_data = pd.DataFrame({
        'x1': x1,
        'x2': x2, 
        'x3': x3,
        'y_linear': y_linear,
        'y_nonlinear': y_nonlinear,
        'group': np.random.choice(['Treatment', 'Control'], n)
    })
    
    regression_data.to_csv(data_dir / "regression_sample.csv", index=False)
    print(f"✅ 回帰分析サンプル: {len(regression_data)} rows")
    
    # 3. 時系列データ
    dates = pd.date_range('2020-01-01', periods=365*3, freq='D')
    trend = np.linspace(100, 150, len(dates))
    seasonal = 10 * np.sin(2 * np.pi * np.arange(len(dates)) / 365.25)
    noise = np.random.normal(0, 5, len(dates))
    
    timeseries_data = pd.DataFrame({
        'date': dates,
        'value': trend + seasonal + noise,
        'category': np.random.choice(['A', 'B', 'C'], len(dates))
    })
    
    timeseries_data.to_csv(data_dir / "timeseries_sample.csv", index=False)
    print(f"✅ 時系列サンプル: {len(timeseries_data)} rows")
    
    # 4. 機械学習用データ（分類）
    from sklearn.datasets import make_classification
    
    X, y = make_classification(
        n_samples=1500,
        n_features=10,
        n_informative=7,
        n_redundant=2,
        n_clusters_per_class=1,
        random_state=42
    )
    
    ml_data = pd.DataFrame(
        X, 
        columns=[f'feature_{i+1}' for i in range(X.shape[1])]
    )
    ml_data['target'] = y
    ml_data['target_name'] = ['Class_A' if t == 0 else 'Class_B' for t in y]
    
    ml_data.to_csv(data_dir / "machine_learning_sample.csv", index=False)
    print(f"✅ 機械学習サンプル: {len(ml_data)} rows")
    
    # 5. 医学統計データ（生存分析）
    survival_time = np.random.exponential(200, 500)
    censoring_time = np.random.exponential(300, 500)
    observed_time = np.minimum(survival_time, censoring_time)
    event = survival_time <= censoring_time
    
    medical_data = pd.DataFrame({
        'patient_id': range(1, 501),
        'survival_time': observed_time,
        'event': event.astype(int),
        'treatment': np.random.choice(['Drug_A', 'Drug_B', 'Placebo'], 500),
        'age': np.random.normal(65, 12, 500),
        'gender': np.random.choice(['M', 'F'], 500),
        'stage': np.random.choice(['I', 'II', 'III', 'IV'], 500, p=[0.2, 0.3, 0.3, 0.2])
    })
    
    medical_data.to_csv(data_dir / "medical_survival_sample.csv", index=False)
    print(f"✅ 医学統計サンプル: {len(medical_data)} rows")
    
    print(f"\n🎯 全てのサンプルデータが {data_dir.absolute()} に保存されました！")
    
    return {
        'basic': basic_data,
        'regression': regression_data,
        'timeseries': timeseries_data,
        'ml': ml_data,
        'medical': medical_data
    }

def main():
    """メイン関数"""
    print("🔬 Professional Statistical Analysis Software")
    print("📊 Sample Data Generator")
    print("=" * 50)
    
    datasets = create_sample_datasets()
    
    print("\n📋 作成されたデータセット:")
    for name, df in datasets.items():
        print(f"• {name}: {df.shape[0]} rows × {df.shape[1]} columns")
    
    print("\n🚀 これらのデータでソフトウェアの機能をテストできます！")

if __name__ == "__main__":
    main() 