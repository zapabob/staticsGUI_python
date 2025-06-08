#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
機械学習・深層学習機能テストスクリプト
"""
import pandas as pd
import numpy as np
from pathlib import Path

# 機械学習ライブラリ
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb

# 深層学習ライブラリ
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔥 PyTorch: Available | Device: {DEVICE}")
except ImportError:
    TORCH_AVAILABLE = False
    print("❌ PyTorch: Not Available")

def test_ml_algorithms():
    """機械学習アルゴリズムテスト"""
    print("\n🤖 機械学習アルゴリズムテスト")
    print("=" * 50)
    
    # サンプルデータ読み込み
    data_path = Path("data/machine_learning_sample.csv")
    if not data_path.exists():
        print("❌ サンプルデータが見つかりません。先にsample_data.pyを実行してください。")
        return
    
    df = pd.read_csv(data_path)
    print(f"📊 データサイズ: {df.shape}")
    
    # 特徴量とターゲット
    feature_cols = [col for col in df.columns if col.startswith('feature_')]
    X = df[feature_cols]
    y = df['target']
    
    # データ分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # アルゴリズムテスト
    algorithms = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
        "LightGBM": lgb.LGBMClassifier(random_state=42, verbose=-1)
    }
    
    results = {}
    
    for name, model in algorithms.items():
        print(f"\n🔧 {name} テスト中...")
        
        # 学習
        model.fit(X_train, y_train)
        
        # 予測
        y_pred = model.predict(X_test)
        
        # 評価
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = accuracy
        
        print(f"✅ {name} Accuracy: {accuracy:.4f}")
        
        # 特徴量重要度
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            top_features = sorted(zip(feature_cols, importance), key=lambda x: x[1], reverse=True)[:3]
            print(f"   Top 3 features: {[f'{feat}: {imp:.3f}' for feat, imp in top_features]}")
    
    # 結果サマリー
    print(f"\n📈 機械学習テスト結果:")
    for name, acc in results.items():
        print(f"• {name}: {acc:.4f}")
    
    best_model = max(results, key=results.get)
    print(f"🏆 最高性能: {best_model} ({results[best_model]:.4f})")

def test_deep_learning():
    """深層学習テスト"""
    if not TORCH_AVAILABLE:
        print("\n❌ PyTorchが利用できないため、深層学習テストをスキップします。")
        return
    
    print("\n🧠 深層学習テスト")
    print("=" * 50)
    
    # サンプルデータ準備
    data_path = Path("data/machine_learning_sample.csv")
    df = pd.read_csv(data_path)
    
    feature_cols = [col for col in df.columns if col.startswith('feature_')]
    X = df[feature_cols].values
    y = df['target'].values
    
    # データ分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # データを正規化
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # PyTorchテンサーに変換
    X_train_tensor = torch.FloatTensor(X_train_scaled).to(DEVICE)
    X_test_tensor = torch.FloatTensor(X_test_scaled).to(DEVICE)
    y_train_tensor = torch.LongTensor(y_train).to(DEVICE)
    y_test_tensor = torch.LongTensor(y_test).to(DEVICE)
    
    print(f"📱 使用デバイス: {DEVICE}")
    print(f"🎯 クラス数: {len(np.unique(y))}")
    print(f"📊 特徴量数: {X_train.shape[1]}")
    
    # ニューラルネットワーク定義
    class SimpleNN(nn.Module):
        def __init__(self, input_size, hidden_size, num_classes):
            super(SimpleNN, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, hidden_size//2)
            self.fc3 = nn.Linear(hidden_size//2, num_classes)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.2)
            self.batch_norm1 = nn.BatchNorm1d(hidden_size)
            self.batch_norm2 = nn.BatchNorm1d(hidden_size//2)
            
        def forward(self, x):
            x = self.relu(self.batch_norm1(self.fc1(x)))
            x = self.dropout(x)
            x = self.relu(self.batch_norm2(self.fc2(x)))
            x = self.dropout(x)
            x = self.fc3(x)
            return x
    
    # モデル初期化
    input_size = X_train.shape[1]
    hidden_size = 128
    num_classes = len(np.unique(y))
    
    model = SimpleNN(input_size, hidden_size, num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print(f"🏗️ モデル構造: {input_size} → {hidden_size} → {hidden_size//2} → {num_classes}")
    
    # 学習
    model.train()
    epochs = 50
    print(f"🎓 学習開始: {epochs} epochs")
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"   Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
    
    # 評価
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        _, predicted = torch.max(test_outputs.data, 1)
        
    y_pred = predicted.cpu().numpy()
    y_test_np = y_test_tensor.cpu().numpy()
    
    accuracy = accuracy_score(y_test_np, y_pred)
    print(f"\n🎯 深層学習結果:")
    print(f"• Accuracy: {accuracy:.4f}")
    print(f"• Device: {DEVICE}")
    print(f"• Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # GPU使用状況（CUDA利用時）
    if torch.cuda.is_available():
        print(f"• GPU Memory Allocated: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
        print(f"• GPU Memory Cached: {torch.cuda.memory_reserved() / 1024**2:.1f} MB")

def test_advanced_features():
    """高度な機能テスト"""
    print("\n⚡ 高度な機能テスト")
    print("=" * 50)
    
    # GPU状況
    if torch.cuda.is_available():
        print(f"✅ CUDA Available: {torch.cuda.is_available()}")
        print(f"✅ GPU Count: {torch.cuda.device_count()}")
        print(f"✅ Current Device: {torch.cuda.current_device()}")
        print(f"✅ Device Name: {torch.cuda.get_device_name(0)}")
        print(f"✅ CUDA Version: {torch.version.cuda}")
    else:
        print("❌ CUDA not available - using CPU")
    
    # メモリ使用量
    import psutil
    memory_info = psutil.virtual_memory()
    print(f"\n💾 システムメモリ:")
    print(f"• Total: {memory_info.total / 1024**3:.1f} GB")
    print(f"• Available: {memory_info.available / 1024**3:.1f} GB")
    print(f"• Used: {memory_info.percent:.1f}%")
    
    # プロセスメモリ
    process = psutil.Process()
    process_memory = process.memory_info()
    print(f"\n📊 プロセスメモリ:")
    print(f"• RSS: {process_memory.rss / 1024**2:.1f} MB")
    print(f"• VMS: {process_memory.vms / 1024**2:.1f} MB")

def main():
    """メイン関数"""
    print("🔬 Professional Statistical Analysis Software")
    print("🤖 Machine Learning & Deep Learning Test Suite")
    print("=" * 60)
    
    # 基本情報
    print(f"Python Version: {'.'.join(map(str, __import__('sys').version_info[:3]))}")
    print(f"NumPy Version: {np.__version__}")
    print(f"Pandas Version: {pd.__version__}")
    
    if TORCH_AVAILABLE:
        print(f"PyTorch Version: {torch.__version__}")
        print(f"CUDA Available: {torch.cuda.is_available()}")
    
    # テスト実行
    try:
        test_ml_algorithms()
        test_deep_learning()
        test_advanced_features()
        
        print(f"\n🎉 全てのテストが完了しました！")
        print("📱 統計ソフトウェアで「🤖 Machine Learning」ボタンを試してください！")
        
    except Exception as e:
        print(f"\n❌ テストエラー: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 