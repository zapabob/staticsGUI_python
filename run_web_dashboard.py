#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Web Dashboard Launcher
Webダッシュボード起動スクリプト

Mac M2 Optimized
Author: Ryo Minegishi
License: MIT
"""

import subprocess
import sys
import os
import platform
from pathlib import Path

def check_dependencies():
    """依存関係チェック"""
    
    required_packages = [
        'streamlit',
        'plotly',
        'pandas',
        'numpy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ 必要なパッケージが不足しています: {', '.join(missing_packages)}")
        print("📦 以下のコマンドでインストールしてください:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

def get_optimal_settings():
    """Mac M2最適化設定取得"""
    
    settings = {}
    
    # プラットフォーム検出
    machine = platform.machine()
    system = platform.system()
    
    if machine == 'arm64' and system == 'Darwin':
        print("🍎 Mac M2/M3 detected - Apple Silicon optimization enabled")
        settings['browser'] = 'safari'
        settings['threads'] = os.cpu_count()
        
        # M2最適化環境変数
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'
        os.environ['OPENBLAS_NUM_THREADS'] = '1'
        
    elif system == 'Windows':
        print("🪟 Windows detected")
        settings['browser'] = 'chrome'
        settings['threads'] = os.cpu_count()
        
    else:
        print("🐧 Linux/Other detected")
        settings['browser'] = 'firefox'
        settings['threads'] = os.cpu_count()
    
    return settings

def run_web_dashboard():
    """Webダッシュボード実行"""
    
    print("🚀 HAD Professional Statistical Analysis Web Dashboard")
    print("=" * 60)
    
    # 依存関係チェック
    if not check_dependencies():
        return False
    
    # 最適化設定
    settings = get_optimal_settings()
    
    # ダッシュボードファイルの存在確認
    dashboard_file = Path("web_dashboard.py")
    if not dashboard_file.exists():
        print("❌ web_dashboard.py が見つかりません")
        return False
    
    print(f"📊 Web Dashboard を起動中...")
    print(f"🌐 ブラウザで自動的に開きます")
    print(f"⚡ 最適化設定: {settings['threads']} threads")
    print("=" * 60)
    
    # Streamlit実行
    try:
        cmd = [
            sys.executable, "-m", "streamlit", "run", 
            str(dashboard_file),
            "--server.address", "localhost",
            "--server.port", "8501",
            "--browser.gatherUsageStats", "false",
            "--server.headless", "false"
        ]
        
        # Mac M2の場合はSafariを優先
        if settings.get('browser') == 'safari':
            cmd.extend(["--server.browserName", "safari"])
        
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        print("\n👋 Webダッシュボードを終了しました")
        return True
        
    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")
        return False

def main():
    """メイン関数"""
    
    print("🔬 HAD Professional Statistical Analysis")
    print("📊 Multilingual Web Dashboard Launcher")
    print("🍎 Mac M2 Optimized Version")
    print()
    
    success = run_web_dashboard()
    
    if success:
        print("\n✅ 正常に終了しました")
    else:
        print("\n❌ エラーで終了しました")
        sys.exit(1)

if __name__ == "__main__":
    main() 