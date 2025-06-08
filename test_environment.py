#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
環境テストスクリプト
"""
import sys

def test_environment():
    print("🔬 Professional Statistical Software - Environment Test")
    print("=" * 60)
    
    # Python環境
    print(f"Python Version: {sys.version}")
    print(f"Platform: {sys.platform}")
    
    # 必須ライブラリテスト
    required_libraries = [
        'numpy', 'pandas', 'scipy', 'matplotlib', 'seaborn',
        'sklearn', 'statsmodels', 'tqdm', 'psutil'
    ]
    
    print("\n📦 Required Libraries:")
    for lib in required_libraries:
        try:
            __import__(lib)
            print(f"✅ {lib}")
        except ImportError:
            print(f"❌ {lib} - NOT INSTALLED")
    
    # GPU/CUDA テスト
    print("\n⚡ GPU/CUDA Status:")
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        print(f"CUDA Available: {'✅ Yes' if torch.cuda.is_available() else '❌ No'}")
        if torch.cuda.is_available():
            print(f"GPU Count: {torch.cuda.device_count()}")
            print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("❌ PyTorch not installed")
    
    # GUI ライブラリテスト
    print("\n🖥️ GUI Libraries:")
    try:
        import tkinter
        print("✅ tkinter")
    except ImportError:
        print("❌ tkinter")
    
    try:
        import customtkinter
        print("✅ customtkinter")
    except ImportError:
        print("❌ customtkinter - Install with: pip install customtkinter")
    
    print("\n🎯 Environment Test Complete!")

if __name__ == "__main__":
    test_environment() 