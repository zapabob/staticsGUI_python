#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AI API Configuration Management
AI API設定管理
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any
import json
import yaml
from dotenv import load_dotenv

# .envファイルを読み込み
load_dotenv()

class AIConfig:
    """AI API設定管理クラス"""
    
    def __init__(self):
        self.config_dir = Path("config")
        self.config_dir.mkdir(exist_ok=True)
        
        # API設定
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        
        # MCP設定
        self.mcp_server_url = os.getenv("MCP_SERVER_URL", "ws://localhost:8080")
        self.mcp_enabled = os.getenv("MCP_ENABLED", "false").lower() == "true"
        
        # AI機能設定
        self.enable_vision = os.getenv("ENABLE_VISION", "true").lower() == "true"
        self.enable_nlp = os.getenv("ENABLE_NLP", "true").lower() == "true"
        self.default_model = os.getenv("DEFAULT_MODEL", "gpt-4-vision-preview")
        
        # 画像処理設定
        self.max_image_size = int(os.getenv("MAX_IMAGE_SIZE", "10485760"))  # 10MB
        self.supported_formats = ["jpg", "jpeg", "png", "gif", "bmp", "tiff", "webp"]
        
        # OCR設定
        self.tesseract_cmd = os.getenv("TESSERACT_CMD")
        self.ocr_languages = os.getenv("OCR_LANGUAGES", "eng+jpn").split("+")
        
        self._load_config()
    
    def _load_config(self):
        """設定ファイル読み込み"""
        config_file = self.config_dir / "ai_config.yaml"
        if config_file.exists():
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                    self._update_from_config(config)
            except Exception as e:
                print(f"設定ファイル読み込みエラー: {e}")
    
    def _update_from_config(self, config: Dict[str, Any]):
        """設定辞書から更新"""
        for key, value in config.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def save_config(self):
        """設定をファイルに保存"""
        config = {
            'mcp_server_url': self.mcp_server_url,
            'mcp_enabled': self.mcp_enabled,
            'enable_vision': self.enable_vision,
            'enable_nlp': self.enable_nlp,
            'default_model': self.default_model,
            'max_image_size': self.max_image_size,
            'supported_formats': self.supported_formats,
            'ocr_languages': self.ocr_languages
        }
        
        config_file = self.config_dir / "ai_config.yaml"
        try:
            with open(config_file, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        except Exception as e:
            print(f"設定保存エラー: {e}")
    
    def is_api_configured(self, provider: str) -> bool:
        """API設定確認"""
        if provider.lower() == "openai":
            return bool(self.openai_api_key)
        elif provider.lower() == "google":
            return bool(self.google_api_key)
        elif provider.lower() == "anthropic":
            return bool(self.anthropic_api_key)
        return False
    
    def get_available_providers(self) -> list:
        """利用可能なAPIプロバイダー一覧"""
        providers = []
        if self.is_api_configured("openai"):
            providers.append("OpenAI")
        if self.is_api_configured("google"):
            providers.append("Google")
        if self.is_api_configured("anthropic"):
            providers.append("Anthropic")
        return providers

# グローバル設定インスタンス
ai_config = AIConfig()

# プロンプトテンプレート
STATISTICAL_ANALYSIS_PROMPTS = {
    'data_description': """
以下のデータについて分析してください：

データ形状: {shape}
列名: {columns}
データ型: {dtypes}
欠損値: {missing_values}

以下の統計解析を実行し、結果をPythonコードとして返してください：
{analysis_request}

回答は以下の形式で：
1. 分析の説明
2. 実行するPythonコード
3. 結果の解釈
""",
    
    'image_data_extraction': """
この画像にはデータ（表、グラフ、チャートなど）が含まれています。
以下を抽出してPandasのDataFrameとして使用できる形式で返してください：

1. 表形式データがある場合：列名とデータを抽出
2. グラフがある場合：軸ラベル、データポイント、傾向を抽出
3. チャートがある場合：カテゴリとその値を抽出

抽出したデータをPythonコードとして返してください：
```python
import pandas as pd
import numpy as np

# 抽出されたデータ
data = {...}
df = pd.DataFrame(data)
```
""",
    
    'natural_language_query': """
ユーザーの自然言語による統計分析要求を解釈し、適切なPythonコードを生成してください。

データ情報:
- 形状: {shape}
- 列: {columns}
- 型: {dtypes}

ユーザー要求: {user_query}

以下の形式で回答してください：
1. 要求の解釈
2. 実行するPythonコード（pandas/scipyを使用）
3. 期待される結果の説明

コードは即座に実行可能な形式で記述してください。
"""
}

# モデル設定
MODEL_CONFIGS = {
    'openai': {
        'gpt-4-vision-preview': {
            'supports_vision': True,
            'max_tokens': 4096,
            'temperature': 0.1
        },
        'gpt-4': {
            'supports_vision': False,
            'max_tokens': 8192,
            'temperature': 0.1
        },
        'gpt-3.5-turbo': {
            'supports_vision': False,
            'max_tokens': 4096,
            'temperature': 0.1
        }
    },
    'google': {
        'gemini-pro-vision': {
            'supports_vision': True,
            'max_tokens': 2048,
            'temperature': 0.1
        },
        'gemini-pro': {
            'supports_vision': False,
            'max_tokens': 2048,
            'temperature': 0.1
        }
    }
} 