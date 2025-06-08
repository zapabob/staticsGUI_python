#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AI Integration Module
AI統合モジュール - OpenAI, Google AI Studio, 画像処理, 自然言語処理
"""

import asyncio
import base64
import re
import json
from typing import Optional, Dict, Any, List
from pathlib import Path
import traceback

# Data processing
import pandas as pd
import numpy as np

# AI API clients (オプション)
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import google.generativeai as genai
    GOOGLE_AI_AVAILABLE = True
except ImportError:
    GOOGLE_AI_AVAILABLE = False

# Image processing (オプション)
try:
    from PIL import Image
    import cv2
    import pytesseract
    import easyocr
    IMAGE_PROCESSING_AVAILABLE = True
except ImportError:
    IMAGE_PROCESSING_AVAILABLE = False

# Configuration
try:
    from config import ai_config, STATISTICAL_ANALYSIS_PROMPTS
except ImportError:
    # デフォルト設定
    class MockConfig:
        def __init__(self):
            self.openai_api_key = None
            self.google_api_key = None
            self.mcp_server_url = "ws://localhost:8080"
            self.mcp_enabled = False
            self.ocr_languages = ["eng", "jpn"]
            self.tesseract_cmd = None
        
        def is_api_configured(self, provider):
            return False
    
    ai_config = MockConfig()
    STATISTICAL_ANALYSIS_PROMPTS = {
        "natural_language_query": "分析要求: {user_query}",
        "image_data_extraction": "画像からデータを抽出してください。"
    }

class AIStatisticalAnalyzer:
    """AI統計解析エンジン"""
    
    def __init__(self):
        self.analysis_history = []
    
    async def analyze_natural_language_query(self, query: str, data: pd.DataFrame) -> Dict[str, Any]:
        """自然言語クエリで統計解析"""
        try:
            # AI API が利用可能な場合
            if OPENAI_AVAILABLE and ai_config.is_api_configured("openai"):
                return await self._analyze_with_openai(query, data)
            elif GOOGLE_AI_AVAILABLE and ai_config.is_api_configured("google"):
                return await self._analyze_with_google(query, data)
            else:
                # ローカル解析
                return self._analyze_locally(query, data)
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _analyze_with_openai(self, query: str, data: pd.DataFrame) -> Dict[str, Any]:
        """OpenAI APIで分析"""
        try:
            client = openai.OpenAI(api_key=ai_config.openai_api_key)
            
            prompt = f"""
データ情報:
- 形状: {data.shape}
- 列: {list(data.columns)}
- 型: {data.dtypes.to_dict()}

ユーザー要求: {query}

以下の形式でPythonコードを生成してください：
```python
# 実行可能なコード
```
"""
            
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "統計分析の専門家として、実行可能なPythonコードを生成してください。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            
            ai_response = response.choices[0].message.content
            code = self._extract_python_code(ai_response)
            
            return {
                "success": True,
                "ai_response": ai_response,
                "python_code": code,
                "provider": "OpenAI",
                "model": "gpt-4"
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _analyze_with_google(self, query: str, data: pd.DataFrame) -> Dict[str, Any]:
        """Google AI Studioで分析"""
        try:
            genai.configure(api_key=ai_config.google_api_key)
            model = genai.GenerativeModel('gemini-pro')
            
            prompt = f"""
データ情報:
- 形状: {data.shape}
- 列: {list(data.columns)}
- 型: {data.dtypes.to_dict()}

ユーザー要求: {query}

実行可能なPythonコードを生成してください。
"""
            
            response = model.generate_content(prompt)
            ai_response = response.text
            code = self._extract_python_code(ai_response)
            
            return {
                "success": True,
                "ai_response": ai_response,
                "python_code": code,
                "provider": "Google",
                "model": "gemini-pro"
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _analyze_locally(self, query: str, data: pd.DataFrame) -> Dict[str, Any]:
        """ローカル解析（ルールベース）"""
        try:
            code = self._generate_code_from_query(query, data)
            
            return {
                "success": True,
                "ai_response": f"クエリ「{query}」を解析しました。以下のコードを実行してください。",
                "python_code": code,
                "provider": "Local Analysis",
                "model": "Rule-based"
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _generate_code_from_query(self, query: str, data: pd.DataFrame) -> str:
        """クエリからPythonコード生成（ルールベース）"""
        query_lower = query.lower()
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if "相関" in query or "correlation" in query_lower:
            return f"""
# 相関分析
correlation_matrix = data[{numeric_cols}].corr()
print("相関行列:")
print(correlation_matrix)

import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()
"""
        
        elif "回帰" in query or "regression" in query_lower:
            if len(numeric_cols) >= 2:
                return f"""
# 線形回帰分析
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

X = data[['{numeric_cols[0]}']]
y = data['{numeric_cols[1]}']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)

print(f"R² Score: {{r2:.4f}}")
print(f"係数: {{model.coef_[0]:.4f}}")
print(f"切片: {{model.intercept_:.4f}}")
"""
        
        elif "記述統計" in query or "descriptive" in query_lower:
            return f"""
# 記述統計
print("データの形状:", data.shape)
print("\\n基本統計量:")
print(data.describe())

print("\\n欠損値:")
print(data.isnull().sum())

print("\\nデータ型:")
print(data.dtypes)
"""
        
        else:
            return f"""
# 基本的なデータ探索
print("データ概要:")
print(f"行数: {{len(data)}}")
print(f"列数: {{len(data.columns)}}")
print("\\n最初の5行:")
print(data.head())

print("\\n基本統計量:")
print(data.describe())
"""
    
    async def analyze_image_data(self, image_path: str, context: str = "") -> Dict[str, Any]:
        """画像からデータ抽出・分析"""
        try:
            # OCRでテキスト抽出（利用可能な場合）
            ocr_text = ""
            if IMAGE_PROCESSING_AVAILABLE:
                try:
                    img = Image.open(image_path)
                    ocr_text = pytesseract.image_to_string(img, lang='eng+jpn')
                except Exception as e:
                    print(f"OCRエラー: {e}")
            
            # AI APIで画像分析
            if OPENAI_AVAILABLE and ai_config.is_api_configured("openai"):
                return await self._analyze_image_with_openai(image_path, context, ocr_text)
            elif GOOGLE_AI_AVAILABLE and ai_config.is_api_configured("google"):
                return await self._analyze_image_with_google(image_path, context, ocr_text)
            else:
                return {
                    "success": True,
                    "ai_response": "画像分析にはAI APIの設定が必要です。",
                    "python_code": "",
                    "ocr_text": ocr_text,
                    "provider": "Local",
                    "model": "OCR-only"
                }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _analyze_image_with_openai(self, image_path: str, context: str, ocr_text: str) -> Dict[str, Any]:
        """OpenAI Vision APIで画像分析"""
        try:
            client = openai.OpenAI(api_key=ai_config.openai_api_key)
            
            # 画像をBase64エンコード
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
            
            prompt = "この画像に含まれるデータ（表、グラフ、チャートなど）を抽出し、Pandasで使用できるPythonコードを生成してください。"
            if context:
                prompt += f"\n追加コンテキスト: {context}"
            
            response = client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=4096
            )
            
            ai_response = response.choices[0].message.content
            code = self._extract_python_code(ai_response)
            
            return {
                "success": True,
                "ai_response": ai_response,
                "python_code": code,
                "ocr_text": ocr_text,
                "provider": "OpenAI",
                "model": "gpt-4-vision-preview"
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _analyze_image_with_google(self, image_path: str, context: str, ocr_text: str) -> Dict[str, Any]:
        """Google Gemini Vision APIで画像分析"""
        try:
            genai.configure(api_key=ai_config.google_api_key)
            model = genai.GenerativeModel('gemini-pro-vision')
            
            img = Image.open(image_path)
            prompt = "この画像に含まれるデータを抽出し、Pandasで使用できるPythonコードを生成してください。"
            if context:
                prompt += f"\n追加コンテキスト: {context}"
            
            response = model.generate_content([prompt, img])
            ai_response = response.text
            code = self._extract_python_code(ai_response)
            
            return {
                "success": True,
                "ai_response": ai_response,
                "python_code": code,
                "ocr_text": ocr_text,
                "provider": "Google",
                "model": "gemini-pro-vision"
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _extract_python_code(self, text: str) -> str:
        """テキストからPythonコードを抽出"""
        # ```python ... ``` ブロックを抽出
        pattern = r'```python\s*\n(.*?)\n```'
        matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
        
        if matches:
            return '\n'.join(matches)
        
        # ``` ... ``` ブロックを抽出（python指定なし）
        pattern = r'```\s*\n(.*?)\n```'
        matches = re.findall(pattern, text, re.DOTALL)
        
        if matches:
            return '\n'.join(matches)
        
        return ""
    
    def execute_generated_code(self, code: str, data: pd.DataFrame) -> Dict[str, Any]:
        """生成されたコードを安全に実行"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            from sklearn.linear_model import LinearRegression
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import r2_score
            
            # 安全な実行環境
            safe_globals = {
                "pd": pd,
                "np": np,
                "data": data,
                "df": data,
                "plt": plt,
                "sns": sns,
                "LinearRegression": LinearRegression,
                "train_test_split": train_test_split,
                "r2_score": r2_score,
                "__builtins__": {"len": len, "str": str, "int": int, "float": float, "print": print}
            }
            
            local_vars = {}
            exec(code, safe_globals, local_vars)
            
            return {
                "success": True,
                "result": local_vars,
                "output": "コード実行完了"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }

# グローバルインスタンス
ai_analyzer = AIStatisticalAnalyzer()
