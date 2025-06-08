#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Professional Reports Generator
プロフェッショナルレポート生成モジュール

Author: Ryo Minegishi
Email: r.minegishi1987@gmail.com
License: MIT
"""

import os
import json
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from jinja2 import Template

class ReportGenerator:
    """プロフェッショナルレポート生成クラス"""
    
    def __init__(self):
        self.report_dir = Path("reports")
        self.report_dir.mkdir(exist_ok=True)
        
        # 日本語フォント設定
        plt.rcParams['font.family'] = ['DejaVu Sans', 'Hiragino Sans', 'Yu Gothic', 'Meiryo', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']
    
    def generate_comprehensive_report(self, data, analysis_results=None, title="Statistical Analysis Report", subtitle=""):
        """包括的レポート生成"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.report_dir / f"report_{timestamp}.html"
        
        # HTMLテンプレート
        html_template = """
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
            border-bottom: 3px solid #2196F3;
            padding-bottom: 20px;
        }
        .header h1 {
            color: #1976D2;
            margin: 0;
            font-size: 2.5em;
        }
        .header p {
            color: #666;
            margin: 10px 0 0 0;
            font-size: 1.1em;
        }
        .section {
            margin: 30px 0;
            padding: 20px;
            background: #fafafa;
            border-left: 4px solid #2196F3;
            border-radius: 5px;
        }
        .section h2 {
            color: #1976D2;
            margin-top: 0;
            display: flex;
            align-items: center;
        }
        .section h2::before {
            content: "📊";
            margin-right: 10px;
            font-size: 1.2em;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        .stat-card {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            text-align: center;
        }
        .stat-card h3 {
            margin: 0 0 10px 0;
            color: #1976D2;
            font-size: 1.8em;
        }
        .stat-card p {
            margin: 0;
            color: #666;
            font-weight: bold;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: white;
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background: #2196F3;
            color: white;
            font-weight: bold;
        }
        tr:hover {
            background-color: #f5f5f5;
        }
        .footer {
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 2px solid #eee;
            color: #888;
        }
        .alert {
            padding: 15px;
            margin: 20px 0;
            border-radius: 5px;
            border-left: 4px solid #ff9800;
            background: #fff3e0;
        }
        .success {
            border-left-color: #4caf50;
            background: #e8f5e8;
        }
        .warning {
            border-left-color: #ff9800;
            background: #fff3e0;
        }
        .error {
            border-left-color: #f44336;
            background: #ffebee;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{{ title }}</h1>
            <p>{{ subtitle }}</p>
            <p>生成日時: {{ generation_time }}</p>
        </div>
        
        <div class="section">
            <h2>データ概要</h2>
            <div class="stats-grid">
                <div class="stat-card">
                    <h3>{{ data_rows }}</h3>
                    <p>総行数</p>
                </div>
                <div class="stat-card">
                    <h3>{{ data_cols }}</h3>
                    <p>総列数</p>
                </div>
                <div class="stat-card">
                    <h3>{{ numeric_cols }}</h3>
                    <p>数値列数</p>
                </div>
                <div class="stat-card">
                    <h3>{{ missing_values }}</h3>
                    <p>欠損値数</p>
                </div>
            </div>
            
            <h3>📋 列情報</h3>
            <table>
                <thead>
                    <tr>
                        <th>列名</th>
                        <th>データ型</th>
                        <th>欠損値数</th>
                        <th>欠損率</th>
                        <th>ユニーク値数</th>
                    </tr>
                </thead>
                <tbody>
                    {% for col_info in column_info %}
                    <tr>
                        <td>{{ col_info.name }}</td>
                        <td>{{ col_info.dtype }}</td>
                        <td>{{ col_info.missing }}</td>
                        <td>{{ col_info.missing_pct }}%</td>
                        <td>{{ col_info.unique }}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
        
        {% if has_numeric_data %}
        <div class="section">
            <h2>基本統計量</h2>
            <table>
                <thead>
                    <tr>
                        <th>統計量</th>
                        {% for col in numeric_columns %}
                        <th>{{ col }}</th>
                        {% endfor %}
                    </tr>
                </thead>
                <tbody>
                    {% for stat_name, stat_values in basic_stats.items() %}
                    <tr>
                        <td><strong>{{ stat_name }}</strong></td>
                        {% for value in stat_values %}
                        <td>{{ "%.4f"|format(value) }}</td>
                        {% endfor %}
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
        {% endif %}
        
        {% if analysis_results %}
        <div class="section">
            <h2>解析結果</h2>
            <div class="alert success">
                <strong>✅ 解析完了:</strong> 実行された解析の結果をまとめています。
            </div>
            
            {% for analysis_name, result in analysis_results.items() %}
            <h3>🔬 {{ analysis_name }}</h3>
            <pre style="background: #f8f8f8; padding: 15px; border-radius: 5px; overflow-x: auto;">{{ result }}</pre>
            {% endfor %}
        </div>
        {% endif %}
        
        <div class="section">
            <h2>データ品質レポート</h2>
            {% if quality_issues %}
            <div class="alert warning">
                <strong>⚠️ 品質上の注意点:</strong> 以下の点にご注意ください。
            </div>
            <ul>
                {% for issue in quality_issues %}
                <li>{{ issue }}</li>
                {% endfor %}
            </ul>
            {% else %}
            <div class="alert success">
                <strong>✅ データ品質:</strong> 特に問題は検出されませんでした。
            </div>
            {% endif %}
        </div>
        
        <div class="footer">
            <p>🔬 HAD Professional Statistical Analysis Software</p>
            <p>Generated by AI-Powered Statistical Analysis Engine</p>
            <p>© 2024 Ryo Minegishi. All rights reserved.</p>
        </div>
    </div>
</body>
</html>
        """
        
        # データ解析
        data_rows = len(data)
        data_cols = len(data.columns)
        numeric_cols = len(data.select_dtypes(include=[np.number]).columns)
        missing_values = data.isnull().sum().sum()
        
        # 列情報
        column_info = []
        for col in data.columns:
            missing_count = data[col].isnull().sum()
            missing_pct = round((missing_count / len(data)) * 100, 2)
            unique_count = data[col].nunique()
            
            column_info.append({
                'name': col,
                'dtype': str(data[col].dtype),
                'missing': missing_count,
                'missing_pct': missing_pct,
                'unique': unique_count
            })
        
        # 基本統計量（数値列のみ）
        numeric_data = data.select_dtypes(include=[np.number])
        has_numeric_data = len(numeric_data.columns) > 0
        
        basic_stats = {}
        numeric_columns = []
        
        if has_numeric_data:
            numeric_columns = list(numeric_data.columns)
            desc_stats = numeric_data.describe()
            
            basic_stats = {
                '平均': desc_stats.loc['mean'].values,
                '標準偏差': desc_stats.loc['std'].values,
                '最小値': desc_stats.loc['min'].values,
                '25%': desc_stats.loc['25%'].values,
                '中央値': desc_stats.loc['50%'].values,
                '75%': desc_stats.loc['75%'].values,
                '最大値': desc_stats.loc['max'].values
            }
        
        # データ品質の問題を検出
        quality_issues = []
        
        # 高い欠損率のチェック
        high_missing = data.isnull().sum() / len(data) > 0.3
        if high_missing.any():
            high_missing_cols = data.columns[high_missing].tolist()
            quality_issues.append(f"高い欠損率（>30%）の列: {', '.join(high_missing_cols)}")
        
        # 重複行のチェック
        duplicate_count = data.duplicated().sum()
        if duplicate_count > 0:
            quality_issues.append(f"重複行が {duplicate_count} 行検出されました")
        
        # 一意値が少ない数値列のチェック
        for col in numeric_data.columns:
            unique_ratio = data[col].nunique() / len(data)
            if unique_ratio < 0.01 and data[col].nunique() > 1:
                quality_issues.append(f"'{col}' 列の一意値比率が非常に低い（{unique_ratio:.3f}）")
        
        # テンプレート描画
        template = Template(html_template)
        html_content = template.render(
            title=title,
            subtitle=subtitle,
            generation_time=datetime.now().strftime("%Y年%m月%d日 %H:%M:%S"),
            data_rows=f"{data_rows:,}",
            data_cols=data_cols,
            numeric_cols=numeric_cols,
            missing_values=f"{missing_values:,}",
            column_info=column_info,
            has_numeric_data=has_numeric_data,
            numeric_columns=numeric_columns,
            basic_stats=basic_stats,
            analysis_results=analysis_results or {},
            quality_issues=quality_issues
        )
        
        # ファイル保存
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return str(report_file)

# グローバルインスタンス
report_generator = ReportGenerator()
