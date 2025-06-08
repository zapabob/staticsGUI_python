#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Professional Report Generation System
プロフェッショナルレポート生成システム

Author: Ryo Minegishi
License: MIT
"""

import os
import sys
import json
import base64
from io import BytesIO
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from jinja2 import Template, Environment, FileSystemLoader
import matplotlib
matplotlib.use('Agg')  # GUIなし環境対応

class ProfessionalReportGenerator:
    """プロフェッショナルレポート生成システム"""
    
    def __init__(self, template_dir: str = "templates", output_dir: str = "reports"):
        self.template_dir = Path(template_dir)
        self.output_dir = Path(output_dir)
        
        # ディレクトリ作成
        self.template_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        
        # Jinja2環境設定
        self.jinja_env = Environment(
            loader=FileSystemLoader(str(self.template_dir)) if self.template_dir.exists() else None,
            autoescape=True
        )
        
        # テンプレート作成
        self._create_default_templates()
        
        # レポートスタイル設定
        self._setup_styles()
    
    def _setup_styles(self):
        """レポートスタイル設定"""
        # matplotlib日本語フォント設定
        plt.rcParams['font.family'] = ['DejaVu Sans', 'Yu Gothic', 'Hiragino Sans', 'Noto Sans CJK JP']
        plt.rcParams['axes.unicode_minus'] = False
        
        # カラーパレット
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72',
            'success': '#F18F01',
            'warning': '#C73E1D',
            'info': '#16537e',
            'light': '#f8f9fa',
            'dark': '#343a40'
        }
    
    def _create_default_templates(self):
        """デフォルトテンプレート作成"""
        
        # HTMLレポートテンプレート
        html_template = """<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ report_title }}</title>
    <style>
        body {
            font-family: 'Yu Gothic', 'Hiragino Sans', sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f8f9fa;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .header {
            text-align: center;
            border-bottom: 3px solid #2E86AB;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }
        .header h1 {
            color: #2E86AB;
            margin: 0;
            font-size: 2.5em;
        }
        .section {
            margin-bottom: 40px;
        }
        .section h2 {
            color: #343a40;
            border-bottom: 2px solid #A23B72;
            padding-bottom: 10px;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        .stat-card {
            background: linear-gradient(135deg, #2E86AB, #A23B72);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }
        .stat-value {
            font-size: 2em;
            font-weight: bold;
            margin: 10px 0;
        }
        .chart-container {
            text-align: center;
            margin: 20px 0;
        }
        .chart-container img {
            max-width: 100%;
            height: auto;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        table {
            width: 100%;
            border-collapse: collapse;
            background-color: white;
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #2E86AB;
            color: white;
        }
        tr:nth-child(even) {
            background-color: #f2f2f2;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{{ report_title }}</h1>
            <p>{{ report_subtitle }}</p>
        </div>
        
        {% for section in sections %}
        <div class="section">
            <h2>{{ section.title }}</h2>
            
            {% if section.stats %}
            <div class="stats-grid">
                {% for stat in section.stats %}
                <div class="stat-card">
                    <div class="stat-label">{{ stat.label }}</div>
                    <div class="stat-value">{{ stat.value }}</div>
                    <div class="stat-unit">{{ stat.unit }}</div>
                </div>
                {% endfor %}
            </div>
            {% endif %}
            
            {% if section.content %}
            <div class="content">
                {{ section.content | safe }}
            </div>
            {% endif %}
            
            {% if section.charts %}
            {% for chart in section.charts %}
            <div class="chart-container">
                <h3>{{ chart.title }}</h3>
                <img src="data:image/png;base64,{{ chart.image }}" alt="{{ chart.title }}">
                {% if chart.description %}
                <p>{{ chart.description }}</p>
                {% endif %}
            </div>
            {% endfor %}
            {% endif %}
            
            {% if section.tables %}
            {% for table in section.tables %}
            <h3>{{ table.title }}</h3>
            {{ table.html | safe }}
            {% endfor %}
            {% endif %}
        </div>
        {% endfor %}
        
        <div class="footer">
            <p>HAD Professional Statistical Analysis Software</p>
        </div>
    </div>
</body>
</html>"""
        
        template_file = self.template_dir / "report_template.html"
        with open(template_file, "w", encoding="utf-8") as f:
            f.write(html_template)
    
    def generate_comprehensive_report(self, data: pd.DataFrame, analysis_results: Dict = None,
                                    title: str = "統計解析レポート",
                                    subtitle: str = "Professional Statistical Analysis Report") -> str:
        """包括的統計解析レポート生成"""
        
        if analysis_results is None:
            analysis_results = {}
        
        report_data = {
            'report_title': title,
            'report_subtitle': subtitle,
            'generation_time': datetime.now().strftime('%Y年%m月%d日 %H:%M:%S'),
            'sections': []
        }
        
        # 1. データ概要セクション
        overview_section = self._create_overview_section(data)
        report_data['sections'].append(overview_section)
        
        # 2. 記述統計セクション
        descriptive_section = self._create_descriptive_section(data)
        report_data['sections'].append(descriptive_section)
        
        # 3. データ品質セクション
        quality_section = self._create_quality_section(data)
        report_data['sections'].append(quality_section)
        
        # 4. 可視化セクション
        visualization_section = self._create_visualization_section(data)
        report_data['sections'].append(visualization_section)
        
        # 5. 解析結果セクション
        if analysis_results:
            analysis_section = self._create_analysis_section(analysis_results)
            report_data['sections'].append(analysis_section)
        
        # HTMLレポート生成
        html_content = self._render_template(report_data)
        
        # ファイル保存
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = self.output_dir / f"statistical_report_{timestamp}.html"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return str(output_file)
    
    def _render_template(self, data: Dict) -> str:
        """テンプレートレンダリング"""
        template_file = self.template_dir / "report_template.html"
        
        if template_file.exists():
            try:
                template = self.jinja_env.get_template('report_template.html')
                return template.render(**data)
            except Exception as e:
                print(f"テンプレートエラー: {e}")
        
        # フォールバック：簡単なHTMLレンダリング
        return self._simple_html_render(data)
    
    def _simple_html_render(self, data: Dict) -> str:
        """シンプルHTMLレンダリング"""
        html = f"""<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <title>{data['report_title']}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .section {{ margin-bottom: 30px; }}
        h1 {{ color: #2E86AB; }}
        h2 {{ color: #A23B72; border-bottom: 2px solid #A23B72; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #2E86AB; color: white; }}
    </style>
</head>
<body>
    <h1>{data['report_title']}</h1>
    <p><strong>生成日時:</strong> {data['generation_time']}</p>
"""
        
        for section in data['sections']:
            html += f"<div class='section'><h2>{section['title']}</h2>"
            
            if 'content' in section:
                html += section['content']
            
            if 'tables' in section:
                for table in section['tables']:
                    html += f"<h3>{table['title']}</h3>"
                    html += table['html']
            
            html += "</div>"
        
        html += "</body></html>"
        return html
    
    def _create_overview_section(self, data: pd.DataFrame) -> Dict:
        """データ概要セクション作成"""
        
        # 基本統計
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns
        
        stats = [
            {'label': '総レコード数', 'value': f"{data.shape[0]:,}", 'unit': '行'},
            {'label': '総変数数', 'value': f"{data.shape[1]:,}", 'unit': '列'},
            {'label': '数値変数', 'value': len(numeric_cols), 'unit': '個'},
            {'label': 'カテゴリ変数', 'value': len(categorical_cols), 'unit': '個'},
            {'label': '欠損値', 'value': f"{data.isnull().sum().sum():,}", 'unit': '個'}
        ]
        
        # データ型情報テーブル
        dtype_info = pd.DataFrame({
            '変数名': data.columns,
            'データ型': data.dtypes.astype(str),
            '欠損値数': data.isnull().sum().values,
            '欠損率': (data.isnull().sum() / len(data) * 100).round(2).astype(str) + '%',
            'ユニーク値数': [data[col].nunique() for col in data.columns]
        })
        
        return {
            'title': '📊 データ概要',
            'stats': stats,
            'tables': [{
                'title': 'データ型情報',
                'html': dtype_info.to_html(classes='table', index=False, escape=False)
            }]
        }
    
    def _create_descriptive_section(self, data: pd.DataFrame) -> Dict:
        """記述統計セクション作成"""
        
        numeric_data = data.select_dtypes(include=[np.number])
        
        if numeric_data.empty:
            return {
                'title': '📈 記述統計',
                'content': '<p>数値データが見つかりませんでした。</p>'
            }
        
        # 記述統計表
        desc_stats = numeric_data.describe().round(3)
        
        # 歪度・尖度追加
        try:
            desc_stats.loc['歪度'] = numeric_data.skew().round(3)
            desc_stats.loc['尖度'] = numeric_data.kurtosis().round(3)
        except:
            pass
        
        # 相関行列
        try:
            correlation_matrix = numeric_data.corr().round(3)
        except:
            correlation_matrix = pd.DataFrame()
        
        tables = [{
            'title': '基本統計量',
            'html': desc_stats.to_html(classes='table', escape=False)
        }]
        
        if not correlation_matrix.empty:
            tables.append({
                'title': '相関行列',
                'html': correlation_matrix.to_html(classes='table', escape=False)
            })
        
        return {
            'title': '📈 記述統計',
            'tables': tables
        }
    
    def _create_quality_section(self, data: pd.DataFrame) -> Dict:
        """データ品質セクション作成"""
        
        # 欠損値分析
        missing_data = data.isnull().sum()
        missing_data = missing_data[missing_data > 0]
        
        # 重複レコード分析
        duplicates = data.duplicated().sum()
        
        quality_stats = [
            {'label': '欠損値のある変数', 'value': len(missing_data), 'unit': '個'},
            {'label': '重複レコード', 'value': duplicates, 'unit': '行'}
        ]
        
        tables = []
        
        if not missing_data.empty:
            missing_df = pd.DataFrame({
                '変数名': missing_data.index,
                '欠損数': missing_data.values,
                '欠損率': (missing_data / len(data) * 100).round(2).astype(str) + '%'
            })
            tables.append({
                'title': '欠損値情報',
                'html': missing_df.to_html(classes='table', index=False, escape=False)
            })
        
        return {
            'title': '🔍 データ品質分析',
            'stats': quality_stats,
            'tables': tables
        }
    
    def _create_visualization_section(self, data: pd.DataFrame) -> Dict:
        """可視化セクション作成"""
        
        charts = []
        
        try:
            # 数値データのヒストグラム
            numeric_data = data.select_dtypes(include=[np.number])
            
            if not numeric_data.empty and len(numeric_data.columns) > 0:
                # 分布図
                fig, axes = plt.subplots(2, 2, figsize=(15, 12))
                axes = axes.ravel()
                
                for i, col in enumerate(numeric_data.columns[:4]):
                    if i < 4:
                        axes[i].hist(numeric_data[col].dropna(), bins=30, alpha=0.7, color=self.colors['primary'])
                        axes[i].set_title(f'{col} Distribution')
                        axes[i].set_xlabel(col)
                        axes[i].set_ylabel('Frequency')
                        axes[i].grid(True, alpha=0.3)
                
                # 余った軸は非表示
                for i in range(len(numeric_data.columns), 4):
                    axes[i].set_visible(False)
                
                plt.tight_layout()
                
                # 画像をBase64に変換
                buffer = BytesIO()
                plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
                buffer.seek(0)
                image_base64 = base64.b64encode(buffer.getvalue()).decode()
                plt.close()
                
                charts.append({
                    'title': '数値変数の分布',
                    'image': image_base64,
                    'description': '主要な数値変数のヒストグラム分布を表示'
                })
                
                # 相関ヒートマップ
                if len(numeric_data.columns) > 1:
                    plt.figure(figsize=(10, 8))
                    correlation_matrix = numeric_data.corr()
                    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                               square=True)
                    plt.title('Correlation Heatmap')
                    plt.tight_layout()
                    
                    buffer = BytesIO()
                    plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
                    buffer.seek(0)
                    image_base64 = base64.b64encode(buffer.getvalue()).decode()
                    plt.close()
                    
                    charts.append({
                        'title': '変数間相関ヒートマップ',
                        'image': image_base64,
                        'description': '数値変数間の相関関係を色で表現'
                    })
        except Exception as e:
            print(f"可視化エラー: {e}")
        
        return {
            'title': '📊 データ可視化',
            'charts': charts
        }
    
    def _create_analysis_section(self, analysis_results: Dict) -> Dict:
        """解析結果セクション作成"""
        
        content = '<div class="analysis-results">'
        
        for analysis_type, results in analysis_results.items():
            content += f'<h3>{analysis_type}</h3>'
            
            if isinstance(results, dict):
                content += '<ul>'
                for key, value in results.items():
                    if isinstance(value, (int, float)):
                        content += f'<li><strong>{key}:</strong> {value:.4f}</li>'
                    else:
                        content += f'<li><strong>{key}:</strong> {value}</li>'
                content += '</ul>'
            else:
                content += f'<p>{results}</p>'
        
        content += '</div>'
        
        return {
            'title': '🧮 解析結果',
            'content': content
        }

# グローバルインスタンス
report_generator = ProfessionalReportGenerator()
