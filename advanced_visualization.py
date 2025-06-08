#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Advanced Visualization Module for Professional Statistical Analysis
高度可視化モジュール

Author: Ryo Minegishi
Email: r.minegishi1987@gmail.com
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.style as mpl_style
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 日本語フォント設定
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial Unicode MS', 'Hiragino Sans', 'Yu Gothic', 'Meiryo', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']

class AdvancedVisualizer:
    """高度可視化クラス"""
    
    def __init__(self):
        """初期化"""
        # カラーパレット設定
        self.color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                             '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
        # スタイル設定
        plt.style.use('seaborn-v0_8')
        sns.set_palette(self.color_palette)
    
    def correlation_heatmap(self, data: pd.DataFrame, method: str = 'pearson', 
                           figsize: tuple = (12, 10), annot: bool = True) -> tuple:
        """相関行列ヒートマップ作成"""
        try:
            # 数値データのみ抽出
            numeric_data = data.select_dtypes(include=[np.number])
            
            if numeric_data.empty:
                return None, "数値データが見つかりません"
            
            if len(numeric_data.columns) < 2:
                return None, "相関計算には2つ以上の数値列が必要です"
            
            # 相関行列計算
            correlation_matrix = numeric_data.corr(method=method)
            
            # プロット作成
            fig, ax = plt.subplots(figsize=figsize)
            
            # ヒートマップ作成
            mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
            heatmap = sns.heatmap(correlation_matrix, 
                                mask=mask,
                                annot=annot, 
                                cmap='RdBu_r', 
                                center=0,
                                square=True, 
                                fmt='.2f',
                                cbar_kws={"shrink": .8},
                                ax=ax)
            
            # タイトルと設定
            ax.set_title(f'相関行列ヒートマップ ({method.capitalize()})', fontsize=16, pad=20)
            ax.set_xlabel('')
            ax.set_ylabel('')
            
            # 軸ラベルの調整
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            
            return fig, f"相関行列ヒートマップを作成しました ({method}法)"
            
        except Exception as e:
            return None, f"ヒートマップ作成エラー: {str(e)}"
    
    def scatter_matrix(self, data: pd.DataFrame, target_col: str = None, 
                      sample_size: int = 1000) -> tuple:
        """散布図マトリックス作成"""
        try:
            # 数値データのみ抽出
            numeric_data = data.select_dtypes(include=[np.number])
            
            if numeric_data.empty:
                return None, "数値データが見つかりません"
            
            # サンプリング（データが大きい場合）
            if len(numeric_data) > sample_size:
                numeric_data = numeric_data.sample(n=sample_size, random_state=42)
            
            # 変数が多すぎる場合は最初の8個まで
            if len(numeric_data.columns) > 8:
                numeric_data = numeric_data.iloc[:, :8]
            
            # Plotlyで散布図マトリックス作成
            if target_col and target_col in data.columns:
                # ターゲット列がある場合、色分け
                target_data = data.loc[numeric_data.index, target_col]
                
                if pd.api.types.is_numeric_dtype(target_data):
                    # 数値ターゲット
                    fig = px.scatter_matrix(numeric_data, 
                                          color=target_data,
                                          title=f"散布図マトリックス (色: {target_col})",
                                          color_continuous_scale='Viridis')
                else:
                    # カテゴリターゲット
                    fig = px.scatter_matrix(numeric_data, 
                                          color=target_data,
                                          title=f"散布図マトリックス (色: {target_col})")
            else:
                # ターゲット列なし
                fig = px.scatter_matrix(numeric_data, 
                                      title="散布図マトリックス")
            
            # レイアウト調整
            fig.update_layout(
                width=800,
                height=800,
                title_x=0.5
            )
            
            return fig, "散布図マトリックスを作成しました"
            
        except Exception as e:
            return None, f"散布図マトリックス作成エラー: {str(e)}"
    
    def distribution_plots(self, data: pd.DataFrame, max_cols: int = 6) -> tuple:
        """分布プロット作成"""
        try:
            # 数値データのみ抽出
            numeric_data = data.select_dtypes(include=[np.number])
            
            if numeric_data.empty:
                return None, "数値データが見つかりません"
            
            # 列数制限
            columns = numeric_data.columns[:max_cols]
            n_cols = len(columns)
            
            # サブプロット設定
            n_rows = (n_cols + 2) // 3  # 3列で配置
            fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5 * n_rows))
            
            if n_rows == 1:
                axes = axes.reshape(1, -1)
            
            # 各変数の分布プロット
            for i, col in enumerate(columns):
                row = i // 3
                col_idx = i % 3
                ax = axes[row, col_idx]
                
                # ヒストグラムと密度プロット
                data_col = numeric_data[col].dropna()
                
                # ヒストグラム
                ax.hist(data_col, bins=30, alpha=0.7, density=True, 
                       color=self.color_palette[i % len(self.color_palette)])
                
                # 密度プロット（KDE）
                try:
                    sns.kdeplot(data=data_col, ax=ax, color='red', linewidth=2)
                except:
                    pass
                
                # 正規分布との比較
                try:
                    mu, sigma = stats.norm.fit(data_col)
                    x = np.linspace(data_col.min(), data_col.max(), 100)
                    normal_dist = stats.norm.pdf(x, mu, sigma)
                    ax.plot(x, normal_dist, 'k--', linewidth=2, alpha=0.7, label='Normal fit')
                    ax.legend()
                except:
                    pass
                
                ax.set_title(f'{col} の分布', fontsize=12)
                ax.set_xlabel(col)
                ax.set_ylabel('密度')
                ax.grid(True, alpha=0.3)
            
            # 空のサブプロットを非表示
            for i in range(n_cols, n_rows * 3):
                row = i // 3
                col_idx = i % 3
                axes[row, col_idx].set_visible(False)
            
            plt.suptitle('変数分布プロット', fontsize=16, y=1.02)
            plt.tight_layout()
            
            return fig, f"{n_cols}個の変数の分布プロットを作成しました"
            
        except Exception as e:
            return None, f"分布プロット作成エラー: {str(e)}"
    
    def boxplot_comparison(self, data: pd.DataFrame, categorical_col: str, 
                          numerical_col: str) -> tuple:
        """箱ひげ図比較プロット作成"""
        try:
            if categorical_col not in data.columns or numerical_col not in data.columns:
                return None, "指定された列が見つかりません"
            
            # データ準備
            plot_data = data[[categorical_col, numerical_col]].dropna()
            
            # プロット作成
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # 箱ひげ図
            sns.boxplot(data=plot_data, x=categorical_col, y=numerical_col, ax=ax1)
            ax1.set_title(f'{numerical_col} by {categorical_col} (Box Plot)', fontsize=12)
            ax1.tick_params(axis='x', rotation=45)
            
            # バイオリンプロット
            sns.violinplot(data=plot_data, x=categorical_col, y=numerical_col, ax=ax2)
            ax2.set_title(f'{numerical_col} by {categorical_col} (Violin Plot)', fontsize=12)
            ax2.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            return fig, f"{categorical_col}別の{numerical_col}の分布比較を作成しました"
            
        except Exception as e:
            return None, f"箱ひげ図作成エラー: {str(e)}"
    
    def time_series_plot(self, data: pd.DataFrame, date_col: str, 
                        value_col: str, rolling_window: int = 30) -> tuple:
        """時系列プロット作成"""
        try:
            if date_col not in data.columns or value_col not in data.columns:
                return None, "指定された列が見つかりません"
            
            # データ準備
            plot_data = data[[date_col, value_col]].copy()
            plot_data = plot_data.dropna()
            
            # 日付列の変換
            try:
                plot_data[date_col] = pd.to_datetime(plot_data[date_col])
            except:
                return None, "日付列を日付型に変換できません"
            
            # ソート
            plot_data = plot_data.sort_values(date_col)
            
            # 移動平均計算
            plot_data[f'{value_col}_rolling'] = plot_data[value_col].rolling(window=rolling_window).mean()
            
            # Plotlyで時系列プロット作成
            fig = go.Figure()
            
            # 元データ
            fig.add_trace(go.Scatter(
                x=plot_data[date_col],
                y=plot_data[value_col],
                mode='lines',
                name='Original',
                line=dict(color='blue', width=1),
                opacity=0.7
            ))
            
            # 移動平均
            fig.add_trace(go.Scatter(
                x=plot_data[date_col],
                y=plot_data[f'{value_col}_rolling'],
                mode='lines',
                name=f'{rolling_window}日移動平均',
                line=dict(color='red', width=2)
            ))
            
            # レイアウト設定
            fig.update_layout(
                title=f'{value_col} の時系列プロット',
                xaxis_title=date_col,
                yaxis_title=value_col,
                width=1000,
                height=500,
                hovermode='x unified'
            )
            
            return fig, f"{value_col}の時系列プロットを作成しました"
            
        except Exception as e:
            return None, f"時系列プロット作成エラー: {str(e)}"
    
    def feature_importance_plot(self, importance_dict: dict, top_n: int = 20) -> tuple:
        """特徴量重要度プロット作成"""
        try:
            if not importance_dict:
                return None, "特徴量重要度データがありません"
            
            # データ準備
            features = list(importance_dict.keys())
            importances = list(importance_dict.values())
            
            # 重要度でソート
            sorted_indices = np.argsort(importances)[::-1]
            sorted_features = [features[i] for i in sorted_indices[:top_n]]
            sorted_importances = [importances[i] for i in sorted_indices[:top_n]]
            
            # プロット作成
            fig, ax = plt.subplots(figsize=(10, max(6, len(sorted_features) * 0.4)))
            
            # 横棒グラフ
            bars = ax.barh(range(len(sorted_features)), sorted_importances, 
                          color=self.color_palette[0], alpha=0.8)
            
            # ラベル設定
            ax.set_yticks(range(len(sorted_features)))
            ax.set_yticklabels(sorted_features)
            ax.set_xlabel('重要度')
            ax.set_title(f'特徴量重要度 (Top {len(sorted_features)})', fontsize=14)
            
            # 値をバーに表示
            for i, (bar, importance) in enumerate(zip(bars, sorted_importances)):
                ax.text(importance + max(sorted_importances) * 0.01, 
                       bar.get_y() + bar.get_height()/2, 
                       f'{importance:.3f}', 
                       va='center', fontsize=9)
            
            # グリッド
            ax.grid(True, axis='x', alpha=0.3)
            
            # レイアウト調整
            plt.gca().invert_yaxis()  # 重要度順（降順）
            plt.tight_layout()
            
            return fig, f"特徴量重要度プロット（上位{len(sorted_features)}個）を作成しました"
            
        except Exception as e:
            return None, f"特徴量重要度プロット作成エラー: {str(e)}"
    
    def qq_plot(self, data: pd.DataFrame, column: str) -> tuple:
        """Q-Qプロット作成"""
        try:
            if column not in data.columns:
                return None, "指定された列が見つかりません"
            
            # データ準備
            data_col = data[column].dropna()
            
            if len(data_col) == 0:
                return None, "データが空です"
            
            # Q-Qプロット作成
            fig, ax = plt.subplots(figsize=(8, 6))
            stats.probplot(data_col, dist="norm", plot=ax)
            
            ax.set_title(f'{column} の Q-Q プロット (正規分布との比較)', fontsize=12)
            ax.grid(True, alpha=0.3)
            
            return fig, f"{column}のQ-Qプロットを作成しました"
            
        except Exception as e:
            return None, f"Q-Qプロット作成エラー: {str(e)}"
    
    def pairplot_enhanced(self, data: pd.DataFrame, target_col: str = None, 
                         sample_size: int = 1000) -> tuple:
        """強化されたペアプロット作成"""
        try:
            # 数値データのみ抽出
            numeric_data = data.select_dtypes(include=[np.number])
            
            if numeric_data.empty:
                return None, "数値データが見つかりません"
            
            # サンプリング
            if len(numeric_data) > sample_size:
                numeric_data = numeric_data.sample(n=sample_size, random_state=42)
            
            # 変数数制限
            if len(numeric_data.columns) > 6:
                numeric_data = numeric_data.iloc[:, :6]
            
            # ターゲット列追加
            if target_col and target_col in data.columns:
                target_data = data.loc[numeric_data.index, target_col]
                plot_data = numeric_data.copy()
                plot_data[target_col] = target_data
                hue_col = target_col
            else:
                plot_data = numeric_data
                hue_col = None
            
            # ペアプロット作成
            g = sns.pairplot(plot_data, hue=hue_col, diag_kind='kde', 
                            plot_kws={'alpha': 0.6}, diag_kws={'alpha': 0.7})
            
            g.fig.suptitle('強化ペアプロット', y=1.02, fontsize=16)
            
            return g.fig, "強化ペアプロットを作成しました"
            
        except Exception as e:
            return None, f"ペアプロット作成エラー: {str(e)}"

# インスタンス作成用
visualizer = AdvancedVisualizer() 