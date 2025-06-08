#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Multilingual Web Dashboard for Statistical Analysis
多言語統計解析Webダッシュボード

Mac M2 Optimized with ARM64 Support
Author: Ryo Minegishi
License: MIT
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from typing import Dict, List, Optional, Any
from datetime import datetime
import io
import base64
import json
import platform

# 多言語対応
import gettext
import locale

# プロフェッショナル統計解析モジュール
try:
    from advanced_statistics import advanced_analyzer
    from survival_analysis import complete_survival_analyzer
    from bayesian_analysis import deep_bayesian_analyzer
    from ml_pipeline_automation import ml_pipeline_automator
    ADVANCED_MODULES_AVAILABLE = True
except ImportError:
    ADVANCED_MODULES_AVAILABLE = False

warnings.filterwarnings('ignore')

# Mac M2最適化設定
if platform.machine() == 'arm64':
    import os
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'

class MultilingualWebDashboard:
    """多言語統計解析Webダッシュボード"""
    
    def __init__(self):
        self.supported_languages = {'en': 'English', 'ja': '日本語'}
        self.current_language = 'ja'
        self.translations = self._load_translations()
        
        # Streamlit設定
        st.set_page_config(
            page_title="HAD Professional Statistical Analysis",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    def _load_translations(self) -> Dict[str, Dict[str, str]]:
        """翻訳データ読み込み"""
        
        translations = {
            'ja': {
                'title': 'HAD プロフェッショナル統計解析システム',
                'subtitle': '高度統計解析・機械学習・ベイズ統計・生存解析',
                'language_selector': '言語選択',
                'data_upload': 'データアップロード',
                'analysis_type': '解析タイプ',
                'basic_stats': '基本統計',
                'advanced_analysis': '高度解析',
                'survival_analysis': '生存解析',
                'bayesian_analysis': 'ベイズ解析',
                'ml_pipeline': '機械学習パイプライン',
                'multivariate': '多変量解析',
                'time_series': '時系列解析',
                'upload_data': 'CSVファイルをアップロード',
                'data_preview': 'データプレビュー',
                'analysis_results': '解析結果',
                'download_results': '結果ダウンロード',
                'target_variable': '目的変数',
                'predictor_variables': '説明変数',
                'group_variable': 'グループ変数',
                'date_variable': '日付変数',
                'event_variable': 'イベント変数',
                'duration_variable': '時間変数',
                'run_analysis': '解析実行',
                'processing': '処理中...',
                'no_data': 'データがアップロードされていません',
                'error': 'エラーが発生しました',
                'success': '解析が完了しました',
                'export_plot': 'プロット保存',
                'model_performance': 'モデル性能',
                'feature_importance': '特徴量重要度',
                'confusion_matrix': '混同行列',
                'survival_curve': '生存曲線',
                'hazard_ratio': 'ハザード比',
                'posterior_distribution': '事後分布',
                'trace_plot': 'トレースプロット',
                'system_info': 'システム情報'
            },
            'en': {
                'title': 'HAD Professional Statistical Analysis System',
                'subtitle': 'Advanced Statistics・Machine Learning・Bayesian Analysis・Survival Analysis',
                'language_selector': 'Language Selection',
                'data_upload': 'Data Upload',
                'analysis_type': 'Analysis Type',
                'basic_stats': 'Basic Statistics',
                'advanced_analysis': 'Advanced Analysis',
                'survival_analysis': 'Survival Analysis',
                'bayesian_analysis': 'Bayesian Analysis',
                'ml_pipeline': 'Machine Learning Pipeline',
                'multivariate': 'Multivariate Analysis',
                'time_series': 'Time Series Analysis',
                'upload_data': 'Upload CSV File',
                'data_preview': 'Data Preview',
                'analysis_results': 'Analysis Results',
                'download_results': 'Download Results',
                'target_variable': 'Target Variable',
                'predictor_variables': 'Predictor Variables',
                'group_variable': 'Group Variable',
                'date_variable': 'Date Variable',
                'event_variable': 'Event Variable',
                'duration_variable': 'Duration Variable',
                'run_analysis': 'Run Analysis',
                'processing': 'Processing...',
                'no_data': 'No data uploaded',
                'error': 'An error occurred',
                'success': 'Analysis completed',
                'export_plot': 'Export Plot',
                'model_performance': 'Model Performance',
                'feature_importance': 'Feature Importance',
                'confusion_matrix': 'Confusion Matrix',
                'survival_curve': 'Survival Curve',
                'hazard_ratio': 'Hazard Ratio',
                'posterior_distribution': 'Posterior Distribution',
                'trace_plot': 'Trace Plot',
                'system_info': 'System Information'
            }
        }
        
        return translations
    
    def t(self, key: str) -> str:
        """翻訳取得"""
        return self.translations.get(self.current_language, {}).get(key, key)
    
    def render_header(self):
        """ヘッダー描画"""
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.title(self.t('title'))
            st.markdown(f"### {self.t('subtitle')}")
        
        with col2:
            self.current_language = st.selectbox(
                self.t('language_selector'),
                options=list(self.supported_languages.keys()),
                format_func=lambda x: self.supported_languages[x],
                index=list(self.supported_languages.keys()).index(self.current_language)
            )
        
        # システム情報表示
        if st.sidebar.checkbox(self.t('system_info')):
            self._display_system_info()
    
    def _display_system_info(self):
        """システム情報表示"""
        
        system_info = {
            'Platform': platform.platform(),
            'Architecture': platform.machine(),
            'Python': platform.python_version(),
            'Advanced Modules': 'Available' if ADVANCED_MODULES_AVAILABLE else 'Not Available'
        }
        
        st.sidebar.markdown("### " + self.t('system_info'))
        for key, value in system_info.items():
            st.sidebar.text(f"{key}: {value}")
    
    def data_upload_section(self) -> Optional[pd.DataFrame]:
        """データアップロードセクション"""
        
        st.sidebar.markdown("### " + self.t('data_upload'))
        
        uploaded_file = st.sidebar.file_uploader(
            self.t('upload_data'),
            type=['csv'],
            help="CSV形式のファイルをアップロードしてください"
        )
        
        if uploaded_file is not None:
            try:
                data = pd.read_csv(uploaded_file)
                
                st.sidebar.success(f"データ読み込み完了: {data.shape[0]}行 × {data.shape[1]}列")
                
                # データプレビュー
                with st.expander(self.t('data_preview')):
                    st.dataframe(data.head())
                    
                    # 基本情報
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("行数", data.shape[0])
                    with col2:
                        st.metric("列数", data.shape[1])
                    with col3:
                        st.metric("欠損値", data.isnull().sum().sum())
                
                return data
                
            except Exception as e:
                st.sidebar.error(f"{self.t('error')}: {str(e)}")
                return None
        
        return None
    
    def analysis_selection(self) -> str:
        """解析タイプ選択"""
        
        analysis_types = {
            'basic_stats': self.t('basic_stats'),
            'multivariate': self.t('multivariate'),
            'time_series': self.t('time_series'),
            'survival_analysis': self.t('survival_analysis'),
            'bayesian_analysis': self.t('bayesian_analysis'),
            'ml_pipeline': self.t('ml_pipeline')
        }
        
        return st.sidebar.selectbox(
            self.t('analysis_type'),
            options=list(analysis_types.keys()),
            format_func=lambda x: analysis_types[x]
        )
    
    def basic_statistics_analysis(self, data: pd.DataFrame):
        """基本統計解析"""
        
        st.header(self.t('basic_stats'))
        
        # 数値変数の選択
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            st.warning("数値変数が見つかりません")
            return
        
        selected_cols = st.multiselect("変数選択", numeric_cols, default=numeric_cols[:5])
        
        if selected_cols:
            # 基本統計量
            st.subheader("記述統計")
            st.dataframe(data[selected_cols].describe())
            
            # 分布プロット
            st.subheader("分布プロット")
            for col in selected_cols[:4]:  # 最大4つまで
                fig = px.histogram(data, x=col, title=f'{col} の分布')
                st.plotly_chart(fig, use_container_width=True)
            
            # 相関行列
            if len(selected_cols) > 1:
                st.subheader("相関行列")
                corr_matrix = data[selected_cols].corr()
                
                fig = px.imshow(corr_matrix, 
                              title="相関行列",
                              color_continuous_scale='RdBu_r',
                              aspect="auto")
                st.plotly_chart(fig, use_container_width=True)
    
    def multivariate_analysis(self, data: pd.DataFrame):
        """多変量解析"""
        
        if not ADVANCED_MODULES_AVAILABLE:
            st.error("高度解析モジュールが利用できません")
            return
        
        st.header(self.t('multivariate'))
        
        with st.form("multivariate_form"):
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) < 2:
                st.warning("多変量解析には最低2つの数値変数が必要です")
                return
            
            target_col = st.selectbox(self.t('target_variable'), numeric_cols)
            
            submitted = st.form_submit_button(self.t('run_analysis'))
            
            if submitted:
                with st.spinner(self.t('processing')):
                    try:
                        results = advanced_analyzer.multivariate_analysis(data, target_col)
                        
                        # PCA結果
                        st.subheader("主成分分析結果")
                        pca_results = results.get('pca', {})
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Kaiser基準成分数", pca_results.get('n_components_kaiser', 0))
                        with col2:
                            st.metric("寄与率80%成分数", pca_results.get('n_components_80', 0))
                        
                        # 寄与率プロット
                        if 'explained_variance_ratio' in pca_results:
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                y=pca_results['explained_variance_ratio'],
                                mode='lines+markers',
                                name='寄与率'
                            ))
                            fig.add_trace(go.Scatter(
                                y=pca_results['cumulative_variance'],
                                mode='lines+markers',
                                name='累積寄与率'
                            ))
                            fig.update_layout(title="主成分の寄与率")
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # クラスタリング結果
                        st.subheader("クラスタリング結果")
                        clustering_results = results.get('clustering', {})
                        
                        if 'kmeans' in clustering_results:
                            kmeans_results = clustering_results['kmeans']
                            st.metric("最適クラスタ数 (K-means)", kmeans_results.get('optimal_k', 0))
                        
                        st.json(results)
                        
                    except Exception as e:
                        st.error(f"{self.t('error')}: {str(e)}")
    
    def survival_analysis(self, data: pd.DataFrame):
        """生存解析"""
        
        if not ADVANCED_MODULES_AVAILABLE:
            st.error("高度解析モジュールが利用できません")
            return
        
        st.header(self.t('survival_analysis'))
        
        with st.form("survival_form"):
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            all_cols = data.columns.tolist()
            
            duration_col = st.selectbox(self.t('duration_variable'), numeric_cols)
            event_col = st.selectbox(self.t('event_variable'), numeric_cols)
            group_col = st.selectbox(self.t('group_variable'), ['None'] + all_cols)
            
            submitted = st.form_submit_button(self.t('run_analysis'))
            
            if submitted:
                with st.spinner(self.t('processing')):
                    try:
                        group_col_param = None if group_col == 'None' else group_col
                        
                        results = complete_survival_analyzer.kaplan_meier_analysis(
                            data, duration_col, event_col, group_col_param
                        )
                        
                        # 全体結果
                        st.subheader("Kaplan-Meier解析結果")
                        overall_results = results.get('overall', {})
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("中央生存時間", f"{overall_results.get('median_survival', 'N/A')}")
                        with col2:
                            st.metric("対象者数", overall_results.get('n_subjects', 0))
                        with col3:
                            st.metric("イベント数", overall_results.get('n_events', 0))
                        
                        # 生存曲線プロット
                        survival_function = overall_results.get('survival_function', {})
                        if survival_function:
                            fig = go.Figure()
                            
                            timeline = survival_function.get('timeline', [])
                            survival_prob = survival_function.get('survival_prob', [])
                            ci_lower = survival_function.get('ci_lower', [])
                            ci_upper = survival_function.get('ci_upper', [])
                            
                            # 生存曲線
                            fig.add_trace(go.Scatter(
                                x=timeline, y=survival_prob,
                                mode='lines',
                                name='生存確率',
                                line=dict(color='blue')
                            ))
                            
                            # 信頼区間
                            fig.add_trace(go.Scatter(
                                x=timeline + timeline[::-1],
                                y=ci_upper + ci_lower[::-1],
                                fill='tonexty',
                                fillcolor='rgba(0,0,255,0.2)',
                                line=dict(color='rgba(255,255,255,0)'),
                                name='95% 信頼区間'
                            ))
                            
                            fig.update_layout(
                                title="Kaplan-Meier生存曲線",
                                xaxis_title="時間",
                                yaxis_title="生存確率"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # グループ別結果
                        if 'group_analysis' in results:
                            st.subheader("グループ別解析")
                            group_results = results['group_analysis']
                            
                            group_df = pd.DataFrame(group_results).T
                            st.dataframe(group_df)
                        
                        # 統計検定結果
                        if 'statistical_tests' in results:
                            st.subheader("統計検定")
                            tests = results['statistical_tests']
                            
                            if 'overall_logrank' in tests:
                                logrank = tests['overall_logrank']
                                st.write(f"Log-rank検定 p値: {logrank.get('p_value', 'N/A')}")
                                st.write(f"有意性: {'有意' if logrank.get('significant', False) else '非有意'}")
                        
                    except Exception as e:
                        st.error(f"{self.t('error')}: {str(e)}")
    
    def machine_learning_pipeline(self, data: pd.DataFrame):
        """機械学習パイプライン"""
        
        if not ADVANCED_MODULES_AVAILABLE:
            st.error("高度解析モジュールが利用できません")
            return
        
        st.header(self.t('ml_pipeline'))
        
        with st.form("ml_form"):
            all_cols = data.columns.tolist()
            
            target_col = st.selectbox(self.t('target_variable'), all_cols)
            task_type = st.selectbox("タスクタイプ", ['auto', 'classification', 'regression'])
            test_size = st.slider("テストサイズ", 0.1, 0.5, 0.2, 0.05)
            optimize_hyperparams = st.checkbox("ハイパーパラメータ最適化", value=True)
            
            submitted = st.form_submit_button(self.t('run_analysis'))
            
            if submitted:
                with st.spinner(self.t('processing')):
                    try:
                        results = ml_pipeline_automator.complete_ml_pipeline(
                            data, target_col, task_type, test_size, optimize_hyperparams
                        )
                        
                        # 最終評価結果
                        st.subheader(self.t('model_performance'))
                        
                        if 'final_evaluation' in results:
                            metrics = results['final_evaluation']
                            
                            # メトリクス表示
                            metric_cols = st.columns(len(metrics))
                            for i, (metric, value) in enumerate(metrics.items()):
                                if isinstance(value, (int, float)):
                                    metric_cols[i].metric(metric, f"{value:.4f}")
                        
                        # モデル選択結果
                        if 'model_selection' in results:
                            st.subheader("モデル選択結果")
                            model_selection = results['model_selection']
                            
                            if 'best_model' in model_selection:
                                best_model = model_selection['best_model']
                                st.write(f"最優秀モデル: {best_model.get('name', 'N/A')}")
                                st.write(f"スコア: {best_model.get('score', 'N/A'):.4f}")
                            
                            # CV結果
                            if 'cv_results' in model_selection:
                                cv_results = model_selection['cv_results']
                                
                                cv_df = pd.DataFrame({
                                    'Model': list(cv_results.keys()),
                                    'Mean Score': [r.get('mean_score', 0) if isinstance(r, dict) else 0 
                                                 for r in cv_results.values()],
                                    'Std Score': [r.get('std_score', 0) if isinstance(r, dict) else 0 
                                                for r in cv_results.values()]
                                })
                                
                                fig = px.bar(cv_df, x='Model', y='Mean Score', 
                                           error_y='Std Score',
                                           title="モデル性能比較")
                                st.plotly_chart(fig, use_container_width=True)
                        
                        # 特徴選択結果
                        if 'feature_selection' in results:
                            st.subheader(self.t('feature_importance'))
                            feature_selection = results['feature_selection']
                            
                            if 'feature_importance' in feature_selection:
                                importance_data = feature_selection['feature_importance']
                                importance_df = pd.DataFrame(importance_data)
                                
                                fig = px.bar(importance_df.head(10), 
                                           x='importance', y='feature',
                                           orientation='h',
                                           title="特徴量重要度 (Top 10)")
                                st.plotly_chart(fig, use_container_width=True)
                        
                        # 結果詳細
                        with st.expander("詳細結果"):
                            st.json(results)
                        
                    except Exception as e:
                        st.error(f"{self.t('error')}: {str(e)}")
    
    def run(self):
        """メインアプリケーション実行"""
        
        # ヘッダー
        self.render_header()
        
        # データアップロード
        data = self.data_upload_section()
        
        if data is not None:
            # 解析タイプ選択
            analysis_type = self.analysis_selection()
            
            # 解析実行
            if analysis_type == 'basic_stats':
                self.basic_statistics_analysis(data)
            elif analysis_type == 'multivariate':
                self.multivariate_analysis(data)
            elif analysis_type == 'survival_analysis':
                self.survival_analysis(data)
            elif analysis_type == 'ml_pipeline':
                self.machine_learning_pipeline(data)
            else:
                st.info(f"{analysis_type} は現在開発中です")
        
        else:
            st.info(self.t('no_data'))
            
            # サンプルデータ生成オプション
            if st.button("サンプルデータ生成"):
                sample_data = self._generate_sample_data()
                st.session_state['sample_data'] = sample_data
                st.success("サンプルデータを生成しました")
                st.rerun()
    
    def _generate_sample_data(self) -> pd.DataFrame:
        """サンプルデータ生成"""
        
        np.random.seed(42)
        n_samples = 200
        
        data = pd.DataFrame({
            'age': np.random.normal(50, 15, n_samples),
            'income': np.random.lognormal(10, 0.5, n_samples),
            'education_years': np.random.randint(8, 20, n_samples),
            'gender': np.random.choice(['Male', 'Female'], n_samples),
            'region': np.random.choice(['North', 'South', 'East', 'West'], n_samples),
            'satisfaction': np.random.randint(1, 6, n_samples),
            'purchase_amount': np.random.exponential(100, n_samples),
            'time_to_event': np.random.exponential(12, n_samples),
            'event_occurred': np.random.binomial(1, 0.7, n_samples)
        })
        
        return data

# メインアプリケーション
def main():
    """メイン関数"""
    
    dashboard = MultilingualWebDashboard()
    
    # セッション状態の初期化
    if 'sample_data' in st.session_state:
        # サンプルデータがある場合は表示
        st.sidebar.success("サンプルデータが利用可能です")
        if st.sidebar.button("サンプルデータを使用"):
            data = st.session_state['sample_data']
            st.session_state['uploaded_data'] = data
    
    dashboard.run()

if __name__ == "__main__":
    main() 