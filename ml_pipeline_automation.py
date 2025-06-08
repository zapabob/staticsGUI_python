#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Machine Learning Pipeline Automation Module
機械学習パイプライン自動化モジュール

Author: Ryo Minegishi
License: MIT
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from datetime import datetime
import scipy.stats as stats
from tqdm import tqdm
import concurrent.futures
import multiprocessing as mp
import joblib
import pickle
import json

# 機械学習ライブラリ
try:
    from sklearn.model_selection import (
        train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV,
        StratifiedKFold, KFold, TimeSeriesSplit
    )
    from sklearn.preprocessing import (
        StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, 
        OneHotEncoder, PolynomialFeatures
    )
    from sklearn.impute import SimpleImputer
    from sklearn.feature_selection import (
        SelectKBest, f_classif, f_regression, mutual_info_classif, 
        mutual_info_regression, RFECV
    )
    from sklearn.ensemble import (
        RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier,
        GradientBoostingRegressor, ExtraTreesClassifier, ExtraTreesRegressor
    )
    from sklearn.linear_model import (
        LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet
    )
    from sklearn.svm import SVC, SVR
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    from sklearn.naive_bayes import GaussianNB
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
        mean_squared_error, mean_absolute_error, r2_score, classification_report,
        confusion_matrix
    )
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# 高度機械学習ライブラリ
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

# AutoMLライブラリ
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

# プロフェッショナル機能インポート
try:
    from professional_utils import professional_logger, performance_monitor
    PROFESSIONAL_LOGGING = True
except ImportError:
    PROFESSIONAL_LOGGING = False

warnings.filterwarnings('ignore')

class MLPipelineAutomator:
    """機械学習パイプライン自動化クラス"""
    
    def __init__(self, random_state: int = 42, n_jobs: int = -1):
        self.random_state = random_state
        self.n_jobs = n_jobs if n_jobs != -1 else mp.cpu_count()
        self.trained_models = {}
        self.preprocessing_pipelines = {}
        self.results_history = []
        self.best_models = {}
        
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for ML automation")
        
        if PROFESSIONAL_LOGGING:
            professional_logger.info("機械学習パイプライン自動化システム初期化",
                                   random_state=random_state, n_jobs=self.n_jobs)
    
    @performance_monitor.monitor_function("automated_preprocessing") if PROFESSIONAL_LOGGING else lambda x: x
    def automated_preprocessing(self, data: pd.DataFrame, target_col: str, 
                              task_type: str = 'auto') -> Dict[str, Any]:
        """自動前処理"""
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'method': 'Automated Preprocessing',
            'task_type': task_type
        }
        
        # データの基本情報
        data_info = {
            'shape': data.shape,
            'missing_values': data.isnull().sum().to_dict(),
            'data_types': data.dtypes.to_dict()
        }
        
        # ターゲット変数の分析
        if task_type == 'auto':
            if data[target_col].dtype in ['object', 'category'] or data[target_col].nunique() < 10:
                task_type = 'classification'
            else:
                task_type = 'regression'
        
        results['detected_task_type'] = task_type
        
        # 特徴量とターゲットの分離
        feature_cols = [col for col in data.columns if col != target_col]
        X = data[feature_cols].copy()
        y = data[target_col].copy()
        
        # 数値・カテゴリ特徴量の識別
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # 前処理パイプラインの構築
        preprocessor_steps = []
        
        # 数値特徴量の前処理
        if numeric_features:
            numeric_transformer = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            preprocessor_steps.append(('num', numeric_transformer, numeric_features))
        
        # カテゴリ特徴量の前処理
        if categorical_features:
            categorical_transformer = Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
            ])
            preprocessor_steps.append(('cat', categorical_transformer, categorical_features))
        
        if preprocessor_steps:
            preprocessor = ColumnTransformer(preprocessor_steps)
        else:
            preprocessor = None
        
        self.preprocessing_pipelines[target_col] = preprocessor
        
        # ターゲット変数の前処理
        if task_type == 'classification':
            if y.dtype == 'object':
                le = LabelEncoder()
                y_processed = le.fit_transform(y)
                target_encoder = le
            else:
                y_processed = y.values
                target_encoder = None
        else:
            y_processed = y.values
            target_encoder = None
        
        results['preprocessing_info'] = {
            'numeric_features': numeric_features,
            'categorical_features': categorical_features,
            'target_encoder': str(type(target_encoder)) if target_encoder else None,
            'data_info': data_info
        }
        
        return results, X, y_processed, preprocessor, target_encoder
    
    @performance_monitor.monitor_function("automated_feature_selection") if PROFESSIONAL_LOGGING else lambda x: x
    def automated_feature_selection(self, X: pd.DataFrame, y: np.ndarray, 
                                   task_type: str, n_features: int = 'auto') -> Dict[str, Any]:
        """自動特徴選択"""
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'method': 'Automated Feature Selection',
            'original_features': X.shape[1]
        }
        
        if n_features == 'auto':
            n_features = min(20, X.shape[1] // 2)
        
        feature_selectors = {}
        selected_features = {}
        
        # 統計的特徴選択
        if task_type == 'classification':
            # F-test
            f_selector = SelectKBest(score_func=f_classif, k=n_features)
            X_f_selected = f_selector.fit_transform(X, y)
            selected_features['f_test'] = X.columns[f_selector.get_support()].tolist()
            
            # 相互情報量
            mi_selector = SelectKBest(score_func=mutual_info_classif, k=n_features)
            X_mi_selected = mi_selector.fit_transform(X, y)
            selected_features['mutual_info'] = X.columns[mi_selector.get_support()].tolist()
            
        else:  # regression
            # F-test
            f_selector = SelectKBest(score_func=f_regression, k=n_features)
            X_f_selected = f_selector.fit_transform(X, y)
            selected_features['f_test'] = X.columns[f_selector.get_support()].tolist()
            
            # 相互情報量
            mi_selector = SelectKBest(score_func=mutual_info_regression, k=n_features)
            X_mi_selected = mi_selector.fit_transform(X, y)
            selected_features['mutual_info'] = X.columns[mi_selector.get_support()].tolist()
        
        # 重要度ベース選択（Random Forest）
        if task_type == 'classification':
            rf = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        else:
            rf = RandomForestRegressor(n_estimators=100, random_state=self.random_state)
        
        rf.fit(X, y)
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        top_features = feature_importance.head(n_features)['feature'].tolist()
        selected_features['importance_based'] = top_features
        
        # RFE（再帰的特徴削除）
        try:
            rfe = RFECV(rf, step=1, cv=5, scoring='accuracy' if task_type == 'classification' else 'r2')
            rfe.fit(X, y)
            selected_features['rfe'] = X.columns[rfe.support_].tolist()
        except Exception as e:
            selected_features['rfe'] = {'error': str(e)}
        
        # 特徴選択の合意
        all_selected = []
        for method, features in selected_features.items():
            if isinstance(features, list):
                all_selected.extend(features)
        
        # 出現頻度による特徴ランキング
        feature_counts = pd.Series(all_selected).value_counts()
        consensus_features = feature_counts.head(n_features).index.tolist()
        
        results['feature_selection_results'] = selected_features
        results['feature_importance'] = feature_importance.to_dict('records')
        results['consensus_features'] = consensus_features
        results['final_n_features'] = len(consensus_features)
        
        return results, consensus_features
    
    @performance_monitor.monitor_function("automated_model_selection") if PROFESSIONAL_LOGGING else lambda x: x
    def automated_model_selection(self, X: pd.DataFrame, y: np.ndarray, 
                                 task_type: str, cv_folds: int = 5,
                                 include_advanced: bool = True) -> Dict[str, Any]:
        """自動モデル選択"""
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'method': 'Automated Model Selection',
            'task_type': task_type,
            'cv_folds': cv_folds
        }
        
        # 基本モデルの定義
        if task_type == 'classification':
            base_models = {
                'logistic_regression': LogisticRegression(random_state=self.random_state),
                'random_forest': RandomForestClassifier(n_estimators=100, random_state=self.random_state),
                'gradient_boosting': GradientBoostingClassifier(random_state=self.random_state),
                'svm': SVC(random_state=self.random_state),
                'knn': KNeighborsClassifier(),
                'naive_bayes': GaussianNB(),
                'decision_tree': DecisionTreeClassifier(random_state=self.random_state)
            }
            scoring = 'accuracy'
        else:
            base_models = {
                'linear_regression': LinearRegression(),
                'ridge': Ridge(random_state=self.random_state),
                'lasso': Lasso(random_state=self.random_state),
                'elastic_net': ElasticNet(random_state=self.random_state),
                'random_forest': RandomForestRegressor(n_estimators=100, random_state=self.random_state),
                'gradient_boosting': GradientBoostingRegressor(random_state=self.random_state),
                'svr': SVR(),
                'knn': KNeighborsRegressor(),
                'decision_tree': DecisionTreeRegressor(random_state=self.random_state)
            }
            scoring = 'r2'
        
        # 高度なモデルの追加
        if include_advanced:
            if XGBOOST_AVAILABLE:
                if task_type == 'classification':
                    base_models['xgboost'] = xgb.XGBClassifier(random_state=self.random_state)
                else:
                    base_models['xgboost'] = xgb.XGBRegressor(random_state=self.random_state)
            
            if LIGHTGBM_AVAILABLE:
                if task_type == 'classification':
                    base_models['lightgbm'] = lgb.LGBMClassifier(random_state=self.random_state)
                else:
                    base_models['lightgbm'] = lgb.LGBMRegressor(random_state=self.random_state)
        
        # クロスバリデーション
        cv_results = {}
        best_model = None
        best_score = -np.inf
        
        for name, model in tqdm(base_models.items(), desc="モデル評価中"):
            try:
                cv_scores = cross_val_score(model, X, y, cv=cv_folds, 
                                          scoring=scoring, n_jobs=self.n_jobs)
                
                cv_results[name] = {
                    'mean_score': cv_scores.mean(),
                    'std_score': cv_scores.std(),
                    'scores': cv_scores.tolist()
                }
                
                if cv_scores.mean() > best_score:
                    best_score = cv_scores.mean()
                    best_model = (name, model)
                    
            except Exception as e:
                cv_results[name] = {'error': str(e)}
        
        results['cv_results'] = cv_results
        
        if best_model:
            results['best_model'] = {
                'name': best_model[0],
                'score': best_score,
                'model_params': best_model[1].get_params()
            }
            
            self.best_models[f'{task_type}_base'] = best_model
        
        return results
    
    @performance_monitor.monitor_function("hyperparameter_optimization") if PROFESSIONAL_LOGGING else lambda x: x
    def hyperparameter_optimization(self, X: pd.DataFrame, y: np.ndarray,
                                   model_name: str, task_type: str,
                                   n_trials: int = 100) -> Dict[str, Any]:
        """ハイパーパラメータ最適化"""
        
        if not OPTUNA_AVAILABLE:
            return {'error': 'Optunaライブラリが必要です'}
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'method': 'Hyperparameter Optimization',
            'model_name': model_name,
            'n_trials': n_trials
        }
        
        # パラメータ空間の定義
        def objective(trial):
            if model_name == 'random_forest':
                if task_type == 'classification':
                    model = RandomForestClassifier(
                        n_estimators=trial.suggest_int('n_estimators', 50, 300),
                        max_depth=trial.suggest_int('max_depth', 3, 20),
                        min_samples_split=trial.suggest_int('min_samples_split', 2, 20),
                        min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 10),
                        random_state=self.random_state
                    )
                    scoring = 'accuracy'
                else:
                    model = RandomForestRegressor(
                        n_estimators=trial.suggest_int('n_estimators', 50, 300),
                        max_depth=trial.suggest_int('max_depth', 3, 20),
                        min_samples_split=trial.suggest_int('min_samples_split', 2, 20),
                        min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 10),
                        random_state=self.random_state
                    )
                    scoring = 'r2'
            
            elif model_name == 'xgboost' and XGBOOST_AVAILABLE:
                if task_type == 'classification':
                    model = xgb.XGBClassifier(
                        n_estimators=trial.suggest_int('n_estimators', 50, 300),
                        max_depth=trial.suggest_int('max_depth', 3, 10),
                        learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3),
                        subsample=trial.suggest_float('subsample', 0.6, 1.0),
                        colsample_bytree=trial.suggest_float('colsample_bytree', 0.6, 1.0),
                        random_state=self.random_state
                    )
                    scoring = 'accuracy'
                else:
                    model = xgb.XGBRegressor(
                        n_estimators=trial.suggest_int('n_estimators', 50, 300),
                        max_depth=trial.suggest_int('max_depth', 3, 10),
                        learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3),
                        subsample=trial.suggest_float('subsample', 0.6, 1.0),
                        colsample_bytree=trial.suggest_float('colsample_bytree', 0.6, 1.0),
                        random_state=self.random_state
                    )
                    scoring = 'r2'
            
            else:
                raise ValueError(f"Model {model_name} not supported for optimization")
            
            # クロスバリデーションスコア
            cv_scores = cross_val_score(model, X, y, cv=5, scoring=scoring, n_jobs=self.n_jobs)
            return cv_scores.mean()
        
        # 最適化実行
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        results['best_params'] = study.best_params
        results['best_score'] = study.best_value
        results['n_trials_completed'] = len(study.trials)
        
        # 最適化したモデルの訓練
        if model_name == 'random_forest':
            if task_type == 'classification':
                optimized_model = RandomForestClassifier(**study.best_params, random_state=self.random_state)
            else:
                optimized_model = RandomForestRegressor(**study.best_params, random_state=self.random_state)
        elif model_name == 'xgboost' and XGBOOST_AVAILABLE:
            if task_type == 'classification':
                optimized_model = xgb.XGBClassifier(**study.best_params, random_state=self.random_state)
            else:
                optimized_model = xgb.XGBRegressor(**study.best_params, random_state=self.random_state)
        
        optimized_model.fit(X, y)
        self.trained_models[f'{model_name}_optimized'] = optimized_model
        
        # パラメータ重要度
        param_importance = optuna.importance.get_param_importances(study)
        results['param_importance'] = param_importance
        
        return results
    
    @performance_monitor.monitor_function("complete_ml_pipeline") if PROFESSIONAL_LOGGING else lambda x: x
    def complete_ml_pipeline(self, data: pd.DataFrame, target_col: str,
                           task_type: str = 'auto', test_size: float = 0.2,
                           optimize_hyperparams: bool = True) -> Dict[str, Any]:
        """完全機械学習パイプライン"""
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'method': 'Complete ML Pipeline',
            'target_column': target_col
        }
        
        # 1. 前処理
        preprocessing_results, X, y, preprocessor, target_encoder = self.automated_preprocessing(
            data, target_col, task_type
        )
        results['preprocessing'] = preprocessing_results
        
        # 2. 訓練・テスト分割
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state,
            stratify=y if preprocessing_results['detected_task_type'] == 'classification' else None
        )
        
        # 前処理パイプラインの適用
        if preprocessor:
            X_train_processed = preprocessor.fit_transform(X_train)
            X_test_processed = preprocessor.transform(X_test)
            
            # DataFrame化（特徴量名の保持）
            feature_names = []
            for name, transformer, features in preprocessor.transformers_:
                if name == 'num':
                    feature_names.extend(features)
                elif name == 'cat':
                    # OneHotEncoderの特徴量名を取得
                    try:
                        cat_features = transformer.named_steps['onehot'].get_feature_names_out(features)
                        feature_names.extend(cat_features)
                    except:
                        feature_names.extend([f'{f}_encoded' for f in features])
            
            X_train_df = pd.DataFrame(X_train_processed, columns=feature_names)
            X_test_df = pd.DataFrame(X_test_processed, columns=feature_names)
        else:
            X_train_df = X_train
            X_test_df = X_test
        
        # 3. 特徴選択
        feature_selection_results, selected_features = self.automated_feature_selection(
            X_train_df, y_train, preprocessing_results['detected_task_type']
        )
        results['feature_selection'] = feature_selection_results
        
        # 選択された特徴量でデータを絞り込み
        X_train_selected = X_train_df[selected_features]
        X_test_selected = X_test_df[selected_features]
        
        # 4. モデル選択
        model_selection_results = self.automated_model_selection(
            X_train_selected, y_train, preprocessing_results['detected_task_type']
        )
        results['model_selection'] = model_selection_results
        
        # 5. ハイパーパラメータ最適化（オプション）
        if optimize_hyperparams and 'best_model' in model_selection_results:
            best_model_name = model_selection_results['best_model']['name']
            if best_model_name in ['random_forest', 'xgboost']:
                optimization_results = self.hyperparameter_optimization(
                    X_train_selected, y_train, best_model_name,
                    preprocessing_results['detected_task_type']
                )
                results['hyperparameter_optimization'] = optimization_results
                final_model = self.trained_models[f'{best_model_name}_optimized']
            else:
                # 基本モデルを使用
                best_model_info = self.best_models[f'{preprocessing_results["detected_task_type"]}_base']
                final_model = best_model_info[1]
                final_model.fit(X_train_selected, y_train)
        else:
            # 基本モデルを使用
            if 'best_model' in model_selection_results:
                best_model_info = self.best_models[f'{preprocessing_results["detected_task_type"]}_base']
                final_model = best_model_info[1]
                final_model.fit(X_train_selected, y_train)
            else:
                results['error'] = 'No suitable model found'
                return results
        
        # 6. 最終評価
        y_pred = final_model.predict(X_test_selected)
        
        if preprocessing_results['detected_task_type'] == 'classification':
            # 分類メトリクス
            evaluation_metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted'),
                'recall': recall_score(y_test, y_pred, average='weighted'),
                'f1_score': f1_score(y_test, y_pred, average='weighted')
            }
            
            # 混同行列
            cm = confusion_matrix(y_test, y_pred)
            evaluation_metrics['confusion_matrix'] = cm.tolist()
            
            # ROC-AUC（バイナリ分類の場合）
            if len(np.unique(y)) == 2:
                try:
                    y_pred_proba = final_model.predict_proba(X_test_selected)[:, 1]
                    evaluation_metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
                except:
                    pass
        
        else:
            # 回帰メトリクス
            evaluation_metrics = {
                'r2_score': r2_score(y_test, y_pred),
                'mean_squared_error': mean_squared_error(y_test, y_pred),
                'mean_absolute_error': mean_absolute_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
            }
        
        results['final_evaluation'] = evaluation_metrics
        
        # 7. モデル保存情報
        model_info = {
            'model_type': type(final_model).__name__,
            'selected_features': selected_features,
            'preprocessor': preprocessor,
            'target_encoder': target_encoder,
            'task_type': preprocessing_results['detected_task_type']
        }
        
        results['model_info'] = model_info
        self.trained_models['final_model'] = final_model
        self.results_history.append(results)
        
        return results
    
    def save_pipeline(self, filepath: str, model_name: str = 'final_model') -> str:
        """パイプライン保存"""
        
        if model_name not in self.trained_models:
            return f"Model {model_name} not found"
        
        pipeline_data = {
            'model': self.trained_models[model_name],
            'preprocessor': self.preprocessing_pipelines,
            'results_history': self.results_history,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(pipeline_data, f)
        
        return f"Pipeline saved to {filepath}"
    
    def load_pipeline(self, filepath: str) -> str:
        """パイプライン読み込み"""
        
        try:
            with open(filepath, 'rb') as f:
                pipeline_data = pickle.load(f)
            
            if 'model' in pipeline_data:
                self.trained_models['loaded_model'] = pipeline_data['model']
            if 'preprocessor' in pipeline_data:
                self.preprocessing_pipelines.update(pipeline_data['preprocessor'])
            if 'results_history' in pipeline_data:
                self.results_history.extend(pipeline_data['results_history'])
            
            return f"Pipeline loaded from {filepath}"
        
        except Exception as e:
            return f"Error loading pipeline: {str(e)}"

# グローバルインスタンス
ml_pipeline_automator = MLPipelineAutomator() 