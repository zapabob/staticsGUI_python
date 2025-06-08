import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.patches as patches
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 日本語フォント設定
plt.rcParams['font.family'] = ['DejaVu Sans', 'Hiragino Sans', 'Yu Gothic', 'Meiryo', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']
plt.rcParams['axes.unicode_minus'] = False

class HADPlottingFunctions:
    """HAD統計ソフトのグラフ作成機能"""
    
    def __init__(self, data=None):
        self.data = data
        self.figure_size = (10, 6)
        self.dpi = 100
        self.style = 'seaborn-v0_8'
        
    def set_data(self, data):
        """データセット"""
        self.data = data
    
    def set_style(self, style='seaborn-v0_8'):
        """グラフスタイル設定"""
        try:
            plt.style.use(style)
            self.style = style
        except:
            plt.style.use('default')
            self.style = 'default'
    
    def create_histogram(self, column, bins=30, density=False, kde=True, title=None):
        """ヒストグラム作成"""
        if self.data is None or column not in self.data.columns:
            return None, "データまたは指定された列が見つかりません。"
        
        try:
            fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)
            
            data = self.data[column].dropna()
            
            # ヒストグラム
            n, bins_edges, patches = ax.hist(data, bins=bins, density=density, 
                                           alpha=0.7, color='skyblue', 
                                           edgecolor='black', linewidth=0.5)
            
            # KDE曲線
            if kde and len(data) > 1:
                from scipy.stats import gaussian_kde
                kde_data = gaussian_kde(data)
                x_range = np.linspace(data.min(), data.max(), 100)
                kde_values = kde_data(x_range)
                
                if density:
                    ax.plot(x_range, kde_values, 'r-', linewidth=2, label='KDE')
                else:
                    # 密度を度数に変換
                    kde_values_scaled = kde_values * len(data) * (bins_edges[1] - bins_edges[0])
                    ax.plot(x_range, kde_values_scaled, 'r-', linewidth=2, label='KDE')
            
            # 統計情報を追加
            mean_val = data.mean()
            std_val = data.std()
            median_val = data.median()
            
            # 平均線
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
            ax.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.2f}')
            
            # タイトルとラベル
            if title is None:
                title = f'Histogram of {column}'
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xlabel(column, fontsize=12)
            ax.set_ylabel('Density' if density else 'Frequency', fontsize=12)
            
            # 統計情報テキスト
            stats_text = f'N = {len(data)}\nMean = {mean_val:.3f}\nSD = {std_val:.3f}\nSkewness = {stats.skew(data):.3f}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            
            return fig, f"ヒストグラム作成完了: {column}"
            
        except Exception as e:
            return None, f"ヒストグラム作成中にエラーが発生しました: {e}"
    
    def create_scatter(self, x_column, y_column, hue_column=None, size_column=None, title=None):
        """散布図作成"""
        if self.data is None:
            return None, "データが設定されていません。"
        
        required_cols = [x_column, y_column]
        if hue_column:
            required_cols.append(hue_column)
        if size_column:
            required_cols.append(size_column)
        
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        if missing_cols:
            return None, f"指定された列が見つかりません: {missing_cols}"
        
        try:
            fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)
            
            # データ準備
            plot_data = self.data[required_cols].dropna()
            
            if len(plot_data) == 0:
                return None, "有効なデータがありません。"
            
            x = plot_data[x_column]
            y = plot_data[y_column]
            
            # 散布図作成
            if hue_column and size_column:
                scatter = ax.scatter(x, y, c=plot_data[hue_column], s=plot_data[size_column]*10, 
                                   alpha=0.7, cmap='viridis')
                plt.colorbar(scatter, label=hue_column)
            elif hue_column:
                if plot_data[hue_column].dtype in ['object', 'category']:
                    # カテゴリカル変数
                    unique_categories = plot_data[hue_column].unique()
                    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_categories)))
                    
                    for i, category in enumerate(unique_categories):
                        mask = plot_data[hue_column] == category
                        ax.scatter(x[mask], y[mask], label=str(category), 
                                 alpha=0.7, color=colors[i])
                    ax.legend(title=hue_column)
                else:
                    # 数値変数
                    scatter = ax.scatter(x, y, c=plot_data[hue_column], alpha=0.7, cmap='viridis')
                    plt.colorbar(scatter, label=hue_column)
            elif size_column:
                scatter = ax.scatter(x, y, s=plot_data[size_column]*10, alpha=0.7, color='steelblue')
            else:
                ax.scatter(x, y, alpha=0.7, color='steelblue')
            
            # 回帰直線
            if len(x) > 1:
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                ax.plot(x, p(x), "r--", alpha=0.8, label=f'Regression line (R² = {np.corrcoef(x, y)[0,1]**2:.3f})')
                if not hue_column or plot_data[hue_column].dtype not in ['object', 'category']:
                    ax.legend()
            
            # 相関係数
            correlation = np.corrcoef(x, y)[0, 1]
            
            # タイトルとラベル
            if title is None:
                title = f'Scatter Plot: {x_column} vs {y_column}'
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xlabel(x_column, fontsize=12)
            ax.set_ylabel(y_column, fontsize=12)
            
            # 統計情報
            stats_text = f'N = {len(plot_data)}\nCorrelation = {correlation:.3f}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            
            return fig, f"散布図作成完了: {x_column} vs {y_column}"
            
        except Exception as e:
            return None, f"散布図作成中にエラーが発生しました: {e}"
    
    def create_boxplot(self, columns=None, by_group=None, title=None):
        """箱ひげ図作成"""
        if self.data is None:
            return None, "データが設定されていません。"
        
        try:
            fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)
            
            if columns is None:
                # 全ての数値列
                numeric_data = self.data.select_dtypes(include=[np.number])
                if numeric_data.empty:
                    return None, "数値データが見つかりません。"
                columns = numeric_data.columns.tolist()
            elif isinstance(columns, str):
                columns = [columns]
            
            # グループ別の場合
            if by_group and by_group in self.data.columns:
                plot_data = []
                labels = []
                positions = []
                pos_counter = 1
                
                group_values = self.data[by_group].unique()
                colors = plt.cm.Set1(np.linspace(0, 1, len(group_values)))
                
                for col in columns:
                    for i, group in enumerate(group_values):
                        group_data = self.data[self.data[by_group] == group][col].dropna()
                        if len(group_data) > 0:
                            plot_data.append(group_data)
                            labels.append(f'{col}\n({group})')
                            positions.append(pos_counter)
                            pos_counter += 1
                
                if plot_data:
                    bp = ax.boxplot(plot_data, positions=positions, labels=labels, patch_artist=True)
                    
                    # 色付け
                    color_idx = 0
                    for i, patch in enumerate(bp['boxes']):
                        patch.set_facecolor(colors[color_idx % len(colors)])
                        if (i + 1) % len(group_values) == 0:
                            color_idx += 1
                
            else:
                # 単純な箱ひげ図
                plot_data = []
                valid_columns = []
                
                for col in columns:
                    if col in self.data.columns:
                        col_data = self.data[col].dropna()
                        if len(col_data) > 0:
                            plot_data.append(col_data)
                            valid_columns.append(col)
                
                if plot_data:
                    bp = ax.boxplot(plot_data, labels=valid_columns, patch_artist=True)
                    
                    # 色付け
                    colors = plt.cm.Set1(np.linspace(0, 1, len(plot_data)))
                    for patch, color in zip(bp['boxes'], colors):
                        patch.set_facecolor(color)
            
            # 外れ値の統計
            outlier_info = []
            for i, col in enumerate(valid_columns if not by_group else columns):
                if by_group:
                    continue  # グループ別の場合は複雑になるのでスキップ
                
                col_data = self.data[col].dropna()
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
                outlier_info.append(f'{col}: {len(outliers)} outliers')
            
            # タイトルとラベル
            if title is None:
                if by_group:
                    title = f'Box Plot by {by_group}'
                else:
                    title = 'Box Plot'
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_ylabel('Values', fontsize=12)
            
            # 統計情報
            if outlier_info and not by_group:
                stats_text = '\n'.join(outlier_info[:5])  # 最大5つまで表示
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            return fig, "箱ひげ図作成完了"
            
        except Exception as e:
            return None, f"箱ひげ図作成中にエラーが発生しました: {e}"
    
    def create_correlation_matrix(self, columns=None, method='pearson', title=None):
        """相関行列ヒートマップ作成"""
        if self.data is None:
            return None, "データが設定されていません。"
        
        try:
            if columns is None:
                numeric_data = self.data.select_dtypes(include=[np.number])
            else:
                numeric_data = self.data[columns].select_dtypes(include=[np.number])
            
            if numeric_data.shape[1] < 2:
                return None, "相関行列には少なくとも2つの数値変数が必要です。"
            
            # 相関行列計算
            corr_matrix = numeric_data.corr(method=method)
            
            # マスク（上三角を隠す）
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            
            # 図の作成
            fig, ax = plt.subplots(figsize=(max(8, len(corr_matrix.columns)), max(6, len(corr_matrix.columns))))
            
            # ヒートマップ
            sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
                       square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                       fmt='.3f', ax=ax)
            
            # タイトル
            if title is None:
                title = f'Correlation Matrix ({method.capitalize()})'
            ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
            
            plt.tight_layout()
            
            return fig, f"相関行列ヒートマップ作成完了 ({method})"
            
        except Exception as e:
            return None, f"相関行列ヒートマップ作成中にエラーが発生しました: {e}"
    
    def create_line_plot(self, x_column, y_columns, title=None):
        """線グラフ作成"""
        if self.data is None:
            return None, "データが設定されていません。"
        
        try:
            fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)
            
            if isinstance(y_columns, str):
                y_columns = [y_columns]
            
            # データ準備
            plot_data = self.data[[x_column] + y_columns].dropna()
            
            if len(plot_data) == 0:
                return None, "有効なデータがありません。"
            
            x = plot_data[x_column]
            
            # 線の色とスタイル
            colors = plt.cm.Set1(np.linspace(0, 1, len(y_columns)))
            line_styles = ['-', '--', '-.', ':']
            
            for i, y_col in enumerate(y_columns):
                y = plot_data[y_col]
                ax.plot(x, y, color=colors[i], linestyle=line_styles[i % len(line_styles)],
                       linewidth=2, marker='o', markersize=4, label=y_col)
            
            # タイトルとラベル
            if title is None:
                title = f'Line Plot: {" & ".join(y_columns)} vs {x_column}'
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xlabel(x_column, fontsize=12)
            ax.set_ylabel('Values', fontsize=12)
            
            if len(y_columns) > 1:
                ax.legend()
            
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            
            return fig, f"線グラフ作成完了: {y_columns} vs {x_column}"
            
        except Exception as e:
            return None, f"線グラフ作成中にエラーが発生しました: {e}"
    
    def create_bar_plot(self, x_column, y_column=None, aggfunc='count', title=None):
        """棒グラフ作成"""
        if self.data is None:
            return None, "データが設定されていません。"
        
        try:
            fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)
            
            if y_column is None:
                # 度数分布
                counts = self.data[x_column].value_counts().sort_index()
                ax.bar(range(len(counts)), counts.values, color='steelblue', alpha=0.7)
                ax.set_xticks(range(len(counts)))
                ax.set_xticklabels(counts.index, rotation=45)
                ax.set_ylabel('Frequency')
                
                # 値をバーの上に表示
                for i, v in enumerate(counts.values):
                    ax.text(i, v + max(counts.values) * 0.01, str(v), 
                           ha='center', va='bottom')
                
            else:
                # 集約関数による棒グラフ
                if aggfunc == 'count':
                    agg_data = self.data.groupby(x_column)[y_column].count()
                elif aggfunc == 'mean':
                    agg_data = self.data.groupby(x_column)[y_column].mean()
                elif aggfunc == 'sum':
                    agg_data = self.data.groupby(x_column)[y_column].sum()
                elif aggfunc == 'median':
                    agg_data = self.data.groupby(x_column)[y_column].median()
                else:
                    agg_data = self.data.groupby(x_column)[y_column].mean()
                
                ax.bar(range(len(agg_data)), agg_data.values, color='steelblue', alpha=0.7)
                ax.set_xticks(range(len(agg_data)))
                ax.set_xticklabels(agg_data.index, rotation=45)
                ax.set_ylabel(f'{aggfunc.capitalize()} of {y_column}')
                
                # 値をバーの上に表示
                for i, v in enumerate(agg_data.values):
                    ax.text(i, v + max(agg_data.values) * 0.01, f'{v:.2f}', 
                           ha='center', va='bottom')
            
            # タイトル
            if title is None:
                if y_column:
                    title = f'Bar Plot: {aggfunc.capitalize()} of {y_column} by {x_column}'
                else:
                    title = f'Bar Plot: Frequency of {x_column}'
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xlabel(x_column, fontsize=12)
            
            ax.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            
            return fig, "棒グラフ作成完了"
            
        except Exception as e:
            return None, f"棒グラフ作成中にエラーが発生しました: {e}"
    
    def create_qq_plot(self, column, title=None):
        """Q-Qプロット作成（正規性の確認）"""
        if self.data is None or column not in self.data.columns:
            return None, "データまたは指定された列が見つかりません。"
        
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=self.dpi)
            
            data = self.data[column].dropna()
            
            if len(data) < 3:
                return None, "Q-Qプロットには少なくとも3つのデータが必要です。"
            
            # Q-Qプロット
            stats.probplot(data, dist="norm", plot=ax1)
            ax1.set_title('Q-Q Plot (Normal Distribution)', fontsize=12, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            
            # ヒストグラム + 正規分布曲線
            ax2.hist(data, bins=20, density=True, alpha=0.7, color='skyblue', edgecolor='black')
            
            # 正規分布曲線
            mu, sigma = stats.norm.fit(data)
            x = np.linspace(data.min(), data.max(), 100)
            ax2.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label='Normal PDF')
            
            ax2.set_title('Histogram with Normal Distribution', fontsize=12, fontweight='bold')
            ax2.set_xlabel(column)
            ax2.set_ylabel('Density')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # 正規性検定
            shapiro_stat, shapiro_p = stats.shapiro(data[:5000])  # Shapiro-Wilk検定（最大5000サンプル）
            
            # 統計情報
            stats_text = f'Shapiro-Wilk Test:\nStatistic: {shapiro_stat:.4f}\np-value: {shapiro_p:.6f}\n'
            if shapiro_p > 0.05:
                stats_text += 'Result: Normal distribution\n(p > 0.05)'
            else:
                stats_text += 'Result: Non-normal distribution\n(p ≤ 0.05)'
            
            fig.text(0.02, 0.02, stats_text, fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            if title:
                fig.suptitle(title, fontsize=14, fontweight='bold')
            else:
                fig.suptitle(f'Normality Assessment: {column}', fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            
            return fig, f"Q-Qプロット作成完了: {column}"
            
        except Exception as e:
            return None, f"Q-Qプロット作成中にエラーが発生しました: {e}" 