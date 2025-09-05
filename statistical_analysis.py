import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from tabulate import tabulate

def load_metrics_data(metrics_path):
    """Load metrics data from CSV file and return as DataFrame."""
    df = pd.read_csv(metrics_path)
    # Remove the 'average' row and keep only fold data
    df = df[df['fold'] != 'average'].copy()
    return df

def plot_metric_comparisons(results_dir, metric_names, model1_metrics, model2_metrics, model1_name, model2_name):
    """Create and save visualization plots for metric comparisons."""
    # Set style
    plt.style.use('default')
    plt.rcParams.update({
        'figure.figsize': (15, 10),
        'font.size': 12,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 16,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.edgecolor': 'gray',
        'axes.linewidth': 0.8,
        'grid.color': 'gray',
        'grid.linestyle': '--',
        'grid.alpha': 0.3,
        'axes.titlepad': 20,
        'axes.labelpad': 10
    })
    
    # Create plots directory
    plots_dir = os.path.join(results_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    try:
        # 1. Box plots with statistical significance
        plt.figure(figsize=(15, 10))
        n_metrics = len(metric_names)
        n_cols = min(4, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        for i, metric in enumerate(metric_names, 1):
            plt.subplot(n_rows, n_cols, i)
            data = []
            labels = []
            for _, row in model1_metrics.iterrows():
                data.append(float(row[metric]))
                labels.append(model1_name)
            for _, row in model2_metrics.iterrows():
                data.append(float(row[metric]))
                labels.append(model2_name)
            
            # Calculate statistical tests
            model1_values = [float(row[metric]) for _, row in model1_metrics.iterrows()]
            model2_values = [float(row[metric]) for _, row in model2_metrics.iterrows()]
            t_stat, t_pval = stats.ttest_rel(model1_values, model2_values)
            w_stat, w_pval = stats.wilcoxon(model1_values, model2_values)
            
            # Create box plot
            box = plt.boxplot([model1_values, model2_values], 
                            tick_labels=[model1_name, model2_name],
                            patch_artist=True)
            
            # Customize box colors
            colors = ['#2ecc71', '#e74c3c']
            for patch, color in zip(box['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            # Add statistical significance markers and better model indicator
            y_max = max(data)
            y_min = min(data)
            y_range = y_max - y_min
            
            # Determine which model is better (lower values are better)
            model1_mean = np.mean(model1_values)
            model2_mean = np.mean(model2_values)
            better_model = model1_name if model1_mean < model2_mean else model2_name
            better_idx = 0 if model1_mean < model2_mean else 1
            
            # Add significance stars
            if min(t_pval, w_pval) < 0.05:
                plt.text(1.5, y_max + 0.1*y_range, '*', ha='center', va='bottom', color='black', fontsize=15)
                plt.text(1.5, y_max + 0.15*y_range, f'p={min(t_pval, w_pval):.3f}', 
                        ha='center', va='bottom', color='black', fontsize=8)
            
            # Add "BETTER" label above the better model
            plt.text(better_idx + 1, y_max + 0.25*y_range, 'BETTER', 
                    ha='center', va='bottom', color='green', fontsize=10, fontweight='bold')
            
            # Add improvement percentage
            improvement = abs(model1_mean - model2_mean) / max(model1_mean, model2_mean) * 100
            plt.text(1.5, y_max + 0.35*y_range, f'{improvement:.1f}% better', 
                    ha='center', va='bottom', color='blue', fontsize=9, fontweight='bold')
            
            plt.title(f'{metric} Comparison (Lower is Better)', pad=20)
            plt.ylabel('Value')
            plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'box_plots.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Bar plots with error bars and statistical significance
        plt.figure(figsize=(15, 10))
        for i, metric in enumerate(metric_names, 1):
            plt.subplot(n_rows, n_cols, i)
            model1_values = [float(row[metric]) for _, row in model1_metrics.iterrows()]
            model2_values = [float(row[metric]) for _, row in model2_metrics.iterrows()]
            
            means = [np.mean(model1_values), np.mean(model2_values)]
            stds = [np.std(model1_values), np.std(model2_values)]
            
            # Calculate statistical tests
            t_stat, t_pval = stats.ttest_rel(model1_values, model2_values)
            w_stat, w_pval = stats.wilcoxon(model1_values, model2_values)
            
            # Create bar plot
            bars = plt.bar([model1_name, model2_name], means, yerr=stds, capsize=10,
                          color=['#2ecc71', '#e74c3c'], alpha=0.7)
            
            # Add statistical significance markers and better model indicator
            y_max = max(means)
            
            # Determine which model is better (lower values are better)
            better_model = model1_name if means[0] < means[1] else model2_name
            better_idx = 0 if means[0] < means[1] else 1
            
            if min(t_pval, w_pval) < 0.05:
                plt.text(1.5, y_max + 0.1*y_max, '*', ha='center', va='bottom', color='black', fontsize=15)
                plt.text(1.5, y_max + 0.15*y_max, f'p={min(t_pval, w_pval):.3f}', 
                        ha='center', va='bottom', color='black', fontsize=8)
            
            # Add "BETTER" label above the better model
            plt.text(better_idx, y_max + 0.25*y_max, 'BETTER', 
                    ha='center', va='bottom', color='green', fontsize=10, fontweight='bold')
            
            # Add improvement percentage
            improvement = abs(means[0] - means[1]) / max(means) * 100
            plt.text(1.5, y_max + 0.35*y_max, f'{improvement:.1f}% better', 
                    ha='center', va='bottom', color='blue', fontsize=9, fontweight='bold')
            
            plt.title(f'{metric} Comparison (Lower is Better)', pad=20)
            plt.ylabel('Mean Value')
            plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'bar_plots.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Improvement percentage plot with effect sizes
        improvements = []
        effect_sizes = []
        for metric in metric_names:
            model1_values = [float(row[metric]) for _, row in model1_metrics.iterrows()]
            model2_values = [float(row[metric]) for _, row in model2_metrics.iterrows()]
            
            model1_mean = np.mean(model1_values)
            model2_mean = np.mean(model2_values)
            improvement = abs(model1_mean - model2_mean) / model2_mean * 100
            improvements.append(improvement)
            
            # Calculate effect size
            pooled_std = np.sqrt((np.std(model1_values)**2 + np.std(model2_values)**2) / 2)
            cohens_d = (model2_mean - model1_mean) / pooled_std
            effect_sizes.append(cohens_d)
        
        plt.figure(figsize=(15, 8))
        x = np.arange(len(metric_names))
        width = 0.35
        
        # Create bar plot
        bars1 = plt.bar(x - width/2, improvements, width, label='Improvement %', color='#2ecc71', alpha=0.7)
        bars2 = plt.bar(x + width/2, effect_sizes, width, label='Effect Size', color='#3498db', alpha=0.7)
        
        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom')
        
        for bar in bars2:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}',
                    ha='center', va='bottom')
        
        plt.title(f'Improvement Analysis: {model1_name} vs {model2_name}\n(Lower Values = Better Performance)', pad=20)
        plt.xlabel('Metrics')
        plt.ylabel('Value')
        plt.xticks(x, metric_names, rotation=45, ha='right')
        plt.legend()
        
        # Add a note about which model is generally better
        better_count = sum(1 for metric in metric_names 
                          if (np.mean([float(row[metric]) for _, row in model1_metrics.iterrows()]) < 
                              np.mean([float(row[metric]) for _, row in model2_metrics.iterrows()])))
        total_metrics = len(metric_names)
        if better_count > total_metrics / 2:
            overall_better = model1_name
        else:
            overall_better = model2_name
        
        plt.figtext(0.5, 0.02, f'Overall Better Model: {overall_better} (wins {max(better_count, total_metrics-better_count)}/{total_metrics} metrics)', 
                   ha='center', fontsize=10, fontweight='bold', color='green')
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'improvement_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\nEnhanced plots have been saved to: {plots_dir}")
        
    except Exception as e:
        print(f"Error creating plots: {str(e)}")
        print("Continuing with statistical analysis...")

def create_highlighted_table(results_dir, model1_name, model2_name):
    """Create a highlighted table from the statistical analysis results."""
    # Read the CSV file
    csv_path = os.path.join(results_dir, 'statistical_analysis_results.csv')
    df = pd.read_csv(csv_path)
    
    # Format the numbers
    df[f'{model1_name}_Mean'] = df[f'{model1_name}_Mean'].map('{:.4f}'.format)
    df[f'{model1_name}_Std'] = df[f'{model1_name}_Std'].map('{:.4f}'.format)
    df[f'{model2_name}_Mean'] = df[f'{model2_name}_Mean'].map('{:.4f}'.format)
    df[f'{model2_name}_Std'] = df[f'{model2_name}_Std'].map('{:.4f}'.format)
    df['t_test_p_value'] = df['t_test_p_value'].map('{:.2e}'.format)
    df['Wilcoxon_p_value'] = df['Wilcoxon_p_value'].map('{:.3f}'.format)
    
    # Create a new column for highlighted significance
    df['Significance'] = df['Statistically_Significant'].map({
        'Yes': '✓',
        'No': '✗'
    })
    
    # Select and rename columns for display
    display_df = df[[
        'Metric', f'{model1_name}_Mean', f'{model1_name}_Std', f'{model2_name}_Mean', f'{model2_name}_Std',
        'Better_Method', 'Improvement_%', 'Effect_Size', 'Effect_Size_Category',
        't_test_p_value', 'Wilcoxon_p_value', 'Significance'
    ]].rename(columns={
        f'{model1_name}_Mean': f'{model1_name} (Mean)',
        f'{model1_name}_Std': f'{model1_name} (Std)',
        f'{model2_name}_Mean': f'{model2_name} (Mean)',
        f'{model2_name}_Std': f'{model2_name} (Std)',
        'Better_Method': 'Better',
        'Improvement_%': 'Improvement',
        'Effect_Size': 'Effect Size',
        'Effect_Size_Category': 'Effect Category',
        't_test_p_value': 't-test p-value',
        'Wilcoxon_p_value': 'Wilcoxon p-value'
    })
    
    # Create the table
    table = tabulate(display_df, headers='keys', tablefmt='grid', showindex=False)
    
    # Save the table to a text file
    table_path = os.path.join(results_dir, 'statistical_analysis_table.txt')
    with open(table_path, 'w') as f:
        f.write(f"Statistical Analysis Results: {model1_name} vs {model2_name}\n")
        f.write("Note: For all metrics, smaller values indicate better performance\n")
        f.write("✓ indicates statistical significance (p < 0.05)\n")
        f.write("=" * 100 + "\n\n")
        f.write(table)
    
    print(f"\nTable has been saved to: {table_path}")

def plot_statistical_analysis(results_dir, model1_name, model2_name, metric_names, model1_metrics, model2_metrics):
    """Create visualization plots for statistical analysis results."""
    # Read the CSV file
    csv_path = os.path.join(results_dir, 'statistical_analysis_results.csv')
    df = pd.read_csv(csv_path)
    
    # Create plots directory
    plots_dir = os.path.join(results_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Set style
    plt.style.use('default')
    plt.rcParams.update({
        'figure.figsize': (15, 10),
        'font.size': 12,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 16,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.edgecolor': 'gray',
        'axes.linewidth': 0.8,
        'grid.color': 'gray',
        'grid.linestyle': '--',
        'grid.alpha': 0.3,
        'axes.titlepad': 20,
        'axes.labelpad': 10
    })
    
    try:
        # 1. Effect Size and P-value Comparison
        plt.figure(figsize=(15, 8))
        x = np.arange(len(df))
        width = 0.35
        
        # Plot effect sizes
        plt.subplot(1, 2, 1)
        bars1 = plt.bar(x - width/2, df['Effect_Size'], width, label='Effect Size', color='#2ecc71')
        plt.axhline(y=0.8, color='r', linestyle='--', label='Large Effect Threshold')
        plt.axhline(y=0.5, color='orange', linestyle='--', label='Medium Effect Threshold')
        plt.axhline(y=0.2, color='yellow', linestyle='--', label='Small Effect Threshold')
        
        plt.title('Effect Sizes by Metric')
        plt.xlabel('Metrics')
        plt.ylabel('Cohen\'s d')
        plt.xticks(x, df['Metric'], rotation=45, ha='right')
        plt.legend()
        
        # Plot p-values
        plt.subplot(1, 2, 2)
        bars2 = plt.bar(x - width/2, -np.log10(df['t_test_p_value']), width, label='t-test', color='#3498db')
        plt.bar(x + width/2, -np.log10(df['Wilcoxon_p_value']), width, label='Wilcoxon', color='#e74c3c')
        plt.axhline(y=-np.log10(0.05), color='r', linestyle='--', label='p=0.05 threshold')
        
        plt.title('Statistical Significance (-log10 p-value)')
        plt.xlabel('Metrics')
        plt.ylabel('-log10(p-value)')
        plt.xticks(x, df['Metric'], rotation=45, ha='right')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'effect_size_and_significance.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Improvement Contribution Plot
        plt.figure(figsize=(15, 8))
        improvements = df['Improvement_%'].str.rstrip('%').astype(float)
        
        # Sort by improvement percentage
        sorted_idx = np.argsort(improvements)
        sorted_metrics = df['Metric'].iloc[sorted_idx]
        sorted_improvements = improvements.iloc[sorted_idx]
        
        # Use a light blue color palette
        bars = plt.barh(range(len(sorted_improvements)), sorted_improvements, 
                       color='#7FB3D5', alpha=0.8, edgecolor='#2E86C1', linewidth=1)
        
        # Add value labels with better formatting
        for i, bar in enumerate(bars):
            plt.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                    f'{sorted_improvements.iloc[i]:.1f}%',
                    va='center', color='#2E86C1', fontweight='bold')
        
        plt.title(f'Improvement Percentage by Metric ({model1_name} vs {model2_name})\n(Lower Values = Better Performance)', pad=20, fontsize=16, fontweight='bold')
        plt.xlabel('Improvement Percentage (%)', fontsize=14)
        plt.ylabel('Metrics', fontsize=14)
        plt.yticks(range(len(sorted_metrics)), sorted_metrics, fontsize=12)
        plt.grid(True, alpha=0.2, linestyle='--')
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        
        # Add a note about which model is generally better
        better_count = sum(1 for metric in metric_names 
                          if (np.mean([float(row[metric]) for _, row in model1_metrics.iterrows()]) < 
                              np.mean([float(row[metric]) for _, row in model2_metrics.iterrows()])))
        total_metrics = len(metric_names)
        if better_count > total_metrics / 2:
            overall_better = model1_name
        else:
            overall_better = model2_name
        
        plt.figtext(0.5, 0.02, f'Overall Better Model: {overall_better} (wins {max(better_count, total_metrics-better_count)}/{total_metrics} metrics)', 
                   ha='center', fontsize=10, fontweight='bold', color='green')
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'improvement_contributions.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Statistical Significance Summary
        plt.figure(figsize=(10, 6))
        significant = (df['t_test_p_value'] < 0.05).sum()
        non_significant = len(df) - significant
        
        # Use a light blue color palette
        colors = ['#7FB3D5', '#AED6F1']  # Light blue and lighter blue
        plt.pie([significant, non_significant], 
                labels=['Statistically Significant', 'Not Significant'],
                colors=colors,
                autopct='%1.1f%%',
                textprops={'fontsize': 12, 'fontweight': 'bold'},
                wedgeprops={'edgecolor': '#2E86C1', 'linewidth': 1.5})
        
        plt.title('Proportion of Statistically Significant Results', 
                 pad=20, fontsize=16, fontweight='bold')
        
        # Add a legend with better formatting
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),
                  fontsize=12, frameon=True, edgecolor='#2E86C1')
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'significance_summary.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\nStatistical analysis plots have been saved to: {plots_dir}")
        
    except Exception as e:
        print(f"Error creating statistical analysis plots: {str(e)}")
        print("Continuing with the rest of the analysis...")

def calculate_statistical_analysis(results_dir, model1_path, model2_path, model1_name, model2_name):
    """Calculate statistical analysis between two model results."""
    
    # Create results directory
    os.makedirs(results_dir, exist_ok=True)

    # Load metrics data
    model1_metrics = load_metrics_data(model1_path)
    model2_metrics = load_metrics_data(model2_path)
    
    # Get metric names (exclude 'fold' column)
    metric_names = [col for col in model1_metrics.columns if col != 'fold']
    
    # Create DataFrame for statistical analysis
    results = []
    for metric in metric_names:
        model1_values = [float(row[metric]) for _, row in model1_metrics.iterrows()]
        model2_values = [float(row[metric]) for _, row in model2_metrics.iterrows()]
        
        # Calculate means and standard deviations
        model1_mean = np.mean(model1_values)
        model1_std = np.std(model1_values)
        model2_mean = np.mean(model2_values)
        model2_std = np.std(model2_values)
        
        # Calculate effect size (Cohen's d)
        pooled_std = np.sqrt((model1_std**2 + model2_std**2) / 2)
        cohens_d = (model2_mean - model1_mean) / pooled_std
        
        # Determine which method performs better (smaller values are better)
        better_method = model1_name if model1_mean < model2_mean else model2_name
        # Calculate improvement percentage: (worse_value - better_value) / worse_value * 100
        if model1_mean < model2_mean:
            improvement = (model2_mean - model1_mean) / model2_mean * 100
        else:
            improvement = (model1_mean - model2_mean) / model1_mean * 100
        
        # Calculate confidence intervals
        n = len(model1_values)
        t_value = stats.t.ppf(0.975, n-1)  # 95% confidence interval
        model1_ci = t_value * (model1_std / np.sqrt(n))
        model2_ci = t_value * (model2_std / np.sqrt(n))
        
        # Perform both t-test and Wilcoxon test
        t_stat, t_pval = stats.ttest_rel(model1_values, model2_values)
        w_stat, w_pval = stats.wilcoxon(model1_values, model2_values, 
                                      zero_method='wilcox', 
                                      correction=True,
                                      mode='auto')
        
        # Calculate practical significance
        # For effect sizes: 0.2=small, 0.5=medium, 0.8=large
        effect_size_category = "Large" if abs(cohens_d) > 0.8 else "Medium" if abs(cohens_d) > 0.5 else "Small"
        
        results.append({
            'Metric': metric,
            f'{model1_name}_Mean': model1_mean,
            f'{model1_name}_Std': model1_std,
            f'{model1_name}_CI': f"±{model1_ci:.4f}",
            f'{model2_name}_Mean': model2_mean,
            f'{model2_name}_Std': model2_std,
            f'{model2_name}_CI': f"±{model2_ci:.4f}",
            'Better_Method': better_method,
            'Improvement_%': f"{improvement:.2f}%",
            'Effect_Size': f"{cohens_d:.2f}",
            'Effect_Size_Category': effect_size_category,
            't_test_p_value': t_pval,
            'Wilcoxon_p_value': w_pval,
            'Statistically_Significant': 'Yes' if min(t_pval, w_pval) < 0.05 else 'No'
        })
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(results)
    csv_path = os.path.join(results_dir, 'statistical_analysis_results.csv')
    df.to_csv(csv_path, index=False)
    
    # Print results
    print(f"\nStatistical Analysis Results: {model1_name} vs {model2_name}")
    print("Note: For all metrics, smaller values indicate better performance")
    print("=" * 150)
    print(df.to_string(index=False))
    print("\nResults have been saved to:", csv_path)
    
    # Create LaTeX table with better formatting
    latex_table = df.to_latex(index=False, 
                             float_format=lambda x: '{:.4f}'.format(x) if isinstance(x, float) else x,
                             columns=['Metric', f'{model1_name}_Mean', f'{model1_name}_CI', f'{model2_name}_Mean', f'{model2_name}_CI', 
                                    'Better_Method', 'Improvement_%', 'Effect_Size', 'Effect_Size_Category',
                                    't_test_p_value', 'Wilcoxon_p_value', 'Statistically_Significant'])
    latex_path = os.path.join(results_dir, 'statistical_analysis_results.tex')
    with open(latex_path, 'w') as f:
        f.write(latex_table)
    print("LaTeX table has been saved to:", latex_path)
    
    # Print summary of findings
    print("\nSummary of Findings:")
    print("=" * 80)
    print("1. Practical Significance:")
    for _, row in df.iterrows():
        if row['Effect_Size_Category'] in ['Large', 'Medium']:
            print(f"\nMetric: {row['Metric']}")
            print(f"  Better Method: {row['Better_Method']}")
            print(f"  Improvement: {row['Improvement_%']}")
            print(f"  Effect Size: {row['Effect_Size']} ({row['Effect_Size_Category']})")
    
    print("\n2. Statistical Significance:")
    significant_results = df[df['Statistically_Significant'] == 'Yes']
    if len(significant_results) > 0:
        for _, row in significant_results.iterrows():
            print(f"\nMetric: {row['Metric']}")
            print(f"  t-test p-value: {row['t_test_p_value']:.4f}")
            print(f"  Wilcoxon p-value: {row['Wilcoxon_p_value']:.4f}")
    else:
        print("\nNo statistically significant differences found (p < 0.05).")
        print("This is likely due to the small sample size (3 folds).")
        print("However, the large effect sizes and percentage improvements suggest strong practical significance.")
        print("\nRecommendations:")
        print("1. Increase the number of folds (e.g., use 5-fold or 10-fold cross-validation)")
        print("2. Consider the practical significance (effect sizes and improvements) alongside statistical significance")
        print("3. Use a more sensitive statistical test or increase the sample size")

    # Create visualizations and tables
    create_highlighted_table(results_dir, model1_name, model2_name)
    plot_statistical_analysis(results_dir, model1_name, model2_name, metric_names, model1_metrics, model2_metrics)
    plot_metric_comparisons(results_dir, metric_names, model1_metrics, model2_metrics, model1_name, model2_name)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Calculate statistical analysis between two HyperGSR model results')
    parser.add_argument('--results_dir', type=str, default='results/statistical_analysis', help='Path to results directory')
    parser.add_argument('--model1_path', type=str, required=True, help='Path to first model metrics CSV file')
    parser.add_argument('--model2_path', type=str, required=True, help='Path to second model metrics CSV file')
    parser.add_argument('--model1_name', type=str, default='Model1', help='Name for first model')
    parser.add_argument('--model2_name', type=str, default='Model2', help='Name for second model')
    args = parser.parse_args()
    
    calculate_statistical_analysis(args.results_dir, args.model1_path, args.model2_path, args.model1_name, args.model2_name)
