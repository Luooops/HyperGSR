import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set style for beautiful plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Define colors for each method
colors = {
    'STP-GSR': '#FF6B6B',      # Red
    'Hyper-GSR (EdgeAttr)': '#4ECDC4',  # Teal
    'Hyper-GSR (Coord)': '#45B7D1'      # Blue
}

def load_data():
    """Load the three CSV files"""
    # File paths
    stp_gsr_path = 'results/stp_gsr/csv/run_stp_gsr/metrics.csv'
    hyper_edgeattr_path = 'results/hyper_gsr/csv/trans/run_hyper_emb_edgeattr_shrink001/metrics.csv'
    hyper_coord_path = 'results/hyper_gsr/csv/trans/run_hyper_emb_coord_shrink001/metrics.csv'
    
    # Load data
    stp_data = pd.read_csv(stp_gsr_path)
    hyper_edgeattr_data = pd.read_csv(hyper_edgeattr_path)
    hyper_coord_data = pd.read_csv(hyper_coord_path)
    
    # Add method labels
    stp_data['Method'] = 'STP-GSR'
    hyper_edgeattr_data['Method'] = 'Hyper-GSR (EdgeAttr)'
    hyper_coord_data['Method'] = 'Hyper-GSR (Coord)'
    
    return stp_data, hyper_edgeattr_data, hyper_coord_data

def create_bar_plots():
    """Create comprehensive bar plots for all metrics"""
    stp_data, hyper_edgeattr_data, hyper_coord_data = load_data()
    
    # Combine all data
    all_data = pd.concat([stp_data, hyper_edgeattr_data, hyper_coord_data], ignore_index=True)
    
    # Get average metrics (excluding fold-specific rows)
    avg_data = all_data[all_data['fold'] == 'average'].copy()
    
    # Define metrics to plot (excluding fold column)
    metrics = ['mae', 'mae_deg', 'mae_bc', 'mae_ec', 'mae_pr', 'mae_katz', 
               'clustering_diff', 'laplacian_frobenius_distance']
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 4, figsize=(20, 12))
    fig.suptitle('Performance Comparison: STP-GSR vs Hyper-GSR Methods', 
                 fontsize=20, fontweight='bold', y=0.98)
    
    # Plot each metric
    for i, metric in enumerate(metrics):
        row = i // 4
        col = i % 4
        ax = axes[row, col]
        
        # Create bar plot
        bars = ax.bar(avg_data['Method'], avg_data[metric], 
                     color=[colors[method] for method in avg_data['Method']],
                     alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Customize plot
        ax.set_title(f'{metric.replace("_", " ").title()}', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_ylabel('Value', fontsize=12, fontweight='bold')
        ax.tick_params(axis='x', rotation=45, labelsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, value in zip(bars, avg_data[metric]):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{value:.4f}', ha='center', va='bottom', 
                   fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    plt.savefig('metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_improvement_heatmap():
    """Create a heatmap showing improvement percentages"""
    stp_data, hyper_edgeattr_data, hyper_coord_data = load_data()
    
    # Get average metrics
    stp_avg = stp_data[stp_data['fold'] == 'average'].iloc[0]
    hyper_edgeattr_avg = hyper_edgeattr_data[hyper_edgeattr_data['fold'] == 'average'].iloc[0]
    hyper_coord_avg = hyper_coord_data[hyper_coord_data['fold'] == 'average'].iloc[0]
    
    # Calculate improvement percentages (lower is better for all metrics)
    metrics = ['mae', 'mae_deg', 'mae_bc', 'mae_ec', 'mae_pr', 'mae_katz', 
               'clustering_diff', 'laplacian_frobenius_distance']
    
    improvement_data = []
    for metric in metrics:
        stp_val = stp_avg[metric]
        edgeattr_val = hyper_edgeattr_avg[metric]
        coord_val = hyper_coord_avg[metric]
        
        # Calculate improvement (positive means better performance)
        edgeattr_improvement = ((stp_val - edgeattr_val) / stp_val) * 100
        coord_improvement = ((stp_val - coord_val) / stp_val) * 100
        
        improvement_data.append([edgeattr_improvement, coord_improvement])
    
    # Create DataFrame
    improvement_df = pd.DataFrame(improvement_data, 
                                 index=[m.replace('_', ' ').title() for m in metrics],
                                 columns=['Hyper-GSR (EdgeAttr)', 'Hyper-GSR (Coord)'])
    
    # Create heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(improvement_df, annot=True, fmt='.2f', cmap='RdYlGn', 
                center=0, cbar_kws={'label': 'Improvement (%)'})
    plt.title('Performance Improvement Over STP-GSR\n(Positive values indicate better performance)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Method', fontsize=14, fontweight='bold')
    plt.ylabel('Metrics', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('improvement_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_radar_chart():
    """Create a radar chart comparing all methods"""
    stp_data, hyper_edgeattr_data, hyper_coord_data = load_data()
    
    # Get average metrics
    stp_avg = stp_data[stp_data['fold'] == 'average'].iloc[0]
    hyper_edgeattr_avg = hyper_edgeattr_data[hyper_edgeattr_data['fold'] == 'average'].iloc[0]
    hyper_coord_avg = hyper_coord_data[hyper_coord_data['fold'] == 'average'].iloc[0]
    
    # Normalize metrics (invert so higher is better)
    metrics = ['mae', 'mae_deg', 'mae_bc', 'mae_ec', 'mae_pr', 'mae_katz', 
               'clustering_diff', 'laplacian_frobenius_distance']
    
    # Normalize data (1 - normalized_value for metrics where lower is better)
    def normalize_data(data):
        normalized = {}
        for metric in metrics:
            # For metrics where lower is better, invert the scale
            max_val = max(stp_avg[metric], hyper_edgeattr_avg[metric], hyper_coord_avg[metric])
            min_val = min(stp_avg[metric], hyper_edgeattr_avg[metric], hyper_coord_avg[metric])
            if max_val != min_val:
                normalized[metric] = 1 - (data[metric] - min_val) / (max_val - min_val)
            else:
                normalized[metric] = 0.5
        return normalized
    
    stp_norm = normalize_data(stp_avg)
    edgeattr_norm = normalize_data(hyper_edgeattr_avg)
    coord_norm = normalize_data(hyper_coord_avg)
    
    # Create radar chart
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # Plot data
    stp_values = [stp_norm[metric] for metric in metrics] + [stp_norm[metrics[0]]]
    edgeattr_values = [edgeattr_norm[metric] for metric in metrics] + [edgeattr_norm[metrics[0]]]
    coord_values = [coord_norm[metric] for metric in metrics] + [coord_norm[metrics[0]]]
    
    ax.plot(angles, stp_values, 'o-', linewidth=2, label='STP-GSR', color=colors['STP-GSR'])
    ax.fill(angles, stp_values, alpha=0.25, color=colors['STP-GSR'])
    
    ax.plot(angles, edgeattr_values, 'o-', linewidth=2, label='Hyper-GSR (EdgeAttr)', color=colors['Hyper-GSR (EdgeAttr)'])
    ax.fill(angles, edgeattr_values, alpha=0.25, color=colors['Hyper-GSR (EdgeAttr)'])
    
    ax.plot(angles, coord_values, 'o-', linewidth=2, label='Hyper-GSR (Coord)', color=colors['Hyper-GSR (Coord)'])
    ax.fill(angles, coord_values, alpha=0.25, color=colors['Hyper-GSR (Coord)'])
    
    # Customize
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics], fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_title('Performance Radar Chart\n(Normalized: Higher is Better)', 
                size=16, fontweight='bold', pad=30)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('radar_chart.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_summary_table():
    """Create a summary table with key statistics"""
    stp_data, hyper_edgeattr_data, hyper_coord_data = load_data()
    
    # Get average metrics
    stp_avg = stp_data[stp_data['fold'] == 'average'].iloc[0]
    hyper_edgeattr_avg = hyper_edgeattr_data[hyper_edgeattr_data['fold'] == 'average'].iloc[0]
    hyper_coord_avg = hyper_coord_data[hyper_coord_data['fold'] == 'average'].iloc[0]
    
    # Create summary DataFrame
    summary_data = {
        'STP-GSR': [stp_avg[metric] for metric in ['mae', 'mae_deg', 'mae_bc', 'mae_ec', 'mae_pr', 'mae_katz', 'clustering_diff', 'laplacian_frobenius_distance']],
        'Hyper-GSR (EdgeAttr)': [hyper_edgeattr_avg[metric] for metric in ['mae', 'mae_deg', 'mae_bc', 'mae_ec', 'mae_pr', 'mae_katz', 'clustering_diff', 'laplacian_frobenius_distance']],
        'Hyper-GSR (Coord)': [hyper_coord_avg[metric] for metric in ['mae', 'mae_deg', 'mae_bc', 'mae_ec', 'mae_pr', 'mae_katz', 'clustering_diff', 'laplacian_frobenius_distance']]
    }
    
    summary_df = pd.DataFrame(summary_data, 
                             index=['MAE', 'MAE Degree', 'MAE Betweenness', 'MAE Eigenvector', 
                                   'MAE PageRank', 'MAE Katz', 'Clustering Diff', 'Laplacian Frobenius'])
    
    # Create table plot
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=summary_df.round(6).values,
                    rowLabels=summary_df.index,
                    colLabels=summary_df.columns,
                    cellLoc='center',
                    loc='center',
                    bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 2)
    
    # Style the table
    for i in range(len(summary_df.columns)):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    for i in range(1, len(summary_df.index) + 1):
        table[(i, 0)].set_facecolor('#f1f1f2')
        table[(i, 0)].set_text_props(weight='bold')
    
    plt.title('Performance Summary Table', fontsize=16, fontweight='bold', pad=20)
    plt.savefig('summary_table.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return summary_df

def main():
    """Main function to generate all plots"""
    print("Generating comparison plots...")
    
    # Create all visualizations
    print("1. Creating bar plots...")
    create_bar_plots()
    
    print("2. Creating improvement heatmap...")
    create_improvement_heatmap()
    
    print("3. Creating radar chart...")
    create_radar_chart()
    
    print("4. Creating summary table...")
    summary_df = create_summary_table()
    
    print("\nAll plots have been generated and saved!")
    print("Files created:")
    print("- metrics_comparison.png")
    print("- improvement_heatmap.png") 
    print("- radar_chart.png")
    print("- summary_table.png")
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(summary_df.round(6))

if __name__ == "__main__":
    main()
