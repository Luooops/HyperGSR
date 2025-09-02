# Statistical Analysis for HyperGSR Model Comparison

This directory contains tools for performing statistical analysis to compare different HyperGSR model variants using paired t-tests and Wilcoxon signed-rank tests.

## Files

- `statistical_analysis.py`: Main statistical analysis script
- `run_statistical_analysis.py`: Example script to run analysis on your specific data
- `STATISTICAL_ANALYSIS_README.md`: This documentation file

## Quick Start

To run the statistical analysis on your existing data:

```bash
python run_statistical_analysis.py
```

This will compare:
- Model 1: `run_shrink001_improved` 
- Model 2: `run_coord_shrink001_improved`

## Manual Usage

To compare any two models manually:

```bash
python statistical_analysis.py \
    --model1_path "path/to/first/model/metrics.csv" \
    --model2_path "path/to/second/model/metrics.csv" \
    --model1_name "Model1_Name" \
    --model2_name "Model2_Name"
```

## Input Format

The script expects CSV files with the following format:
```csv
fold,mae,mae_deg,mae_bc,mae_ec,mae_ic,mae_pr,mae_katz,clustering_diff,laplacian_frobenius_distance
fold_1,0.176035,0.210376,0.023574,0.016850,0.047109,0.000762,0.067176,0.057431,405.044281
fold_2,0.188763,0.187391,0.024325,0.018065,0.102534,0.000837,0.066115,0.067577,470.557587
fold_3,0.186089,0.170972,0.025212,0.018105,0.100201,0.000874,0.067065,0.065050,459.804352
average,0.183629,0.189580,0.024370,0.017673,0.083281,0.000825,0.066785,0.063352,445.135406
```

**Note**: The script automatically excludes the "average" row and uses only the fold data for statistical analysis.

## Output

The analysis generates several output files in `results/statistical_analysis/`:

### 1. Statistical Results
- `statistical_analysis_results.csv`: Detailed statistical results
- `statistical_analysis_table.txt`: Formatted table for easy reading
- `statistical_analysis_results.tex`: LaTeX table for papers

### 2. Visualizations (in `plots/` directory)
- `box_plots.png`: Box plots with statistical significance markers
- `bar_plots.png`: Bar plots with error bars and significance markers
- `improvement_analysis.png`: Improvement percentages and effect sizes
- `effect_size_and_significance.png`: Effect sizes and p-values
- `improvement_contributions.png`: Horizontal bar chart of improvements
- `significance_summary.png`: Pie chart of significant vs non-significant results

## Statistical Tests

The script performs two statistical tests for each metric:

1. **Paired t-test**: Tests if the means of the two models are significantly different
2. **Wilcoxon signed-rank test**: Non-parametric alternative to the t-test

## Metrics Analyzed

**Important**: All metrics in this analysis follow the principle that **lower values indicate better performance**.

The script analyzes the following metrics:
- `mae`: Mean Absolute Error
- `mae_deg`: Mean Absolute Error for degree
- `mae_bc`: Mean Absolute Error for betweenness centrality
- `mae_ec`: Mean Absolute Error for eigenvector centrality
- `mae_ic`: Mean Absolute Error for information centrality
- `mae_pr`: Mean Absolute Error for PageRank
- `mae_katz`: Mean Absolute Error for Katz centrality
- `clustering_diff`: Clustering coefficient difference
- `laplacian_frobenius_distance`: Laplacian Frobenius distance

## Interpretation

### Statistical Significance
- **p < 0.05**: Statistically significant difference
- **p ≥ 0.05**: No statistically significant difference

### Effect Size (Cohen's d)
- **|d| > 0.8**: Large effect
- **0.5 < |d| ≤ 0.8**: Medium effect  
- **0.2 < |d| ≤ 0.5**: Small effect
- **|d| ≤ 0.2**: Negligible effect

### Practical Significance
- **Improvement %**: Percentage improvement of the better model (calculated as: (worse_value - better_value) / worse_value × 100)
- **Better Method**: Which model performs better (lower values indicate better performance)

## Example Output

```
Statistical Analysis Results: Shrink001_Improved vs Coord_Shrink001_Improved
Note: For all metrics, smaller values indicate better performance
================================================================================

Metric                    Shrink001_Improved_Mean  Coord_Shrink001_Improved_Mean  Better_Method           Improvement_%  Effect_Size  Effect_Size_Category  t_test_p_value  Wilcoxon_p_value  Statistically_Significant
mae                       0.1836                   0.1615                         Coord_Shrink001_Improved  12.05%         -1.23        Large                0.0234          0.0456           Yes
mae_deg                   0.1896                   0.1401                         Coord_Shrink001_Improved  26.10%         -1.45        Large                0.0156          0.0234           Yes
...
```

## Requirements

Make sure you have the following Python packages installed:
```bash
pip install numpy pandas matplotlib seaborn scipy tabulate
```

## Notes

- The analysis uses paired tests since the same test data is used for both models
- With only 3 folds, statistical power is limited; consider practical significance alongside statistical significance
- For more robust results, consider using 5-fold or 10-fold cross-validation
- All visualizations are saved as high-resolution PNG files (300 DPI) suitable for publications
