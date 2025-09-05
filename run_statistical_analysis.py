#!/usr/bin/env python3
"""
Example script to run statistical analysis on HyperGSR model results.
This script compares two different HyperGSR model variants.
"""

import os
import subprocess
import sys

def run_statistical_analysis():
    """Run statistical analysis comparing two HyperGSR model variants."""
    
    # Define paths to your metrics files
    results_dir = "results/statistical_analysis_coord_shrink001"
    model1_path = "results/stp_gsr/csv/run_stp_gsr/metrics.csv"
    model2_path = "results/hyper_gsr/csv/trans/run_hyper_emb_coord_shrink001/metrics.csv"
    
    # Define model names for better readability
    model1_name = "STP-GSR"
    model2_name = "HyperGSR(coord + shrink001)"
    
    # Check if files exist
    if not os.path.exists(model1_path):
        print(f"Error: Model 1 metrics file not found: {model1_path}")
        return False
    
    if not os.path.exists(model2_path):
        print(f"Error: Model 2 metrics file not found: {model2_path}")
        return False
    
    print("Running statistical analysis...")
    print(f"Model 1: {model1_name} ({model1_path})")
    print(f"Model 2: {model2_name} ({model2_path})")
    print("-" * 80)
    
    # Run the statistical analysis
    cmd = [
        sys.executable, "statistical_analysis.py",
        "--results_dir", results_dir,
        "--model1_path", model1_path,
        "--model2_path", model2_path,
        "--model1_name", model1_name,
        "--model2_name", model2_name
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("Warnings/Errors:", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running statistical analysis: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return False

if __name__ == "__main__":
    success = run_statistical_analysis()
    if success:
        print("\n" + "="*80)
        print("Statistical analysis completed successfully!")
        print("Check the following files:")
        print("  - statistical_analysis_results.csv")
        print("  - statistical_analysis_table.txt")
        print("  - statistical_analysis_results.tex")
        print("  - plots/ (directory with visualizations)")
    else:
        print("\nStatistical analysis failed. Please check the error messages above.")
        sys.exit(1)
