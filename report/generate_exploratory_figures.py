"""
Script to generate exploratory analysis figures for the VERONECA report.
Generates:
1. Class distribution bar chart (CN/MCI/AD)
2. Missing value heatmap for clinical features
3. Feature correlation matrix
4. Summary statistics table
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Import config
from src.configs.config import LABELS_FILE, ADNI_WASTE, IMAGE_FOLDER

# Create output directory
OUTPUT_DIR = Path(__file__).parent / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)

# Set style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

def load_data():
    """Load and preprocess ADNI data"""
    print("Loading ADNI data...")
    df = pd.read_csv(LABELS_FILE, on_bad_lines='skip', sep=',')
    
    # Remove unwanted columns
    clinical_df = df.drop(columns=['DX', 'PTID', 'VISCODE'], errors='ignore')
    for col in ADNI_WASTE:
        if col in clinical_df.columns:
            clinical_df = clinical_df.drop(col, axis=1)
    
    # Keep only numeric columns
    clinical_df = clinical_df.select_dtypes(include='number')
    
    # Remove baseline columns
    cols_to_keep = [col for col in clinical_df.columns if '_bl' not in col.lower()]
    clinical_df = clinical_df[cols_to_keep]
    
    # Apply 90% missing threshold
    threshold = 0.9 * len(clinical_df)
    clinical_df = clinical_df.loc[:, clinical_df.isnull().sum() < threshold]
    
    return df, clinical_df

def plot_class_distribution(df, output_path):
    """Plot class distribution bar chart"""
    print("Generating class distribution chart...")
    
    # Use DX_bl for baseline diagnosis
    dx_bl_counts = df['DX_bl'].value_counts()
    
    # Map to simplified categories
    class_mapping = {
        'CN': 'CN',
        'LMCI': 'MCI',
        'EMCI': 'MCI',
        'AD': 'AD',
        'SMC': 'CN'  # Group SMC with CN for simplicity
    }
    
    simplified_counts = {'CN': 0, 'MCI': 0, 'AD': 0}
    for dx_bl, count in dx_bl_counts.items():
        if pd.notna(dx_bl) and dx_bl in class_mapping:
            simplified_counts[class_mapping[dx_bl]] += count
    
    # Create bar chart
    fig, ax = plt.subplots(figsize=(8, 6))
    classes = list(simplified_counts.keys())
    counts = list(simplified_counts.values())
    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    
    bars = ax.bar(classes, counts, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_xlabel('Diagnostic Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Subjects', fontsize=12, fontweight='bold')
    ax.set_title('Baseline Diagnosis Distribution (ADNI Dataset)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")
    
    return simplified_counts

def plot_missing_heatmap(clinical_df, output_path):
    """Plot missing value heatmap"""
    print("Generating missing value heatmap...")
    
    # Calculate missing percentages
    missing_pct = (clinical_df.isnull().sum() / len(clinical_df) * 100).sort_values(ascending=False)
    
    # Take top 30 features with most missing values
    top_missing = missing_pct.head(30)
    
    # Create binary missing matrix for these features
    missing_matrix = clinical_df[top_missing.index].isnull().astype(int)
    
    # Sample rows for visualization (every 20th row)
    sampled_matrix = missing_matrix.iloc[::20, :]
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(sampled_matrix.T, cmap=['#2ecc71', '#e74c3c'], 
                cbar_kws={'label': 'Missing (1) / Present (0)'},
                yticklabels=True, xticklabels=False, ax=ax)
    
    ax.set_xlabel('Samples (every 20th)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Clinical Features', fontsize=12, fontweight='bold')
    ax.set_title('Missing Value Pattern in Clinical Features (Top 30 by Missingness)', 
                 fontsize=13, fontweight='bold', pad=15)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")
    
    return missing_pct

def plot_correlation_matrix(clinical_df, output_path):
    """Plot feature correlation matrix"""
    print("Generating correlation matrix...")
    
    # Select features with less than 50% missing
    low_missing = clinical_df.columns[clinical_df.isnull().sum() / len(clinical_df) < 0.5]
    subset = clinical_df[low_missing]
    
    # Compute correlation matrix
    corr = subset.corr()
    
    # Take top 20 features by variance for readability
    variances = subset.var().sort_values(ascending=False)
    top_features = variances.head(20).index
    corr_subset = corr.loc[top_features, top_features]
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr_subset, annot=False, cmap='coolwarm', center=0,
                vmin=-1, vmax=1, square=True, linewidths=0.5,
                cbar_kws={'label': 'Pearson Correlation'}, ax=ax)
    
    ax.set_title('Feature Correlation Matrix (Top 20 Features by Variance)', 
                 fontsize=13, fontweight='bold', pad=15)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def generate_summary_table(df, clinical_df, class_counts, output_path):
    """Generate summary statistics table"""
    print("Generating summary table...")
    
    # Count subjects with MRI
    cache_dir = Path(IMAGE_FOLDER)
    pkl_files = list(cache_dir.glob('processed_*.pkl'))
    ptids_with_mri = set()
    for pkl_path in pkl_files:
        filename = pkl_path.stem
        if '_I' in filename:
            ptid = filename.split('_I')[0].replace('processed_', '')
            ptids_with_mri.add(ptid)
    
    # Map MRI subjects to classes
    mri_class_counts = {'CN': 0, 'MCI': 0, 'AD': 0}
    class_mapping = {'CN': 'CN', 'LMCI': 'MCI', 'EMCI': 'MCI', 'AD': 'AD', 'SMC': 'CN'}
    
    for ptid in ptids_with_mri:
        subject_data = df[df['PTID'] == ptid]
        if not subject_data.empty:
            dx_bl = subject_data['DX_bl'].iloc[0]
            if pd.notna(dx_bl) and dx_bl in class_mapping:
                mri_class_counts[class_mapping[dx_bl]] += 1
    
    # Create summary table
    summary_data = {
        'Class': ['CN', 'MCI', 'AD', 'Total'],
        'Total Subjects (Clinical)': [
            class_counts['CN'],
            class_counts['MCI'],
            class_counts['AD'],
            sum(class_counts.values())
        ],
        'Subjects with MRI': [
            mri_class_counts['CN'],
            mri_class_counts['MCI'],
            mri_class_counts['AD'],
            sum(mri_class_counts.values())
        ],
        'Clinical Features': [
            len(clinical_df.columns),
            len(clinical_df.columns),
            len(clinical_df.columns),
            len(clinical_df.columns)
        ]
    }
    
    summary_df = pd.DataFrame(summary_data)
    
    # Create table figure
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=summary_df.values,
                     colLabels=summary_df.columns,
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.15, 0.25, 0.25, 0.2])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2)
    
    # Style header
    for i in range(len(summary_df.columns)):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style rows
    for i in range(1, len(summary_df) + 1):
        for j in range(len(summary_df.columns)):
            if i == len(summary_df):  # Total row
                table[(i, j)].set_facecolor('#ecf0f1')
                table[(i, j)].set_text_props(weight='bold')
            else:
                table[(i, j)].set_facecolor('#ffffff' if i % 2 == 0 else '#f8f9fa')
    
    plt.title('Dataset Summary: Clinical and Imaging Modalities', 
              fontsize=13, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")
    
    # Also save as CSV
    csv_path = output_path.parent / "summary_table.csv"
    summary_df.to_csv(csv_path, index=False)
    print(f"Saved CSV: {csv_path}")

def plot_brain_volumes_by_class(df, output_path):
    """Plot brain volumes stratified by diagnostic class"""
    print("Generating brain volumes by class chart...")
    
    # Map diagnosis to simplified categories
    class_mapping = {'CN': 'CN', 'LMCI': 'MCI', 'EMCI': 'MCI', 'AD': 'AD', 'SMC': 'CN'}
    df['DX_simple'] = df['DX_bl'].map(class_mapping)
    
    # Select volumetric features (these are in ADNI_WASTE but we'll extract from raw data)
    volume_features = ['Hippocampus', 'Ventricles', 'Entorhinal']
    available_features = [f for f in volume_features if f in df.columns]
    
    if not available_features:
        print("Warning: No volumetric features found in data. Skipping brain volumes plot.")
        return
    
    # Filter to subjects with valid diagnosis and at least one volume measure
    df_volumes = df[df['DX_simple'].notna()].copy()
    df_volumes = df_volumes[['DX_simple'] + available_features].dropna(subset=available_features, how='all')
    
    # Melt for easier plotting
    df_melt = df_volumes.melt(id_vars=['DX_simple'], value_vars=available_features,
                               var_name='Region', value_name='Volume')
    df_melt = df_melt.dropna(subset=['Volume'])
    
    # Create figure
    fig, axes = plt.subplots(1, len(available_features), figsize=(14, 5))
    if len(available_features) == 1:
        axes = [axes]
    
    colors = {'CN': '#2ecc71', 'MCI': '#f39c12', 'AD': '#e74c3c'}
    
    for i, region in enumerate(available_features):
        ax = axes[i]
        data_region = df_melt[df_melt['Region'] == region]
        
        # Box plot
        parts = ax.boxplot(
            [data_region[data_region['DX_simple'] == cls]['Volume'].values 
             for cls in ['CN', 'MCI', 'AD']],
            labels=['CN', 'MCI', 'AD'],
            patch_artist=True,
            widths=0.6
        )
        
        # Color boxes
        for patch, cls in zip(parts['boxes'], ['CN', 'MCI', 'AD']):
            patch.set_facecolor(colors[cls])
            patch.set_alpha(0.7)
        
        ax.set_title(f'{region} Volume', fontsize=12, fontweight='bold')
        ax.set_ylabel('Volume (mm³)' if i == 0 else '', fontsize=11)
        ax.set_xlabel('Diagnostic Class', fontsize=11)
        ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Brain Regional Volumes by Diagnostic Class', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_clinical_features_by_class(df, output_path):
    """Plot key clinical features stratified by diagnostic class"""
    print("Generating clinical features by class chart...")
    
    # Map diagnosis to simplified categories
    class_mapping = {'CN': 'CN', 'LMCI': 'MCI', 'EMCI': 'MCI', 'AD': 'AD', 'SMC': 'CN'}
    df['DX_simple'] = df['DX_bl'].map(class_mapping)
    
    # Select key discriminative features
    clinical_features = ['MMSE', 'ADAS13', 'FAQ', 'AGE']
    available_features = [f for f in clinical_features if f in df.columns]
    
    if not available_features:
        print("Warning: No clinical features found. Skipping clinical features plot.")
        return
    
    # Filter to subjects with valid diagnosis
    df_clinical = df[df['DX_simple'].notna()].copy()
    df_clinical = df_clinical[['DX_simple'] + available_features].dropna(subset=available_features, how='all')
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    colors = {'CN': '#2ecc71', 'MCI': '#f39c12', 'AD': '#e74c3c'}
    
    for i, feature in enumerate(available_features[:4]):  # Max 4 features
        ax = axes[i]
        
        # Prepare data
        data_by_class = [df_clinical[df_clinical['DX_simple'] == cls][feature].dropna().values 
                         for cls in ['CN', 'MCI', 'AD']]
        
        # Violin plot with box plot overlay
        parts = ax.violinplot(data_by_class, positions=[1, 2, 3], 
                              showmeans=True, showmedians=True, widths=0.7)
        
        # Color violin bodies
        for pc, cls in zip(parts['bodies'], ['CN', 'MCI', 'AD']):
            pc.set_facecolor(colors[cls])
            pc.set_alpha(0.6)
        
        # Add box plot overlay
        bp = ax.boxplot(data_by_class, positions=[1, 2, 3], widths=0.3,
                        patch_artist=False, showfliers=False,
                        boxprops=dict(color='black', linewidth=1.5),
                        whiskerprops=dict(color='black', linewidth=1.5),
                        capprops=dict(color='black', linewidth=1.5),
                        medianprops=dict(color='darkred', linewidth=2))
        
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(['CN', 'MCI', 'AD'])
        ax.set_ylabel(feature, fontsize=11, fontweight='bold')
        ax.set_xlabel('Diagnostic Class', fontsize=11)
        ax.set_title(f'{feature} Distribution', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
    
    # Hide unused subplots if less than 4 features
    for i in range(len(available_features), 4):
        axes[i].axis('off')
    
    plt.suptitle('Key Clinical Features by Diagnostic Class', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    """Main execution"""
    print("=" * 60)
    print("VERONECA Exploratory Analysis Figure Generation")
    print("=" * 60)
    
    # Load data
    df, clinical_df = load_data()
    print(f"Loaded {len(df)} records, {len(clinical_df.columns)} clinical features")
    
    # Generate figures
    class_counts = plot_class_distribution(
        df, OUTPUT_DIR / "class_distribution.png"
    )
    
    missing_pct = plot_missing_heatmap(
        clinical_df, OUTPUT_DIR / "missing_heatmap.png"
    )
    
    plot_correlation_matrix(
        clinical_df, OUTPUT_DIR / "correlation_matrix.png"
    )
    
    generate_summary_table(
        df, clinical_df, class_counts, OUTPUT_DIR / "summary_table.png"
    )
    
    # NEW: Brain volumes and clinical features by class
    plot_brain_volumes_by_class(
        df, OUTPUT_DIR / "brain_volumes_by_class.png"
    )
    
    plot_clinical_features_by_class(
        df, OUTPUT_DIR / "clinical_features_by_class.png"
    )
    
    print("=" * 60)
    print("All figures generated successfully!")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main()
