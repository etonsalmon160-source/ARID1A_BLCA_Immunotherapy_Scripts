import os
import urllib.request
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# ======== 🎨 顶刊级绘图配置 ========
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['pdf.fonttype'] = 42

study_id = 'blca_tcga_pan_can_atlas_2018'
target_gene = 8289  # ARID1A

signatures = {
    'T_Cell_Exhaustion': [5133, 29126, 1493, 84868, 3902, 201633, 30048], # PDCD1, CD274, CTLA4, HAVCR2, LAG3, TIGIT, TOX
    'M2_Macrophage': [968, 9332, 4360, 1436, 3586],                    # CD68, CD163, MRC1, CSF1R, IL10
    'Fibroblast_CAF': [59, 2191, 1277, 1278, 5159, 7040]                # ACTA2, FAP, COL1A1, COL1A2, PDGFRB, TGFB1
}

# Collect all Entrez IDs to fetch
entrez_ids = {target_gene}
for genes in signatures.values():
    for g in genes:
        entrez_ids.add(g)
entrez_list = sorted(list(entrez_ids))

print("1. Fetching TCGA BLCA clinical data...")
url_samples = f'https://www.cbioportal.org/api/studies/{study_id}/clinical-data?projection=DETAILED&clinicalDataType=SAMPLE'
req = urllib.request.Request(url_samples, headers={'Accept': 'application/json'})
with urllib.request.urlopen(req) as response:
    sample_data = json.loads(response.read().decode('utf-8'))

samples = {}
for item in sample_data:
    sid = item.get('sampleId')
    if sid not in samples:
        samples[sid] = {'SAMPLE_ID': sid}
    samples[sid][item.get('clinicalAttributeId')] = item.get('value')
df_samples = pd.DataFrame(list(samples.values()))

# Convert columns to numeric
for col in ['MUTATION_COUNT']:
    if col in df_samples.columns:
        df_samples[col] = pd.to_numeric(df_samples[col], errors='coerce')

# Drop samples without Mutation Count
df_samples = df_samples.dropna(subset=['MUTATION_COUNT'])

# Determine High TMB threshold (top 33%, percentile 0.67)
tmb_threshold = df_samples['MUTATION_COUNT'].quantile(0.67)
df_high_tmb = df_samples[df_samples['MUTATION_COUNT'] >= tmb_threshold].copy()
high_tmb_sample_ids = df_high_tmb['SAMPLE_ID'].tolist()
print(f"High TMB cohort size (>= {tmb_threshold:.1f} mutations): {len(df_high_tmb)} samples")

print("2. Fetching RNA expression Z-scores for High TMB cohort...")
url_rna = f'https://www.cbioportal.org/api/molecular-profiles/{study_id}_rna_seq_v2_mrna_median_Zscores/molecular-data/fetch'
fetch_data_rna = {"entrezGeneIds": entrez_list, "sampleIds": high_tmb_sample_ids}
req = urllib.request.Request(url_rna, data=json.dumps(fetch_data_rna).encode('utf-8'), method='POST')
req.add_header('Content-Type', 'application/json')
req.add_header('Accept', 'application/json')
with urllib.request.urlopen(req) as response:
    rna_data = json.loads(response.read().decode('utf-8'))

rna_dict = {}
for item in rna_data:
    sid = item['sampleId']
    gene = item['entrezGeneId']
    val = item['value']
    if val is not None:
        if sid not in rna_dict:
            rna_dict[sid] = {}
        rna_dict[sid][gene] = val

df_rna = pd.DataFrame.from_dict(rna_dict, orient='index')

# Clean target gene
if target_gene in df_rna.columns:
    df_rna = df_rna.dropna(subset=[target_gene])
else:
    raise ValueError("ARID1A gene data not found in RNA-seq dataset!")

# Group patients into ARID1A-Low and ARID1A-High based on median expression within High TMB cohort
median_val = df_rna[target_gene].median()
def group_arid1a(val):
    return 'ARID1A Low' if val <= median_val else 'ARID1A High'

df_rna['ARID1A_Group'] = df_rna[target_gene].apply(group_arid1a)
print(f"Grouped patients (High TMB): {df_rna['ARID1A_Group'].value_counts().to_dict()}")

# Calculate Indices
print("Calculating signature indices...")
for sig_name, markers in signatures.items():
    present_markers = [m for m in markers if m in df_rna.columns]
    print(f"Signature '{sig_name}': {len(present_markers)} out of {len(markers)} markers present.")
    df_rna[sig_name] = df_rna[present_markers].mean(axis=1)

# Generate publication-grade figure
print("Generating publication-grade figure...")
fig, axes = plt.subplots(1, 3, figsize=(15, 6), dpi=300, facecolor='white')
palette = {'ARID1A Low': '#E64B35', 'ARID1A High': '#4DBBD5'}

titles = {
    'T_Cell_Exhaustion': 'T-Cell Exhaustion Index\n(PD-1, CTLA-4, TOX, etc.)',
    'M2_Macrophage': 'Immunosuppressive M2 TAM Index\n(CD163, MRC1, CSF1R, etc.)',
    'Fibroblast_CAF': 'Cancer-Associated Fibroblast Index\n(α-SMA, FAP, COL1A1, etc.)'
}

for i, sig in enumerate(signatures.keys()):
    ax = axes[i]
    sub_df = df_rna[['ARID1A_Group', sig]].dropna()
    
    # Elegant Boxplot
    sns.boxplot(
        x='ARID1A_Group', y=sig, data=sub_df,
        palette=palette, width=0.45, fliersize=0, linewidth=1.2, ax=ax
    )
    # Strip plot with jitter for individual data points
    sns.stripplot(
        x='ARID1A_Group', y=sig, data=sub_df,
        color='black', alpha=0.35, size=3.5, jitter=0.2, ax=ax
    )
    
    # Mann-Whitney U test statistics
    low_vals = sub_df[sub_df['ARID1A_Group'] == 'ARID1A Low'][sig]
    high_vals = sub_df[sub_df['ARID1A_Group'] == 'ARID1A High'][sig]
    
    stat, p_val = stats.mannwhitneyu(low_vals, high_vals, alternative='two-sided')
    
    sig_label = 'ns'
    if p_val < 0.05: sig_label = '*'
    if p_val < 0.01: sig_label = '**'
    if p_val < 0.001: sig_label = '***'
    if p_val < 0.0001: sig_label = '****'
    
    ax.set_title(titles[sig], fontsize=12, fontweight='bold', pad=15)
    ax.set_ylabel('Signature Expression (Z-score Mean)', fontsize=10)
    ax.set_xlabel('')
    ax.set_xticklabels(['ARID1A Low', 'ARID1A High'], fontsize=11, fontweight='bold')
    sns.despine(ax=ax)
    
    # Add statistical significance bar and text on plot
    y_max = sub_df[sig].max()
    y_min = sub_df[sig].min()
    h = (y_max - y_min) * 0.05
    
    ax.plot([0, 0, 1, 1], [y_max + h, y_max + 1.5*h, y_max + 1.5*h, y_max + h], color='black', lw=1.0)
    ax.text(0.5, y_max + 1.8*h, f"Mann-Whitney U\nP = {p_val:.2e} ({sig_label})", ha='center', va='bottom', fontsize=9.5, fontweight='bold')
    
    ax.set_ylim(y_min - 2*h, y_max + 6*h)

plt.suptitle('Immunosuppressive Stroma & Exhausted TME Profiling Stratified by ARID1A Expression\nin Bladder Cancer High TMB Cohort (TCGA-BLCA, N = ' + str(len(df_rna)) + ')', 
             fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.93])

output_path = r'[YOUR_WORKING_DIRECTORY]\arid1a_bulk_tme_indices_high_tmb.png'
plt.savefig(output_path, facecolor='white', edgecolor='none', transparent=False, bbox_inches='tight')
plt.close()

print(f"Successfully generated the ARID1A stratified high TMB bulk TME indices figure to {output_path}")
