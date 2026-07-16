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

# Convert MUTATION_COUNT to numeric
df_samples['MUTATION_COUNT'] = pd.to_numeric(df_samples['MUTATION_COUNT'], errors='coerce')
df_samples = df_samples.dropna(subset=['MUTATION_COUNT'])

# Determine High TMB threshold (top 33%, percentile 0.67)
tmb_threshold = df_samples['MUTATION_COUNT'].quantile(0.67)
df_high_tmb = df_samples[df_samples['MUTATION_COUNT'] >= tmb_threshold].copy()
high_tmb_sample_ids = df_high_tmb['SAMPLE_ID'].tolist()
print(f"High TMB cohort size (>= {tmb_threshold:.1f} mutations): {len(df_high_tmb)} samples")

# Fetch ARID1A mutation status for High TMB cohort
print("2. Fetching somatic mutations for High TMB cohort...")
url_mut = f'https://www.cbioportal.org/api/molecular-profiles/{study_id}_mutations/mutations/fetch'
fetch_data_mut = {"entrezGeneIds": [8289], "sampleIds": high_tmb_sample_ids}
req = urllib.request.Request(url_mut, data=json.dumps(fetch_data_mut).encode('utf-8'), method='POST')
req.add_header('Content-Type', 'application/json')
req.add_header('Accept', 'application/json')
with urllib.request.urlopen(req) as response:
    mut_data = json.loads(response.read().decode('utf-8'))

mutated_samples = set([m['sampleId'] for m in mut_data])
df_high_tmb['ARID1A_Status'] = df_high_tmb['SAMPLE_ID'].apply(lambda x: 'Mutant' if x in mutated_samples else 'Wild-Type')

print("3. Fetching RNA expression Z-scores for High TMB cohort...")
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

# Calculate Indices
print("Calculating signature indices...")
for sig_name, markers in signatures.items():
    present_markers = [m for m in markers if m in df_rna.columns]
    df_rna[sig_name] = df_rna[present_markers].mean(axis=1)

# Merge mutation status
df_merged = pd.merge(df_rna[list(signatures.keys())], df_high_tmb[['SAMPLE_ID', 'ARID1A_Status']], left_index=True, right_on='SAMPLE_ID')

# Generate publication-grade figure
print("Generating publication-grade figure...")
fig, axes = plt.subplots(1, 3, figsize=(15, 6), dpi=300, facecolor='white')
palette = {'Wild-Type': '#4DBBD5', 'Mutant': '#E64B35'}

titles = {
    'T_Cell_Exhaustion': 'T-Cell Exhaustion Index\n(PD-1, CTLA-4, TOX, etc.)',
    'M2_Macrophage': 'Immunosuppressive M2 TAM Index\n(CD163, MRC1, CSF1R, etc.)',
    'Fibroblast_CAF': 'Cancer-Associated Fibroblast Index\n(α-SMA, FAP, COL1A1, etc.)'
}

for i, sig in enumerate(signatures.keys()):
    ax = axes[i]
    sub_df = df_merged[['ARID1A_Status', sig]].dropna()
    
    # Elegant Boxplot
    sns.boxplot(
        x='ARID1A_Status', y=sig, data=sub_df, hue='ARID1A_Status',
        hue_order=['Wild-Type', 'Mutant'], order=['Wild-Type', 'Mutant'],
        palette=palette, width=0.45, fliersize=0, linewidth=1.2, ax=ax, legend=False
    )
    # Strip plot with jitter for individual data points
    sns.stripplot(
        x='ARID1A_Status', y=sig, data=sub_df, hue='ARID1A_Status',
        hue_order=['Wild-Type', 'Mutant'], order=['Wild-Type', 'Mutant'],
        palette=palette, alpha=0.35, size=3.5, jitter=0.2, ax=ax, legend=False
    )
    
    # Mann-Whitney U test statistics
    wt_vals = sub_df[sub_df['ARID1A_Status'] == 'Wild-Type'][sig]
    mut_vals = sub_df[sub_df['ARID1A_Status'] == 'Mutant'][sig]
    
    stat, p_val = stats.mannwhitneyu(wt_vals, mut_vals, alternative='two-sided')
    
    sig_label = 'ns'
    if p_val < 0.05: sig_label = '*'
    if p_val < 0.01: sig_label = '**'
    if p_val < 0.001: sig_label = '***'
    if p_val < 0.0001: sig_label = '****'
    
    ax.set_title(titles[sig], fontsize=12, fontweight='bold', pad=15)
    ax.set_ylabel('Signature Expression (Z-score Mean)', fontsize=10)
    ax.set_xlabel('')
    ax.set_xticklabels([f'Wild-Type\n(N={len(wt_vals)})', f'Mutant\n(N={len(mut_vals)})'], fontsize=11, fontweight='bold')
    sns.despine(ax=ax)
    
    # Add statistical significance bar and text on plot
    y_max = sub_df[sig].max()
    y_min = sub_df[sig].min()
    h = (y_max - y_min) * 0.05
    
    ax.plot([0, 0, 1, 1], [y_max + h, y_max + 1.5*h, y_max + 1.5*h, y_max + h], color='black', lw=1.0)
    ax.text(0.5, y_max + 1.8*h, f"Mann-Whitney U\nP = {p_val:.2f} ({sig_label})", ha='center', va='bottom', fontsize=9.5, fontweight='bold')
    
    ax.set_ylim(y_min - 2*h, y_max + 6*h)

plt.suptitle('Immunosuppressive Stroma & Exhausted TME Profiling Stratified by ARID1A Somatic Mutation\nin Bladder Cancer High TMB Cohort (TCGA-BLCA, N = ' + str(len(df_merged)) + ')', 
             fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.93])

output_path = r'[YOUR_WORKING_DIRECTORY]\arid1a_bulk_mutation_indices_high_tmb.png'
plt.savefig(output_path, facecolor='white', edgecolor='none', transparent=False, bbox_inches='tight')
plt.close()

print(f"Successfully generated the ARID1A mutation stratified high TMB bulk TME indices figure to {output_path}")
