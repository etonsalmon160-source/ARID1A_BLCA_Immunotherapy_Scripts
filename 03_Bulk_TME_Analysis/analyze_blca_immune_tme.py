import urllib.request
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# ======== 顶刊级绘图配置 ========
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['pdf.fonttype'] = 42

study_id = 'blca_tcga_pan_can_atlas_2018'

print(f"1. Fetching {study_id} clinical data...")
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

# In TCGA PanCancer Atlas, mutation count is 'MUTATION_COUNT'
df_samples['MUTATION_COUNT'] = pd.to_numeric(df_samples.get('MUTATION_COUNT', np.nan), errors='coerce')
df_samples = df_samples.dropna(subset=['MUTATION_COUNT'])

# Determine High TMB (Top 33% as in FAT1 paper "divided into TMB-low, -medium, and -high")
tmb_threshold = df_samples['MUTATION_COUNT'].quantile(0.67)
high_tmb_df = df_samples[df_samples['MUTATION_COUNT'] >= tmb_threshold].copy()
high_tmb_sample_ids = high_tmb_df['SAMPLE_ID'].tolist()

print(f"TCGA BLCA High TMB (>= {tmb_threshold:.1f} mutations) cohort size: {len(high_tmb_df)}")

print("2. Fetching ARID1A mutations for High TMB cohort...")
url_mutations = f'https://www.cbioportal.org/api/molecular-profiles/{study_id}_mutations/mutations/fetch'
fetch_data = {"entrezGeneIds": [8289], "sampleIds": high_tmb_sample_ids}
req = urllib.request.Request(url_mutations, data=json.dumps(fetch_data).encode('utf-8'), method='POST')
req.add_header('Content-Type', 'application/json')
req.add_header('Accept', 'application/json')
with urllib.request.urlopen(req) as response:
    mut_data = json.loads(response.read().decode('utf-8'))

mutated_samples = set([m['sampleId'] for m in mut_data])
print(f"Found {len(mutated_samples)} High TMB samples with ARID1A mutation.")
high_tmb_df['ARID1A_Status'] = high_tmb_df['SAMPLE_ID'].apply(lambda x: 'Mutant' if x in mutated_samples else 'Wild-Type')

print("3. Fetching transcriptome data (TME & Metabolism markers)...")
# Genes: FOXP3 (Tregs), CD8A (Cytotoxic T), CD68 (Macrophages), HK2 (Glycolysis)
gene_map = {50943: 'FOXP3 (Tregs)', 925: 'CD8A (CD8+ T Cells)', 968: 'CD68 (Macrophages)', 3099: 'HK2 (Glycolysis)'}
url_rna = f'https://www.cbioportal.org/api/molecular-profiles/{study_id}_rna_seq_v2_mrna_median_Zscores/molecular-data/fetch'
fetch_data_rna = {"entrezGeneIds": list(gene_map.keys()), "sampleIds": high_tmb_sample_ids}
req = urllib.request.Request(url_rna, data=json.dumps(fetch_data_rna).encode('utf-8'), method='POST')
req.add_header('Content-Type', 'application/json')
req.add_header('Accept', 'application/json')
with urllib.request.urlopen(req) as response:
    rna_data = json.loads(response.read().decode('utf-8'))

# Merge RNA data
rna_records = []
for item in rna_data:
    rna_records.append({
        'SAMPLE_ID': item['sampleId'],
        'Gene': gene_map[item['entrezGeneId']],
        'Expression_Zscore': item['value']
    })
df_rna = pd.DataFrame(rna_records)
df_final = pd.merge(df_rna, high_tmb_df[['SAMPLE_ID', 'ARID1A_Status']], on='SAMPLE_ID')

# Plotting
print("4. Generating TME comparison plots...")
plt.figure(figsize=(12, 8), dpi=300, facecolor='white')
palette = {'Wild-Type': '#4DBBD5', 'Mutant': '#E64B35'}

genes_to_plot = ['CD8A (CD8+ T Cells)', 'FOXP3 (Tregs)', 'CD68 (Macrophages)', 'HK2 (Glycolysis)']
for i, gene in enumerate(genes_to_plot):
    plt.subplot(2, 2, i+1)
    sub_df = df_final[df_final['Gene'] == gene].copy()
    
    sns.boxplot(x='ARID1A_Status', y='Expression_Zscore', data=sub_df, 
                palette=palette, width=0.5, fliersize=0)
    sns.stripplot(x='ARID1A_Status', y='Expression_Zscore', data=sub_df, 
                  color='black', alpha=0.5, size=4, jitter=True)
    
    # Statistics
    mut_vals = sub_df[sub_df['ARID1A_Status'] == 'Mutant']['Expression_Zscore'].dropna()
    wt_vals = sub_df[sub_df['ARID1A_Status'] == 'Wild-Type']['Expression_Zscore'].dropna()
    if len(mut_vals) > 0 and len(wt_vals) > 0:
        _, p_val = stats.mannwhitneyu(mut_vals, wt_vals, alternative='two-sided')
    else:
        p_val = 1.0
        
    sig = 'ns'
    if p_val < 0.05: sig = '*'
    if p_val < 0.01: sig = '**'
    if p_val < 0.001: sig = '***'
    if p_val < 0.0001: sig = '****'
        
    plt.title(f"{gene}\nMann-Whitney P={p_val:.3f} ({sig})", fontsize=12, fontweight='bold')
    plt.ylabel('mRNA Expression (Z-score)')
    plt.xlabel('')
    sns.despine()

plt.suptitle('Tumor Microenvironment (TME) & Metabolism Remodeling by ARID1A Mutation\nin High-TMB Bladder Cancer (TCGA-BLCA Cohort)', 
             fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.95])

output_path = 'bladder_tcga_tme_arid1a_plot.png'
plt.savefig(output_path, transparent=False, facecolor='white')
print(f"Saved plot to {output_path}")
