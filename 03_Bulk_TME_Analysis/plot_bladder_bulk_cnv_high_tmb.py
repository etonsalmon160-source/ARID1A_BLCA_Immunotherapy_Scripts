import urllib.request
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# ======== 顶刊级绘图配置 (Nature / Cell Style) ========
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

study_id = 'blca_tcga_pan_can_atlas_2018'

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
for col in ['ANEUPLOIDY_SCORE', 'FRACTION_GENOME_ALTERED', 'MUTATION_COUNT']:
    if col in df_samples.columns:
        df_samples[col] = pd.to_numeric(df_samples[col], errors='coerce')

# Drop samples without Fraction Genome Altered or Mutation Count
df_samples = df_samples.dropna(subset=['FRACTION_GENOME_ALTERED', 'MUTATION_COUNT'])

# Determine High TMB (Top 33%, percentile 0.67)
tmb_threshold = df_samples['MUTATION_COUNT'].quantile(0.67)
df_high_tmb = df_samples[df_samples['MUTATION_COUNT'] >= tmb_threshold].copy()
print(f"High TMB cohort size (>= {tmb_threshold:.1f} mutations): {len(df_high_tmb)} samples")

print("2. Fetching ARID1A mutations for High TMB cohort...")
url_mutations = f'https://www.cbioportal.org/api/molecular-profiles/{study_id}_mutations/mutations/fetch'
sample_ids = df_high_tmb['SAMPLE_ID'].tolist()
fetch_data = {"entrezGeneIds": [8289], "sampleIds": sample_ids}
req = urllib.request.Request(url_mutations, data=json.dumps(fetch_data).encode('utf-8'), method='POST')
req.add_header('Content-Type', 'application/json')
req.add_header('Accept', 'application/json')
with urllib.request.urlopen(req) as response:
    mut_data = json.loads(response.read().decode('utf-8'))

mutated_samples = set([m['sampleId'] for m in mut_data])
df_high_tmb['ARID1A_Status'] = df_high_tmb['SAMPLE_ID'].apply(lambda x: 'Mutant' if x in mutated_samples else 'Wild-Type')

# Set up figure
fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=300, facecolor='white')
palette = {'Wild-Type': '#4DBBD5', 'Mutant': '#E64B35'}

# Plot 1: Fraction Genome Altered (FGA)
ax1 = axes[0]
sns.violinplot(data=df_high_tmb, x='ARID1A_Status', y='FRACTION_GENOME_ALTERED', ax=ax1,
               palette=palette, inner='quartile', linewidth=1.5)
sns.stripplot(data=df_high_tmb, x='ARID1A_Status', y='FRACTION_GENOME_ALTERED', ax=ax1,
              color='#2C3E50', alpha=0.3, size=3.5, jitter=0.2)

wt_fga = df_high_tmb[df_high_tmb['ARID1A_Status'] == 'Wild-Type']['FRACTION_GENOME_ALTERED'].dropna()
mut_fga = df_high_tmb[df_high_tmb['ARID1A_Status'] == 'Mutant']['FRACTION_GENOME_ALTERED'].dropna()
stat_fga, pval_fga = stats.mannwhitneyu(wt_fga, mut_fga)

ax1.set_title('Fraction Genome Altered\n(High TMB Cohort)', fontsize=11, fontweight='bold', pad=15)
ax1.set_xlabel('ARID1A Mutation Status', fontsize=10, fontweight='bold')
ax1.set_ylabel('FGA (Proportion Altered)', fontsize=10, fontweight='bold')
ax1.set_xticklabels([f'Wild-Type\n(N={len(wt_fga)})', f'Mutant\n(N={len(mut_fga)})'], fontsize=9)
ax1.text(0.5, 0.95, f"p = {pval_fga:.4f}", transform=ax1.transAxes, ha='center', va='top', fontsize=10, fontweight='bold')
ax1.set_ylim(-0.05, 1.15)

# Plot 2: Aneuploidy Score
ax2 = axes[1]
sns.violinplot(data=df_high_tmb, x='ARID1A_Status', y='ANEUPLOIDY_SCORE', ax=ax2,
               palette=palette, inner='quartile', linewidth=1.5)
sns.stripplot(data=df_high_tmb, x='ARID1A_Status', y='ANEUPLOIDY_SCORE', ax=ax2,
              color='#2C3E50', alpha=0.3, size=3.5, jitter=0.2)

wt_aneu = df_high_tmb[df_high_tmb['ARID1A_Status'] == 'Wild-Type']['ANEUPLOIDY_SCORE'].dropna()
mut_aneu = df_high_tmb[df_high_tmb['ARID1A_Status'] == 'Mutant']['ANEUPLOIDY_SCORE'].dropna()
stat_aneu, pval_aneu = stats.mannwhitneyu(wt_aneu, mut_aneu)

ax2.set_title('Aneuploidy Score\n(High TMB Cohort)', fontsize=11, fontweight='bold', pad=15)
ax2.set_xlabel('ARID1A Mutation Status', fontsize=10, fontweight='bold')
ax2.set_ylabel('Aneuploidy Score', fontsize=10, fontweight='bold')
ax2.set_xticklabels([f'Wild-Type\n(N={len(wt_aneu)})', f'Mutant\n(N={len(mut_aneu)})'], fontsize=9)
ax2.text(0.5, 0.95, f"p = {pval_aneu:.4f}", transform=ax2.transAxes, ha='center', va='top', fontsize=10, fontweight='bold')
ax2.set_ylim(-5, 45)

sns.despine(fig)
plt.suptitle('Genomic Instability Comparison by ARID1A Status in High TMB TCGA BLCA Cohort', 
             fontsize=12, fontweight='bold', y=1.02, color='#1A252C')
plt.tight_layout()

output_path = 'bladder_tcga_bulk_cnv_comparison_high_tmb.png'
plt.savefig(output_path, bbox_inches='tight', transparent=False, facecolor='white')
print(f"Plot successfully saved to {output_path}")
