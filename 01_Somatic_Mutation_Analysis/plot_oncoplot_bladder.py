import urllib.request
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch

# ======== 顶刊级绘图配置 ========
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['pdf.fonttype'] = 42

print("1. Fetching clinical and mutation data from cBioPortal...")
url_samples = 'https://www.cbioportal.org/api/studies/tmb_mskcc_2018/clinical-data?projection=DETAILED&clinicalDataType=SAMPLE'
req = urllib.request.Request(url_samples, headers={'Accept': 'application/json'})
with urllib.request.urlopen(req) as response:
    sample_data = json.loads(response.read().decode('utf-8'))

samples = {}
for item in sample_data:
    sid = item.get('sampleId')
    if sid not in samples:
        samples[sid] = {'SAMPLE_ID': sid, 'PATIENT_ID': item.get('patientId')}
    samples[sid][item.get('clinicalAttributeId')] = item.get('value')
df_samples = pd.DataFrame(list(samples.values()))

# Filter Bladder Cancer
df_samples['TMB_NONSYNONYMOUS'] = pd.to_numeric(df_samples.get('TMB_NONSYNONYMOUS', np.nan), errors='coerce')
blca_df = df_samples[df_samples['CANCER_TYPE'] == 'Bladder Cancer'].copy()

# High TMB threshold (Top 20%)
tmb_threshold = blca_df['TMB_NONSYNONYMOUS'].quantile(0.8)
high_tmb_df = blca_df[blca_df['TMB_NONSYNONYMOUS'] >= tmb_threshold].copy()
high_tmb_samples = high_tmb_df['SAMPLE_ID'].tolist()

# Define top 10 genes
genes = ['TERT', 'TP53', 'KMT2D', 'ARID1A', 'KMT2C', 'SMARCA4', 'PIK3CA', 'KMT2A', 'APC', 'CREBBP']
gene_ids = [7015, 7157, 8085, 8289, 7378, 6597, 5290, 4297, 324, 1387] # mapping Hugo to Entrez
gene_map = dict(zip(gene_ids, genes))

# Fetch mutations for top 10 genes in Bladder Cancer
url_mut = 'https://www.cbioportal.org/api/molecular-profiles/tmb_mskcc_2018_mutations/mutations/fetch'
fetch_data = {
    "entrezGeneIds": gene_ids,
    "sampleIds": high_tmb_samples
}
req = urllib.request.Request(url_mut, data=json.dumps(fetch_data).encode('utf-8'), method='POST')
req.add_header('Content-Type', 'application/json')
req.add_header('Accept', 'application/json')
with urllib.request.urlopen(req) as response:
    mut_data = json.loads(response.read().decode('utf-8'))

mut_records = []
for m in mut_data:
    mut_records.append({
        'SAMPLE_ID': m['sampleId'],
        'Hugo_Symbol': gene_map.get(m['entrezGeneId'], str(m['entrezGeneId'])),
        'Mutation_Type': m.get('mutationType', 'Unknown')
    })
df_mut = pd.DataFrame(mut_records)

# Simplify mutation types
def simplify_mut_type(mtype):
    m = str(mtype).lower()
    if 'missense' in m: return 'Missense'
    if 'nonsense' in m: return 'Nonsense'
    if 'frame_shift' in m or 'frameshift' in m: return 'Frame Shift'
    if 'splice' in m: return 'Splice Site'
    if 'in_frame' in m: return 'In Frame'
    return 'Other Mut'

df_mut['Simple_Type'] = df_mut['Mutation_Type'].apply(simplify_mut_type)

# Create mutation matrix
mut_matrix = pd.DataFrame(index=genes, columns=high_tmb_samples).fillna('WT')
for _, row in df_mut.iterrows():
    sid = row['SAMPLE_ID']
    gene = row['Hugo_Symbol']
    mtype = row['Simple_Type']
    if mut_matrix.loc[gene, sid] == 'WT':
        mut_matrix.loc[gene, sid] = mtype
    else:
        # Multi-hit if there's already a mutation in that gene for the sample
        if mut_matrix.loc[gene, sid] != mtype:
            mut_matrix.loc[gene, sid] = 'Multi Hit'

# Sort samples dynamically to get the "waterfall" effect (hierarchical sorting)
# We sort by genes in order of frequency or specific genes: ARID1A first, then TERT, TP53...
def get_sort_key(sample):
    key = []
    # Sort key: True/False for each gene mutation (WT=True, Mutant=False so Mutants come first)
    for g in ['ARID1A', 'TERT', 'TP53', 'KMT2D', 'KMT2C', 'SMARCA4', 'PIK3CA', 'KMT2A', 'APC', 'CREBBP']:
        val = mut_matrix.loc[g, sample]
        key.append(0 if val != 'WT' else 1)
    return tuple(key)

sorted_samples = sorted(high_tmb_samples, key=get_sort_key)
mut_matrix = mut_matrix[sorted_samples]
high_tmb_df = high_tmb_df.set_index('SAMPLE_ID').loc[sorted_samples].reset_index()

# Color mapping
color_map = {
    'WT': '#EAECEE',
    'Missense': '#3CB371',
    'Nonsense': '#E64B35',
    'Frame Shift': '#4DBBD5',
    'Splice Site': '#F39B7F',
    'In Frame': '#91D1C2',
    'Multi Hit': '#8491B4',
    'Other Mut': '#B0C4DE'
}

print("2. Plotting highly-polished publication-grade Oncoplot...")
fig = plt.figure(figsize=(14, 8), dpi=300)
gs = gridspec.GridSpec(2, 2, width_ratios=[12, 2], height_ratios=[2, 8], wspace=0.03, hspace=0.08)

ax_top = fig.add_subplot(gs[0, 0])
ax_main = fig.add_subplot(gs[1, 0], sharex=ax_top)
ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)

# 1. Top Barplot (TMB / Mutation Count per patient)
ax_top.bar(np.arange(len(sorted_samples)), high_tmb_df['TMB_NONSYNONYMOUS'], color='#4A90E2', width=0.8, edgecolor='none')
ax_top.set_ylabel('TMB\n(Mut/Mb)', fontsize=10, fontweight='bold')
ax_top.set_xlim(-0.5, len(sorted_samples) - 0.5)
ax_top.tick_params(bottom=False, labelbottom=False, left=True, labelleft=True, labelsize=9)
ax_top.spines['top'].set_visible(False)
ax_top.spines['right'].set_visible(False)
ax_top.spines['bottom'].set_visible(False)
ax_top.grid(axis='y', linestyle=':', alpha=0.5)

# 2. Main Grid
for row_idx, gene in enumerate(genes):
    for col_idx, sample in enumerate(sorted_samples):
        mtype = mut_matrix.loc[gene, sample]
        color = color_map.get(mtype, '#B0C4DE')
        # Draw cell
        rect = plt.Rectangle((col_idx - 0.45, row_idx - 0.45), 0.9, 0.9, color=color, edgecolor='none')
        ax_main.add_patch(rect)

ax_main.set_yticks(np.arange(len(genes)))
ax_main.set_yticklabels(genes, fontsize=11, fontweight='bold', va='center')
ax_main.set_xticks([])
ax_main.set_ylim(-0.5, len(genes) - 0.5)
ax_main.invert_yaxis()  # Put top genes at the top
ax_main.spines['top'].set_visible(False)
ax_main.spines['right'].set_visible(False)
ax_main.spines['bottom'].set_visible(False)
ax_main.spines['left'].set_visible(False)
ax_main.tick_params(left=False, bottom=False)

# Add a dashed separating line for ARID1A mutant vs WT patients
arid1a_mut_count = sum(mut_matrix.loc['ARID1A'] != 'WT')
ax_main.axvline(arid1a_mut_count - 0.5, color='black', linestyle='--', linewidth=1.5)
ax_top.axvline(arid1a_mut_count - 0.5, color='black', linestyle='--', linewidth=1.5)

ax_main.text(arid1a_mut_count / 2 - 0.5, -0.8, 'ARID1A Mutant', ha='center', va='center', fontsize=11, color='#E64B35', fontweight='bold', transform=ax_main.get_xaxis_transform())
ax_main.text(arid1a_mut_count + (len(sorted_samples) - arid1a_mut_count) / 2 - 0.5, -0.8, 'ARID1A Wild-Type', ha='center', va='center', fontsize=11, color='#4DBBD5', fontweight='bold', transform=ax_main.get_xaxis_transform())

# 3. Right Barplot (Mutation Frequency per gene)
mut_freq = (mut_matrix != 'WT').sum(axis=1) / len(sorted_samples) * 100
ax_right.barh(np.arange(len(genes)), mut_freq, color='#7F8C8D', height=0.6, edgecolor='none')
ax_right.set_xlabel('Mutation Freq (%)', fontsize=10, fontweight='bold')
ax_right.set_xlim(0, 100)
ax_right.tick_params(left=False, labelleft=False, bottom=True, labelbottom=True, labelsize=9)
ax_right.spines['top'].set_visible(False)
ax_right.spines['right'].set_visible(False)
ax_right.spines['left'].set_visible(False)
ax_right.grid(axis='x', linestyle=':', alpha=0.5)

# Add percentages on right barplot
for idx, freq in enumerate(mut_freq):
    ax_right.text(freq + 3, idx, f"{freq:.1f}%", va='center', ha='left', fontsize=9, fontweight='bold')

# 4. Legend
legend_elements = [
    Patch(facecolor=color_map['Missense'], label='Missense Mutation'),
    Patch(facecolor=color_map['Nonsense'], label='Nonsense Mutation'),
    Patch(facecolor=color_map['Frame Shift'], label='Frame Shift Indel'),
    Patch(facecolor=color_map['Splice Site'], label='Splice Site'),
    Patch(facecolor=color_map['In Frame'], label='In Frame Indel'),
    Patch(facecolor=color_map['Multi Hit'], label='Multi Hit'),
    Patch(facecolor=color_map['WT'], label='Wild-Type')
]
ax_legend = fig.add_subplot(gs[0, 1])
ax_legend.axis('off')
ax_legend.legend(handles=legend_elements, loc='center left', frameon=False, fontsize=10)

plt.suptitle('Genomic Landscape of High-TMB Bladder Cancer (MSK-IMPACT Cohort)\nStratified by ARID1A Mutation Status', 
             fontsize=16, fontweight='bold', y=0.98)

output_path = 'bladder_oncoplot_waterfall.png'
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(output_path, transparent=True)
print(f"Saved oncoplot to {output_path}")
