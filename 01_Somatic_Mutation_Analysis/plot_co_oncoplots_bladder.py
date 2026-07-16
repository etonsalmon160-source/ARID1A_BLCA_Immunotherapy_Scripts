import urllib.request
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
import os
import shutil
import ssl
import time

ssl._create_default_https_context = ssl._create_unverified_context

# ======== 🎨 顶刊级绘图配置 ========
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['pdf.fonttype'] = 42

study_id_tcga = 'blca_tcga_pan_can_atlas_2018'
study_id_msk = 'tmb_mskcc_2018'

studies = {
    study_id_msk: 'tmb_mskcc_2018_mutations',
    study_id_tcga: 'blca_tcga_pan_can_atlas_2018_mutations'
}

def fetch_cbioportal(url, data=None, retries=5):
    for i in range(retries):
        try:
            req = urllib.request.Request(url)
            req.add_header('Accept', 'application/json')
            if data is not None:
                req.add_header('Content-Type', 'application/json')
                encoded_data = json.dumps(data).encode('utf-8')
                with urllib.request.urlopen(req, data=encoded_data, timeout=90) as response:
                    return json.loads(response.read().decode('utf-8'))
            else:
                with urllib.request.urlopen(req, timeout=90) as response:
                    return json.loads(response.read().decode('utf-8'))
        except Exception as e:
            print(f"Request failed (attempt {i+1}/{retries}): {e}. Retrying in 2 seconds...")
            time.sleep(2)
            if i == retries - 1:
                raise e

print("1. Fetching clinical data for combined cohorts from cBioPortal...")
all_samples = []

for study_id, mut_profile in studies.items():
    print(f"Loading clinical data for {study_id}...")
    url = f'https://www.cbioportal.org/api/studies/{study_id}/clinical-data?projection=DETAILED&clinicalDataType=SAMPLE'
    sample_data = fetch_cbioportal(url)
    
    samples = {}
    for item in sample_data:
        sid = item.get('sampleId')
        if sid not in samples:
            samples[sid] = {'SAMPLE_ID': sid, 'Study_ID': study_id, 'Mut_Profile': mut_profile}
        samples[sid][item.get('clinicalAttributeId')] = item.get('value')
    df_study = pd.DataFrame(list(samples.values()))
    all_samples.append(df_study)
    print(f"Loaded {len(df_study)} samples from {study_id}")

df_combined = pd.concat(all_samples, axis=0, ignore_index=True)

df_combined['TMB_NONSYNONYMOUS'] = pd.to_numeric(df_combined.get('TMB_NONSYNONYMOUS', np.nan), errors='coerce')
df_combined['MUTATION_COUNT'] = pd.to_numeric(df_combined.get('MUTATION_COUNT', np.nan), errors='coerce')

# Filter for Bladder Cancer
if 'CANCER_TYPE' in df_combined.columns:
    blca_df = df_combined[df_combined['CANCER_TYPE'] == 'Bladder Cancer'].copy()
else:
    blca_df = df_combined[df_combined['Study_ID'] == study_id_tcga].copy()
    blca_df_msk = df_combined[(df_combined['Study_ID'] == study_id_msk) & (df_combined['CANCER_TYPE'] == 'Bladder Cancer')]
    blca_df = pd.concat([blca_df, blca_df_msk], axis=0)

# Clean TMB: Use TMB_NONSYNONYMOUS, fallback to MUTATION_COUNT / 38.0 for panel scaling
blca_df['TMB_NONSYNONYMOUS'] = blca_df['TMB_NONSYNONYMOUS'].fillna(blca_df['MUTATION_COUNT'] / 38.0)
blca_df = blca_df.dropna(subset=['TMB_NONSYNONYMOUS'])

print(f"\nTotal combined Bladder Cancer cohort size: N = {len(blca_df)}")

# Calculate Tertiles for TMB-Low, Medium, High across the integrated cohort
q33 = blca_df['TMB_NONSYNONYMOUS'].quantile(1/3)
q67 = blca_df['TMB_NONSYNONYMOUS'].quantile(2/3)

blca_df['TMB_Group'] = blca_df['TMB_NONSYNONYMOUS'].apply(
    lambda x: 'Low' if x < q33 else ('Medium' if x < q67 else 'High')
)

low_samples = blca_df[blca_df['TMB_Group'] == 'Low']['SAMPLE_ID'].tolist()
med_samples = blca_df[blca_df['TMB_Group'] == 'Medium']['SAMPLE_ID'].tolist()
high_samples = blca_df[blca_df['TMB_Group'] == 'High']['SAMPLE_ID'].tolist()

print(f"Integrated Group Sizes: Low N = {len(low_samples)}, Medium N = {len(med_samples)}, High N = {len(high_samples)}")

# Load gene list for MSK-IMPACT 468 panel
print("\n2. Loading IMPACT468 panel gene list...")
url_panel = 'https://www.cbioportal.org/api/gene-panels/IMPACT468'
panel_data = fetch_cbioportal(url_panel)
entrez_gene_ids = [g['entrezGeneId'] for g in panel_data['genes']]
entrez_gene_map = {g['entrezGeneId']: g['hugoGeneSymbol'] for g in panel_data['genes']}

# Fetch mutations from cBioPortal
print("\n3. Fetching somatic mutations from cBioPortal...")
mut_records = []
for study_id, profile_id in studies.items():
    study_samples = blca_df[blca_df['Study_ID'] == study_id]['SAMPLE_ID'].tolist()
    if not study_samples:
        continue
    print(f"Fetching mutations for {len(study_samples)} samples in {study_id}...")
    batch_size = 100
    for i in range(0, len(study_samples), batch_size):
        batch_samples = study_samples[i:i+batch_size]
        url_mut = f'https://www.cbioportal.org/api/molecular-profiles/{profile_id}/mutations/fetch'
        fetch_data = {"entrezGeneIds": entrez_gene_ids, "sampleIds": batch_samples}
        batch_muts = fetch_cbioportal(url_mut, data=fetch_data)
        mut_records.extend(batch_muts)

print(f"Loaded {len(mut_records)} total mutation events.")

# Process mutations
sample_muts = {g: {'Low': set(), 'Medium': set(), 'High': set()} for g in entrez_gene_ids}
sample_mut_types = {g: {} for g in entrez_gene_ids}

def simplify_mut_type(mtype):
    m = str(mtype).lower()
    if 'missense' in m: return 'Missense'
    if 'nonsense' in m: return 'Nonsense'
    if 'frame_shift' in m or 'frameshift' in m: return 'Frame Shift'
    if 'splice' in m: return 'Splice Site'
    if 'in_frame' in m: return 'In Frame'
    return 'Other Mut'

for m in mut_records:
    gene_id = m['entrezGeneId']
    sid = m['sampleId']
    group_series = blca_df[blca_df['SAMPLE_ID'] == sid]['TMB_Group'].values
    if len(group_series) == 0:
        continue
    group = group_series[0]
    sample_muts[gene_id][group].add(sid)
    
    mtype = simplify_mut_type(m.get('mutationType', 'Unknown'))
    if sid not in sample_mut_types[gene_id]:
        sample_mut_types[gene_id][sid] = mtype
    else:
        if sample_mut_types[gene_id][sid] != mtype:
            sample_mut_types[gene_id][sid] = 'Multi Hit'

# Perform Fisher's Exact Tests
print("\n4. Performing Fisher's Exact Tests...")
low_vs_med_results = []
med_vs_high_results = []

for g in entrez_gene_ids:
    low_mut = len(sample_muts[g]['Low'])
    low_wt = len(low_samples) - low_mut
    med_mut = len(sample_muts[g]['Medium'])
    med_wt = len(med_samples) - med_mut
    odds, p_lm = stats.fisher_exact([[low_mut, low_wt], [med_mut, med_wt]])
    low_vs_med_results.append({
        'entrezGeneId': g,
        'Gene': entrez_gene_map[g],
        'Low_Freq': low_mut / len(low_samples) * 100,
        'Med_Freq': med_mut / len(med_samples) * 100,
        'Pval': p_lm
    })
    
    high_mut = len(sample_muts[g]['High'])
    high_wt = len(high_samples) - high_mut
    odds, p_mh = stats.fisher_exact([[med_mut, med_wt], [high_mut, high_wt]])
    med_vs_high_results.append({
        'entrezGeneId': g,
        'Gene': entrez_gene_map[g],
        'Med_Freq': med_mut / len(med_samples) * 100,
        'High_Freq': high_mut / len(high_samples) * 100,
        'Pval': p_mh
    })

df_lm = pd.DataFrame(low_vs_med_results).sort_values('Pval')
df_mh = pd.DataFrame(med_vs_high_results).sort_values('Pval')

df_lm_filtered = df_lm[(df_lm['Low_Freq'] >= 2.0) | (df_lm['Med_Freq'] >= 2.0)].head(30).copy()
df_lm_filtered = df_lm_filtered.sort_values(by='Med_Freq', ascending=False)

df_mh_filtered = df_mh[(df_mh['Med_Freq'] >= 2.0) | (df_mh['High_Freq'] >= 2.0)].head(30).copy()
df_mh_filtered = df_mh_filtered.sort_values(by='High_Freq', ascending=False)

# ======== 🎨 顶刊级配色方案 ========
color_map = {
    'WT': '#EAECEE',          # Light gray
    'Missense': '#3498DB',    # Missense: Sky Blue
    'Nonsense': '#E74C3C',    # Stop gained/Nonsense: Red
    'Frame Shift': '#2ECC71', # Frameshift: Green
    'Splice Site': '#F1C40F', # Splice Site: Yellow
    'In Frame': '#9B59B6',    # In Frame Indel: Purple
    'Multi Hit': '#111111',   # Multi Hit: Black
    'Other Mut': '#E67E22'    # Other: Orange
}

legend_handles = [
    plt.Rectangle((0, 0), 1, 1, facecolor='#3498DB', label='Missense variant'),
    plt.Rectangle((0, 0), 1, 1, facecolor='#E74C3C', label='Nonsense variant'),
    plt.Rectangle((0, 0), 1, 1, facecolor='#2ECC71', label='Frameshift variant'),
    plt.Rectangle((0, 0), 1, 1, facecolor='#F1C40F', label='Splice site variant'),
    plt.Rectangle((0, 0), 1, 1, facecolor='#9B59B6', label='In frame indel'),
    plt.Rectangle((0, 0), 1, 1, facecolor='#111111', label='Multi Hit')
]

def generate_panel_figure_premium(filename, df_genes, group_left, group_right, left_samples, right_samples, title_left, title_right, title_pval_col, panel_letter):
    # Set up master-class Figure with Gridspec
    fig = plt.figure(figsize=(22, 11), dpi=300, facecolor='white')
    
    # 2 rows (Top Barplot: Height=1.3, Main Heatmap: Height=5)
    # 5 columns (Heatmap Left: 7, Freq Left: 1.2, Heatmap Right: 7, Freq Right: 1.2, P-val: 1.2)
    gs = gridspec.GridSpec(2, 5, 
                           width_ratios=[7, 1.2, 7, 1.2, 1.2], 
                           height_ratios=[1.3, 5],
                           wspace=0.15, hspace=0.06,
                           left=0.08, right=0.94, bottom=0.12, top=0.90)
    
    # Create axes
    ax_top_l = fig.add_subplot(gs[0, 0])
    ax_main_l = fig.add_subplot(gs[1, 0])
    ax_freq_l = fig.add_subplot(gs[1, 1])
    
    ax_top_r = fig.add_subplot(gs[0, 2])
    ax_main_r = fig.add_subplot(gs[1, 2])
    ax_freq_r = fig.add_subplot(gs[1, 3])
    
    ax_p = fig.add_subplot(gs[1, 4])
    
    genes_list = df_genes['Gene'].tolist()
    gene_ids_list = df_genes['entrezGeneId'].tolist()
    
    # Sort samples for left and right groups based on gene mutations (waterfall order)
    grp_l = 'Medium' if group_left == 'Med' else group_left
    grp_r = 'Medium' if group_right == 'Med' else group_right
    
    # Waterfall sort left samples
    left_samples_mutated_counts = {s: sum(1 for g in gene_ids_list if s in sample_muts[g][grp_l]) for s in left_samples}
    left_samples_sorted = sorted(left_samples, key=lambda s: (-left_samples_mutated_counts[s], s))
    
    # Waterfall sort right samples
    right_samples_mutated_counts = {s: sum(1 for g in gene_ids_list if s in sample_muts[g][grp_r]) for s in right_samples}
    right_samples_sorted = sorted(right_samples, key=lambda s: (-right_samples_mutated_counts[s], s))
    
    # Get TMB / Mutation Load for Top Barplots
    tmb_l = [blca_df[blca_df['SAMPLE_ID'] == s]['TMB_NONSYNONYMOUS'].values[0] for s in left_samples_sorted]
    tmb_r = [blca_df[blca_df['SAMPLE_ID'] == s]['TMB_NONSYNONYMOUS'].values[0] for s in right_samples_sorted]
    
    # ==================== 1. TOP BARPLOTS (Mutation Load) ====================
    # Left Top Barplot
    ax_top_l.bar(np.arange(len(left_samples_sorted)), tmb_l, width=0.85, color='#34495E', edgecolor='none')
    ax_top_l.set_xlim(-0.5, len(left_samples_sorted) - 0.5)
    ax_top_l.set_ylabel('TMB', fontsize=9.5, fontweight='bold')
    ax_top_l.set_xticks([])
    ax_top_l.set_title(title_left, fontsize=12, fontweight='bold', pad=10)
    sns.despine(ax=ax_top_l, bottom=True)
    ax_top_l.tick_params(bottom=False, labelsize=8.5)
    
    # Right Top Barplot
    ax_top_r.bar(np.arange(len(right_samples_sorted)), tmb_r, width=0.85, color='#34495E', edgecolor='none')
    ax_top_r.set_xlim(-0.5, len(right_samples_sorted) - 0.5)
    ax_top_r.set_ylabel('TMB', fontsize=9.5, fontweight='bold')
    ax_top_r.set_xticks([])
    ax_top_r.set_title(title_right, fontsize=12, fontweight='bold', pad=10)
    sns.despine(ax=ax_top_r, bottom=True)
    ax_top_r.tick_params(bottom=False, labelsize=8.5)
    
    # ==================== 2. MAIN HEATMAPS ====================
    # Draw Matrix Left Group
    matrix_l = np.zeros((len(genes_list), len(left_samples_sorted)))
    color_matrix_l = np.full((len(genes_list), len(left_samples_sorted)), 'WT', dtype=object)
    for r_idx, gene_id in enumerate(gene_ids_list):
        for c_idx, s in enumerate(left_samples_sorted):
            if s in sample_muts[gene_id][grp_l]:
                mtype = sample_mut_types[gene_id].get(s, 'Other Mut')
                color_matrix_l[r_idx, c_idx] = mtype
                matrix_l[r_idx, c_idx] = 1.0
                
    # Draw Matrix Right Group
    matrix_r = np.zeros((len(genes_list), len(right_samples_sorted)))
    color_matrix_r = np.full((len(genes_list), len(right_samples_sorted)), 'WT', dtype=object)
    for r_idx, gene_id in enumerate(gene_ids_list):
        for c_idx, s in enumerate(right_samples_sorted):
            if s in sample_muts[gene_id][grp_r]:
                mtype = sample_mut_types[gene_id].get(s, 'Other Mut')
                color_matrix_r[r_idx, c_idx] = mtype
                matrix_r[r_idx, c_idx] = 1.0

    # 2.1 Plot Left Group Matrix
    # First, draw light gray background cells to form a clean continuous grid structure!
    for r_idx in range(len(genes_list)):
        for c_idx in range(len(left_samples_sorted)):
            # WT background rectangle
            bg_rect = plt.Rectangle((c_idx - 0.45, r_idx - 0.45), 0.90, 0.90, color='#EAECEE', edgecolor='none')
            ax_main_l.add_patch(bg_rect)
            
            # Mutation rectangle (if mutated, draw on top)
            mtype = color_matrix_l[r_idx, c_idx]
            if mtype != 'WT':
                color = color_map.get(mtype, '#3498DB')
                rect = plt.Rectangle((c_idx - 0.42, r_idx - 0.42), 0.84, 0.84, color=color, edgecolor='none')
                ax_main_l.add_patch(rect)
                
    ax_main_l.set_yticks([])
    ax_main_l.set_xticks([])
    ax_main_l.set_xlim(-0.5, len(left_samples_sorted) - 0.5)
    ax_main_l.set_ylim(-0.5, len(genes_list) - 0.5)
    ax_main_l.invert_yaxis()
    sns.despine(ax=ax_main_l, left=True, bottom=True)
    ax_main_l.tick_params(left=False, bottom=False)
    
    # Add both gene names and mutation frequencies perfectly aligned on the left
    for r_idx, gene in enumerate(genes_list):
        freq = df_genes[df_genes['Gene'] == gene].iloc[0][f'{group_left}_Freq']
        label_text = f"{gene}   {freq:.1f}%"
        ax_main_l.text(-1.5, r_idx, label_text, ha='right', va='center', fontsize=9.0, fontweight='bold')

    # 2.2 Plot Right Group Matrix
    # First, draw light gray background cells to form a clean continuous grid structure!
    for r_idx, gene in enumerate(genes_list):
        for c_idx in range(len(right_samples_sorted)):
            # WT background rectangle
            bg_rect = plt.Rectangle((c_idx - 0.45, r_idx - 0.45), 0.90, 0.90, color='#EAECEE', edgecolor='none')
            ax_main_r.add_patch(bg_rect)
            
            # Mutation rectangle (if mutated, draw on top)
            mtype = color_matrix_r[r_idx, c_idx]
            if mtype != 'WT':
                color = color_map.get(mtype, '#3498DB')
                rect = plt.Rectangle((c_idx - 0.42, r_idx - 0.42), 0.84, 0.84, color=color, edgecolor='none')
                ax_main_r.add_patch(rect)
                
    ax_main_r.set_yticks([])
    ax_main_r.set_xticks([])
    ax_main_r.set_xlim(-0.5, len(right_samples_sorted) - 0.5)
    ax_main_r.set_ylim(-0.5, len(genes_list) - 0.5)
    ax_main_r.invert_yaxis()
    sns.despine(ax=ax_main_r, left=True, bottom=True)
    ax_main_r.tick_params(left=False, bottom=False)

    # ==================== 3. SIDE FREQUENCY BARPLOTS ====================
    # 3.1 Left Group Side Barplot (Mutation Frequency Breakdown)
    left_freqs = [df_genes[df_genes['Gene'] == g].iloc[0][f'{group_left}_Freq'] for g in genes_list]
    ax_freq_l.barh(np.arange(len(genes_list)), left_freqs, color='#34495E', height=0.65, edgecolor='none')
    ax_freq_l.set_yticks([])
    ax_freq_l.set_ylim(-0.5, len(genes_list) - 0.5)
    ax_freq_l.invert_yaxis()
    ax_freq_l.set_xlabel('% Mutated', fontsize=8.5, fontweight='bold')
    ax_freq_l.set_xlim(0, max(max(left_freqs)*1.1, 5))
    sns.despine(ax=ax_freq_l)
    ax_freq_l.tick_params(labelsize=8.0)
    
    # 3.2 Right Group Side Barplot (Mutation Frequency Breakdown)
    right_freqs = [df_genes[df_genes['Gene'] == g].iloc[0][f'{group_right}_Freq'] for g in genes_list]
    ax_freq_r.barh(np.arange(len(genes_list)), right_freqs, color='#E64B35', height=0.65, edgecolor='none')
    ax_freq_r.set_yticks([])
    ax_freq_r.set_ylim(-0.5, len(genes_list) - 0.5)
    ax_freq_r.invert_yaxis()
    ax_freq_r.set_xlabel('% Mutated', fontsize=8.5, fontweight='bold')
    ax_freq_r.set_xlim(0, max(max(right_freqs)*1.1, 5))
    sns.despine(ax=ax_freq_r)
    ax_freq_r.tick_params(labelsize=8.0)

    # ==================== 4. P-VALUES COLUMN ====================
    ax_p.axis('off')
    ax_p.set_title(title_pval_col, fontsize=10, fontweight='bold', pad=10)
    for r_idx, gene in enumerate(genes_list):
        p_val = df_genes[df_genes['Gene'] == gene].iloc[0]['Pval']
        if p_val < 0.0001:
            p_text = f"{p_val:.2e}"
        else:
            p_text = f"{p_val:.4f}"
            
        # Draw text beautifully
        if p_val < 0.05:
            # Highlight significant genes
            ax_p.text(0.5, r_idx, f"{p_text} *", ha='center', va='center', fontsize=9.0, fontweight='bold', color='#C0392B')
        else:
            ax_p.text(0.5, r_idx, p_text, ha='center', va='center', fontsize=8.5, color='#7F8C8D')
    ax_p.set_ylim(-0.5, len(genes_list) - 0.5)
    ax_p.invert_yaxis()

    # ==================== 5. LEGENDS ====================
    fig.legend(handles=legend_handles, loc='lower center', bbox_to_anchor=(0.5, 0.02), ncol=6, frameon=True, fontsize=10, facecolor='white', edgecolor='#BDC3C7')
    
    fig.text(0.02, 0.94, panel_letter, fontsize=24, fontweight='bold')
    
    plt.savefig(filename, bbox_inches='tight', dpi=300, transparent=False, facecolor='white')
    plt.close()
    print(f"Saved Premium Panel {panel_letter} to {filename}")

# Generate Panel A: Low vs Med
print("\n5. Generating Figure 3A Premium (Low vs Med TMB comparative oncoplot)...")
generate_panel_figure_premium(
    'bladder_co_oncoplot_panel_A.png', df_lm_filtered, 'Low', 'Med', low_samples, med_samples,
    f'TMB Low group (N={len(low_samples)})', f'TMB Medium group (N={len(med_samples)})', 'Low vs. Medium\nP value', 'A'
)

# Generate Panel B: Med vs High
print("6. Generating Figure 3B Premium (Med vs High TMB comparative oncoplot)...")
generate_panel_figure_premium(
    'bladder_co_oncoplot_panel_B.png', df_mh_filtered, 'Med', 'High', med_samples, high_samples,
    f'TMB Medium group (N={len(med_samples)})', f'TMB High group (N={len(high_samples)})', 'Medium vs. High\nP value', 'B'
)

# Copy to brain folder for UI visualization
brain_dir = r'[LOCAL_ABSOLUTE_PATH]'
if os.path.exists(brain_dir):
    shutil.copy('bladder_co_oncoplot_panel_A.png', os.path.join(brain_dir, 'bladder_co_oncoplot_panel_A.png'))
    shutil.copy('bladder_co_oncoplot_panel_B.png', os.path.join(brain_dir, 'bladder_co_oncoplot_panel_B.png'))
    print("Successfully copied plots to brain directory.")

print("\nSuccessfully updated clinical co-oncoplots to world-class premium level!")
