import urllib.request
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ======== 🎨 顶刊级绘图配置 ========
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['pdf.fonttype'] = 42

study_id = 'tmb_mskcc_2018'
target_gene = 8289  # ARID1A

print("1. Fetching Bladder Cancer clinical data...")
url_samples = f'https://www.cbioportal.org/api/studies/{study_id}/clinical-data?projection=DETAILED&clinicalDataType=SAMPLE'
req = urllib.request.Request(url_samples, headers={'Accept': 'application/json'})
with urllib.request.urlopen(req) as response:
    sample_data = json.loads(response.read().decode('utf-8'))

# Reconstruct sample attributes
samples = {}
for item in sample_data:
    sid = item.get('sampleId')
    if sid not in samples:
        samples[sid] = {'SAMPLE_ID': sid}
    samples[sid][item.get('clinicalAttributeId')] = item.get('value')
df_samples = pd.DataFrame(list(samples.values()))

# Filter Bladder Cancer samples with nonsynonymous TMB
df_samples['TMB_NONSYNONYMOUS'] = pd.to_numeric(df_samples.get('TMB_NONSYNONYMOUS', np.nan), errors='coerce')
blca_df = df_samples[df_samples['CANCER_TYPE'] == 'Bladder Cancer'].dropna(subset=['TMB_NONSYNONYMOUS']).copy()

# Calculate Tertile Splitting Cutoffs
q33 = blca_df['TMB_NONSYNONYMOUS'].quantile(1/3)
q67 = blca_df['TMB_NONSYNONYMOUS'].quantile(2/3)
print(f"TMB Cutoffs: 33.3% = {q33:.2f}, 66.7% = {q67:.2f}")

def get_tmb_group(tmb):
    if tmb < q33: return 'Low'
    elif tmb < q67: return 'Medium'
    else: return 'High'

blca_df['TMB_Group'] = blca_df['TMB_NONSYNONYMOUS'].apply(get_tmb_group)
sample_ids = blca_df['SAMPLE_ID'].tolist()

print("2. Fetching somatic mutations of ARID1A...")
url_mutations = f'https://www.cbioportal.org/api/molecular-profiles/{study_id}_mutations/mutations/fetch'
fetch_data = {
    "entrezGeneIds": [target_gene],
    "sampleIds": sample_ids
}
req = urllib.request.Request(url_mutations, data=json.dumps(fetch_data).encode('utf-8'), method='POST')
req.add_header('Content-Type', 'application/json')
req.add_header('Accept', 'application/json')
with urllib.request.urlopen(req) as response:
    mut_data = json.loads(response.read().decode('utf-8'))

# Map each mutant sample to its mutation type category
# paper's categories: Deletion, Missense, Stop gained, Frameshift
mut_types = {}
for m in mut_data:
    sid = m['sampleId']
    mtype = m.get('mutationType', 'Missense_Mutation')
    
    if 'Missense' in mtype:
        cat = 'Missense'
    elif 'Nonsense' in mtype or 'Splice' in mtype:
        cat = 'Stop gained'
    elif 'Frame_Shift' in mtype or 'Frameshift' in mtype:
        cat = 'Frameshift'
    elif 'In_Frame' in mtype or 'Deletion' in mtype:
        cat = 'Deletion'
    else:
        cat = 'Missense' # fallback
    mut_types[sid] = cat

blca_df['ARID1A_Mut'] = blca_df['SAMPLE_ID'].apply(lambda x: 1 if x in mut_types else 0)
blca_df['Mutation_Type'] = blca_df['SAMPLE_ID'].map(mut_types)

# Compute Panel C: Mutation frequency by TMB group
prop_df = blca_df.groupby('TMB_Group')['ARID1A_Mut'].mean().reset_index()
prop_df['ARID1A_Mut'] = prop_df['ARID1A_Mut'] * 100
# Reorder groups
prop_df['TMB_Group'] = pd.Categorical(prop_df['TMB_Group'], categories=['Low', 'Medium', 'High'], ordered=True)
prop_df = prop_df.sort_values('TMB_Group')

# Compute Panel D: Mutation type proportions in Medium and High groups
type_counts = blca_df[blca_df['TMB_Group'].isin(['Medium', 'High']) & (blca_df['ARID1A_Mut'] == 1)].groupby(['TMB_Group', 'Mutation_Type']).size().unstack(fill_value=0)
type_props = type_counts.div(type_counts.sum(axis=1), axis=0) * 100
type_props = type_props.reindex(['Medium', 'High'])

# Ensure all 4 categories are present for consistent plotting
for col in ['Deletion', 'Missense', 'Stop gained', 'Frameshift']:
    if col not in type_props.columns:
        type_props[col] = 0.0

# Arrange columns in consistent order
type_props = type_props[['Deletion', 'Missense', 'Stop gained', 'Frameshift']]

print("\n--- Calculated Stats for Plotting ---")
print("Panel C: Mutation Frequency (%) by TMB Group:")
print(prop_df)
print("\nPanel D: Mutation Type Breakdown (%) in Medium vs High TMB:")
print(type_props)

# ======== 🎨 Plotting Panel C & Panel D Side-by-Side ========
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300, facecolor='white')

# --- Panel C: Mutation Frequency ---
# Left Plot
ax_c = axes[0]
c_colors = ['#8491B4', '#4DBBD5', '#E64B35']  # Low (gray-blue), Medium (light blue), High (red-orange)
bars_c = ax_c.bar(prop_df['TMB_Group'], prop_df['ARID1A_Mut'], color=c_colors, width=0.45, edgecolor='black', linewidth=0.8)

# Add value labels on top of bars
for bar in bars_c:
    height = bar.get_height()
    ax_c.text(bar.get_x() + bar.get_width()/2., height + 1.0, f'{height:.1f}%', ha='center', va='bottom', fontsize=9.5, fontweight='bold')

ax_c.set_title('ARID1A Mutation Frequency by TMB', fontsize=12, fontweight='bold', pad=15)
ax_c.set_ylabel('Mutation frequency (%)', fontsize=11, fontweight='bold')
ax_c.set_ylim(0, 60)
ax_c.tick_params(axis='both', which='major', labelsize=10.5)
ax_c.set_xlabel('TMB Status', fontsize=11, fontweight='bold')
sns.despine(ax=ax_c)
ax_c.grid(axis='y', linestyle=':', color='#DDDDDD', alpha=0.7)

# Add "C" panel identifier
ax_c.text(-0.25, 1.05, 'C', transform=ax_c.transAxes, fontsize=20, fontweight='bold', va='top', ha='right')


# --- Panel D: Mutation Type Frequency ---
# Right Plot (Stacked Bar Plot)
ax_d = axes[1]
d_colors = ['#2CA02C', '#9467BD', '#1F77B4', '#E64B35'] # Deletion (green), Missense (magenta), Stop gained (blue), Frameshift (coral)
categories = ['Deletion', 'Missense', 'Stop gained', 'Frameshift']

bottoms = np.zeros(2)
x_labels = ['Medium', 'High']

for idx, cat in enumerate(categories):
    vals = type_props[cat].values
    ax_d.bar(x_labels, vals, bottom=bottoms, label=cat, color=d_colors[idx], width=0.35, edgecolor='black', linewidth=0.8)
    bottoms += vals

ax_d.set_title('ARID1A Mutation Types by TMB', fontsize=12, fontweight='bold', pad=15)
ax_d.set_ylabel('Frequency (%)', fontsize=11, fontweight='bold')
ax_d.set_ylim(0, 105)
ax_d.set_xlabel('TMB Status', fontsize=11, fontweight='bold')
ax_d.tick_params(axis='both', which='major', labelsize=10.5)
ax_d.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=10)
sns.despine(ax=ax_d)
ax_d.grid(axis='y', linestyle=':', color='#DDDDDD', alpha=0.7)

# Add "D" panel identifier
ax_d.text(-0.2, 1.05, 'D', transform=ax_d.transAxes, fontsize=20, fontweight='bold', va='top', ha='right')

plt.tight_layout()

# Save paths
output_path = r'[YOUR_WORKING_DIRECTORY]\arid1a_mutation_tmb_panels_C_D.png'
plt.savefig(output_path, facecolor='white', edgecolor='none', transparent=False, bbox_inches='tight')
plt.close()
print(f"Successfully generated and saved Panels C & D to {output_path}")
