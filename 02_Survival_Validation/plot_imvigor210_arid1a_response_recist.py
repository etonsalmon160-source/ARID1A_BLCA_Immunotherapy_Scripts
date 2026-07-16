import urllib.request
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ======== 🎨 顶刊级绘图配置 ========
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['pdf.fonttype'] = 42

study_id = 'blca_iatlas_imvigor210_2017'
target_gene = 8289  # ARID1A

print("1. Fetching clinical data for IMvigor210...")
# Fetch sample-level clinical attributes (like TMB, RESPONSE)
url_samples = f'https://www.cbioportal.org/api/studies/{study_id}/clinical-data?projection=DETAILED&clinicalDataType=SAMPLE'
req = urllib.request.Request(url_samples, headers={'Accept': 'application/json'})
with urllib.request.urlopen(req) as r:
    samples = json.loads(r.read().decode('utf-8'))
s_dict = {}
for s in samples:
    sid = s['sampleId']
    if sid not in s_dict: 
        s_dict[sid] = {'SAMPLE_ID': sid, 'PATIENT_ID': s.get('patientId')}
    s_dict[sid][s['clinicalAttributeId']] = s['value']
df_s = pd.DataFrame(list(s_dict.values()))

# Fetch patient-level clinical attributes (like OS_MONTHS, OS_STATUS)
url_patients = f'https://www.cbioportal.org/api/studies/{study_id}/clinical-data?projection=DETAILED&clinicalDataType=PATIENT'
req = urllib.request.Request(url_patients, headers={'Accept': 'application/json'})
with urllib.request.urlopen(req) as r:
    patients = json.loads(r.read().decode('utf-8'))
p_dict = {}
for p in patients:
    pid = p['patientId']
    if pid not in p_dict: 
        p_dict[pid] = {'PATIENT_ID': pid}
    p_dict[pid][p['clinicalAttributeId']] = p['value']
df_p = pd.DataFrame(list(p_dict.values()))

# Merge sample and patient data
df = pd.merge(df_s, df_p, on='PATIENT_ID', how='inner')
df['TMB_NONSYNONYMOUS'] = pd.to_numeric(df['TMB_NONSYNONYMOUS'], errors='coerce')

# Fetch mutations for ARID1A
s_ids = df['SAMPLE_ID'].tolist()
url_mut = f'https://www.cbioportal.org/api/molecular-profiles/{study_id}_mutations/mutations/fetch'
fetch_data = {'entrezGeneIds': [target_gene], 'sampleIds': s_ids}
req = urllib.request.Request(url_mut, data=json.dumps(fetch_data).encode('utf-8'), method='POST')
req.add_header('Content-Type', 'application/json')
req.add_header('Accept', 'application/json')
with urllib.request.urlopen(req) as r:
    muts = json.loads(r.read().decode('utf-8'))

mut_samples = set([m['sampleId'] for m in muts])
df['ARID1A_Status'] = df['SAMPLE_ID'].apply(lambda x: 'Mutant' if x in mut_samples else 'Wild-Type')

# Map RESPONSE to RECIST Group (CR/PR vs SD/PD vs NA)
def get_recist_group(val):
    if pd.isna(val): return 'NA'
    val = str(val).strip()
    if val in ['Complete Response', 'Partial Response']:
        return 'CR/PR'
    elif val in ['Stable Disease', 'Progressive Disease']:
        return 'SD/PD'
    else:
        return 'NA'

df['RECIST_Group'] = df['RESPONSE'].apply(get_recist_group)

# Generate Crosstab and percentages
# 1. Whole Cohort
ct_all = pd.crosstab(df['ARID1A_Status'], df['RECIST_Group'])
# Fill missing categories
for col in ['CR/PR', 'SD/PD', 'NA']:
    if col not in ct_all.columns:
        ct_all[col] = 0
ct_all = ct_all[['CR/PR', 'SD/PD', 'NA']]
ct_all_pct = ct_all.div(ct_all.sum(axis=1), axis=0) * 100
# Reorder index to Wt and Mutation
ct_all_pct = ct_all_pct.reindex(['Wild-Type', 'Mutant'])
ct_all = ct_all.reindex(['Wild-Type', 'Mutant'])

# 2. High-TMB Cohort (Top 33%)
tmb_threshold = df['TMB_NONSYNONYMOUS'].quantile(0.67)
high_tmb_df = df[df['TMB_NONSYNONYMOUS'] >= tmb_threshold].copy()
ct_high = pd.crosstab(high_tmb_df['ARID1A_Status'], high_tmb_df['RECIST_Group'])
for col in ['CR/PR', 'SD/PD', 'NA']:
    if col not in ct_high.columns:
        ct_high[col] = 0
ct_high = ct_high[['CR/PR', 'SD/PD', 'NA']]
ct_high_pct = ct_high.div(ct_high.sum(axis=1), axis=0) * 100
ct_high_pct = ct_high_pct.reindex(['Wild-Type', 'Mutant'])
ct_high = ct_high.reindex(['Wild-Type', 'Mutant'])

print("Whole Cohort Percentages:\n", ct_all_pct)
print("High TMB Cohort Percentages:\n", ct_high_pct)

# ======== 🎨 Plotting stacked bar plots ========
fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=300, facecolor='white')
colors = ['#1F77B4', '#E64B35', '#A6ACAF'] # Blue for CR/PR, Red for SD/PD, Gray for NA
categories = ['CR/PR', 'SD/PD', 'NA']

# Left Plot: Whole Cohort
ax1 = axes[0]
bottoms1 = np.zeros(2)
x_labels = ['Wt\n(N=283)', 'Mutation\n(N=64)']
for idx, cat in enumerate(categories):
    vals = ct_all_pct[cat].values
    ax1.bar(x_labels, vals, bottom=bottoms1, label=cat, color=colors[idx], width=0.45, edgecolor='black', linewidth=0.8)
    # Add labels on bars
    for i, v in enumerate(vals):
        if v > 5.0: # Only plot label if percentage > 5% for readability
            ax1.text(i, bottoms1[i] + v/2.0, f'{v:.1f}%', ha='center', va='center', color='white', fontweight='bold', fontsize=9.5)
    bottoms1 += vals

ax1.set_ylabel('Frequency (%)', fontsize=11, fontweight='bold')
ax1.set_ylim(0, 105)
ax1.set_title('Immunotherapy Response Rate (RECIST)\nWhole IMvigor210 Cohort', fontsize=12, fontweight='bold', pad=15)
ax1.tick_params(axis='both', which='major', labelsize=10.5)
sns.despine(ax=ax1)
ax1.grid(axis='y', linestyle=':', color='#DDDDDD', alpha=0.7)

# Right Plot: High TMB Cohort
ax2 = axes[1]
bottoms2 = np.zeros(2)
x_labels_high = ['Wt\n(N=48)', 'Mutation\n(N=32)']
for idx, cat in enumerate(categories):
    vals = ct_high_pct[cat].values
    ax2.bar(x_labels_high, vals, bottom=bottoms2, label=cat, color=colors[idx], width=0.45, edgecolor='black', linewidth=0.8)
    for i, v in enumerate(vals):
        if v > 5.0:
            ax2.text(i, bottoms2[i] + v/2.0, f'{v:.1f}%', ha='center', va='center', color='white', fontweight='bold', fontsize=9.5)
    bottoms2 += vals

ax2.set_ylabel('Frequency (%)', fontsize=11, fontweight='bold')
ax2.set_ylim(0, 105)
ax2.set_title('Immunotherapy Response Rate (RECIST)\nHigh-TMB Cohort', fontsize=12, fontweight='bold', pad=15)
ax2.tick_params(axis='both', which='major', labelsize=10.5)
ax2.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=10)
sns.despine(ax=ax2)
ax2.grid(axis='y', linestyle=':', color='#DDDDDD', alpha=0.7)

plt.tight_layout()

# Save paths
output_path = r'[YOUR_WORKING_DIRECTORY]\imvigor210_arid1a_response_recist_panels.png'
plt.savefig(output_path, facecolor='white', edgecolor='none', transparent=False, bbox_inches='tight')
plt.close()
print(f"Successfully generated and saved RECIST response panels to {output_path}")
