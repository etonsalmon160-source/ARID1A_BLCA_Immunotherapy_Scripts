import urllib.request
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

# ======== 顶刊级绘图配置 ========
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['pdf.fonttype'] = 42

study_id = 'blca_iatlas_imvigor210_2017'

print(f"1. Fetching {study_id} (IMvigor210) clinical data...")
url_samples = f'https://www.cbioportal.org/api/studies/{study_id}/clinical-data?projection=DETAILED&clinicalDataType=SAMPLE'
req = urllib.request.Request(url_samples, headers={'Accept': 'application/json'})
with urllib.request.urlopen(req) as response:
    sample_data = json.loads(response.read().decode('utf-8'))

url_patients = f'https://www.cbioportal.org/api/studies/{study_id}/clinical-data?projection=DETAILED&clinicalDataType=PATIENT'
req = urllib.request.Request(url_patients, headers={'Accept': 'application/json'})
with urllib.request.urlopen(req) as response:
    patient_data = json.loads(response.read().decode('utf-8'))

samples = {}
for item in sample_data:
    sid = item.get('sampleId')
    if sid not in samples:
        samples[sid] = {'SAMPLE_ID': sid, 'PATIENT_ID': item.get('patientId')}
    samples[sid][item.get('clinicalAttributeId')] = item.get('value')
df_samples = pd.DataFrame(list(samples.values()))

patients = {}
for item in patient_data:
    pid = item.get('patientId')
    if pid not in patients:
        patients[pid] = {'PATIENT_ID': pid}
    patients[pid][item.get('clinicalAttributeId')] = item.get('value')
df_patients = pd.DataFrame(list(patients.values()))

df = pd.merge(df_samples, df_patients, on='PATIENT_ID', how='inner')
df['TMB_NONSYNONYMOUS'] = pd.to_numeric(df.get('TMB_NONSYNONYMOUS', np.nan), errors='coerce')
df['OS_MONTHS'] = pd.to_numeric(df.get('OS_MONTHS', np.nan), errors='coerce')

# Status: 0 = censored, 1 = dead
def parse_status(status_str):
    if pd.isna(status_str): return np.nan
    s = str(status_str).upper()
    if '1' in s or 'DECEASED' in s or 'DEAD' in s: return 1
    if '0' in s or 'LIVING' in s or 'ALIVE' in s: return 0
    return np.nan

df['EVENT'] = df['OS_STATUS'].apply(parse_status)

# Drop missing TMB
df = df.dropna(subset=['TMB_NONSYNONYMOUS', 'OS_MONTHS'])

# Determine High TMB (Top 33%)
tmb_threshold = df['TMB_NONSYNONYMOUS'].quantile(0.67)
high_tmb_df = df[df['TMB_NONSYNONYMOUS'] >= tmb_threshold].copy()
high_tmb_sample_ids = high_tmb_df['SAMPLE_ID'].tolist()

print(f"IMvigor210 High TMB (>= {tmb_threshold:.1f}) cohort size: {len(high_tmb_df)}")

print("2. Fetching ARID1A mutations...")
url_mutations = f'https://www.cbioportal.org/api/molecular-profiles/{study_id}_mutations/mutations/fetch'
fetch_data = {"entrezGeneIds": [8289], "sampleIds": high_tmb_sample_ids}
req = urllib.request.Request(url_mutations, data=json.dumps(fetch_data).encode('utf-8'), method='POST')
req.add_header('Content-Type', 'application/json')
req.add_header('Accept', 'application/json')
try:
    with urllib.request.urlopen(req) as response:
        mut_data = json.loads(response.read().decode('utf-8'))
except Exception as e:
    print("Error fetching mutations:", e)
    mut_data = []

mutated_samples = set([m['sampleId'] for m in mut_data])
print(f"Found {len(mutated_samples)} High TMB samples with ARID1A mutation.")

high_tmb_df['ARID1A_Status'] = high_tmb_df['SAMPLE_ID'].apply(lambda x: 'Mutant' if x in mutated_samples else 'Wild-Type')

mut_df = high_tmb_df[high_tmb_df['ARID1A_Status'] == 'Mutant']
wt_df = high_tmb_df[high_tmb_df['ARID1A_Status'] == 'Wild-Type']

# Plotting
fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300, facecolor='white')

# --- Panel 1: Kaplan-Meier Survival ---
ax1 = axes[0]
kmf_wt = KaplanMeierFitter()
kmf_mut = KaplanMeierFitter()

if len(wt_df) > 0:
    kmf_wt.fit(wt_df['OS_MONTHS'], wt_df['EVENT'], label=f"ARID1A Wild-Type (N={len(wt_df)}, mOS={wt_df['OS_MONTHS'].median():.1f} mo)")
    kmf_wt.plot_survival_function(ax=ax1, color='#4DBBD5', linewidth=2.5, ci_show=False)

if len(mut_df) > 0:
    kmf_mut.fit(mut_df['OS_MONTHS'], mut_df['EVENT'], label=f"ARID1A Mutant (N={len(mut_df)}, mOS={mut_df['OS_MONTHS'].median():.1f} mo)")
    kmf_mut.plot_survival_function(ax=ax1, color='#E64B35', linewidth=2.5, ci_show=False)

if len(wt_df) > 0 and len(mut_df) > 0:
    res = logrank_test(wt_df['OS_MONTHS'], mut_df['OS_MONTHS'], event_observed_A=wt_df['EVENT'], event_observed_B=mut_df['EVENT'])
    ax1.text(0.05, 0.05, f"Log-rank P = {res.p_value:.3f}", transform=ax1.transAxes, fontsize=12, fontweight='bold')

ax1.set_title('IMvigor210 Cohort (Anti-PD-L1)\nSurvival of High-TMB Patients', fontsize=14, fontweight='bold', pad=15)
ax1.set_xlabel('Overall Survival (Months)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Survival Probability', fontsize=12, fontweight='bold')
ax1.set_ylim(0, 1.05)
sns.despine(ax=ax1)

# --- Panel 2: Response Rate ---
ax2 = axes[1]
# Assume RESPONSE or RESPONDER column exists
response_col = 'RESPONDER' if 'RESPONDER' in high_tmb_df.columns else ('RESPONSE' if 'RESPONSE' in high_tmb_df.columns else None)

if response_col and not high_tmb_df[response_col].isna().all():
    # Map to Responder (True/False)
    def is_responder(val):
        if pd.isna(val): return np.nan
        val = str(val).upper()
        if val in ['CR', 'PR', 'TRUE', 'YES', 'RESPONDER']: return 'Responder'
        if val in ['SD', 'PD', 'FALSE', 'NO', 'NON-RESPONDER']: return 'Non-Responder'
        return np.nan
        
    high_tmb_df['Response_Binary'] = high_tmb_df[response_col].apply(is_responder)
    plot_df = high_tmb_df.dropna(subset=['Response_Binary'])
    
    if len(plot_df) > 0:
        agg_df = plot_df.groupby(['ARID1A_Status', 'Response_Binary']).size().unstack().fillna(0)
        # Convert to percentages
        agg_df_pct = agg_df.div(agg_df.sum(axis=1), axis=0) * 100
        
        # Ensure correct column order
        cols = []
        if 'Non-Responder' in agg_df_pct.columns: cols.append('Non-Responder')
        if 'Responder' in agg_df_pct.columns: cols.append('Responder')
        agg_df_pct = agg_df_pct[cols]
        
        colors = ['#CCCCCC', '#4DBBD5'] # Gray for Non-Responder, Blue for Responder
        agg_df_pct.plot(kind='bar', stacked=True, ax=ax2, color=colors, edgecolor='white', width=0.6)
        
        ax2.set_title('Immunotherapy Response Rate (ORR)', fontsize=14, fontweight='bold', pad=15)
        ax2.set_ylabel('Percentage of Patients (%)', fontsize=12, fontweight='bold')
        ax2.set_xlabel('')
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=0, fontweight='bold')
        ax2.legend(loc='upper right', bbox_to_anchor=(1.2, 1), frameon=False)
        sns.despine(ax=ax2)
        
        # Add text
        for i, row in enumerate(agg_df_pct.index):
            if 'Responder' in agg_df_pct.columns:
                val = agg_df_pct.loc[row, 'Responder']
                ax2.text(i, 100 - val/2, f"{val:.1f}%", ha='center', va='center', color='white', fontweight='bold')
else:
    ax2.text(0.5, 0.5, "Response data not available", ha='center', va='center')
    ax2.axis('off')

plt.tight_layout()
output_path = 'imvigor210_arid1a_validation.png'
plt.savefig(output_path, transparent=False, facecolor='white', bbox_inches='tight')
print(f"Saved plot to {output_path}")
