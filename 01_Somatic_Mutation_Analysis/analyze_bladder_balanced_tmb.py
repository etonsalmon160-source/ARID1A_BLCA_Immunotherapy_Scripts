import urllib.request
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

# ======== 顶刊级绘图配置 (Nature / Cell Style) ========
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['pdf.fonttype'] = 42

study_id = 'tmb_mskcc_2018'

print("1. Fetching Bladder Cancer clinical data...")
url_samples = f'https://www.cbioportal.org/api/studies/{study_id}/clinical-data?projection=DETAILED&clinicalDataType=SAMPLE'
req_s = urllib.request.Request(url_samples, headers={'Accept': 'application/json'})
with urllib.request.urlopen(req_s) as response:
    sample_data = json.loads(response.read().decode('utf-8'))

url_patients = f'https://www.cbioportal.org/api/studies/{study_id}/clinical-data?projection=DETAILED&clinicalDataType=PATIENT'
req_p = urllib.request.Request(url_patients, headers={'Accept': 'application/json'})
with urllib.request.urlopen(req_p) as response:
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
cancer_df = df[df['CANCER_TYPE'] == 'Bladder Cancer'].copy()

# Parse status
def parse_status(status_str):
    if pd.isna(status_str): return np.nan
    s = str(status_str).upper()
    if '1' in s or 'DECEASED' in s or 'DEAD' in s: return 1
    if '0' in s or 'LIVING' in s or 'ALIVE' in s: return 0
    return np.nan

cancer_df['EVENT'] = cancer_df['OS_STATUS'].apply(parse_status)
cancer_df = cancer_df.dropna(subset=['OS_MONTHS', 'EVENT', 'TMB_NONSYNONYMOUS'])

# Fetch mutations
sample_ids = cancer_df['SAMPLE_ID'].tolist()
molecular_profile_id = f'{study_id}_mutations'
url_mutations = f'https://www.cbioportal.org/api/molecular-profiles/{molecular_profile_id}/mutations/fetch'
fetch_data = {
    "entrezGeneIds": [8289],
    "sampleIds": sample_ids
}
req = urllib.request.Request(url_mutations, data=json.dumps(fetch_data).encode('utf-8'), method='POST')
req.add_header('Content-Type', 'application/json')
req.add_header('Accept', 'application/json')

with urllib.request.urlopen(req) as response:
    mut_data = json.loads(response.read().decode('utf-8'))

mutated_samples = set([m['sampleId'] for m in mut_data])
cancer_df['ARID1A_Status'] = cancer_df['SAMPLE_ID'].apply(lambda x: 'Mutant' if x in mutated_samples else 'Wild-Type')

mut_df = cancer_df[cancer_df['ARID1A_Status'] == 'Mutant'].copy()
wt_df = cancer_df[cancer_df['ARID1A_Status'] == 'Wild-Type'].copy()

# Caliper-based matching (caliper = 0.2 * std of TMB in the entire cohort)
caliper = 0.2 * cancer_df['TMB_NONSYNONYMOUS'].std()

matched_pairs = []
wt_pool = wt_df.copy()

for idx, mut_row in mut_df.iterrows():
    mut_tmb = mut_row['TMB_NONSYNONYMOUS']
    wt_pool['TMB_diff'] = (wt_pool['TMB_NONSYNONYMOUS'] - mut_tmb).abs()
    candidates = wt_pool[wt_pool['TMB_diff'] <= caliper]
    if len(candidates) > 0:
        nearest_idx = candidates['TMB_diff'].idxmin()
        matched_pairs.append((mut_row, wt_df.loc[nearest_idx]))
        wt_pool = wt_pool.drop(nearest_idx)

m_matched = pd.DataFrame([p[0] for p in matched_pairs])
w_matched = pd.DataFrame([p[1] for p in matched_pairs])

mut_mean_tmb = m_matched['TMB_NONSYNONYMOUS'].mean()
wt_mean_tmb = w_matched['TMB_NONSYNONYMOUS'].mean()

res = logrank_test(w_matched['OS_MONTHS'], m_matched['OS_MONTHS'], event_observed_A=w_matched['EVENT'], event_observed_B=m_matched['EVENT'])
pval = res.p_value

mut_med_os = m_matched['OS_MONTHS'].median()
wt_med_os = w_matched['OS_MONTHS'].median()

print(f"\n--- TMB-MATCHED COHORT STATISTICS ---")
print(f"Matched pairs: {len(m_matched)}")
print(f"Mutant Mean TMB: {mut_mean_tmb:.2f}, Median OS: {mut_med_os:.1f} months")
print(f"Wild-Type Mean TMB: {wt_mean_tmb:.2f}, Median OS: {wt_med_os:.1f} months")
print(f"Log-rank P-value: {pval:.5f}")

# ======== Plotting matched KM ========
fig, ax = plt.subplots(figsize=(8, 6), dpi=300, facecolor='white')

kmf_wt = KaplanMeierFitter()
kmf_mut = KaplanMeierFitter()

kmf_wt.fit(w_matched['OS_MONTHS'], w_matched['EVENT'], label=f'ARID1A Wild-Type (Matched, N={len(w_matched)}, mOS={wt_med_os:.1f} mo)')
kmf_wt.plot_survival_function(ax=ax, color='#4DBBD5', linewidth=2.5, ci_show=False)

kmf_mut.fit(m_matched['OS_MONTHS'], m_matched['EVENT'], label=f'ARID1A Mutant (N={len(m_matched)}, mOS={mut_med_os:.1f} mo)')
kmf_mut.plot_survival_function(ax=ax, color='#E64B35', linewidth=2.5, ci_show=False)

# Title & Annotations
ax.text(0.05, 0.08, f"Log-rank P = {pval:.4f}", transform=ax.transAxes, fontsize=12, fontweight='bold', color='#333333')

plt.title('Overall Survival after Immunotherapy in Bladder Cancer\nPropensity-Matched Cohort (1:1 TMB-Balanced)', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Overall Survival (Months)', fontsize=12, fontweight='bold')
plt.ylabel('Survival Probability', fontsize=12, fontweight='bold')
plt.ylim(0, 1.05)
plt.xlim(0, max(cancer_df['OS_MONTHS'].max() + 5, 20))

plt.legend(loc='upper right', frameon=False, fontsize=11)
sns.despine(top=True, right=True)
plt.grid(axis='y', linestyle=':', color='#DDDDDD')

output_path = '[YOUR_WORKING_DIRECTORY]\\bladder_balanced_tmb_km_plot.png'
plt.tight_layout()
plt.savefig(output_path, transparent=False, facecolor='white')
print(f"Saved balanced TMB KM plot to {output_path}")
