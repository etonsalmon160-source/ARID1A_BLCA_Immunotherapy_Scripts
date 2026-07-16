import urllib.request
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ======== 顶刊级绘图配置 (Nature / Cell Style) ========
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['pdf.fonttype'] = 42 # 保证能在AI中编辑

print("1. Fetching Bladder Cancer clinical data...")
url_samples = 'https://www.cbioportal.org/api/studies/tmb_mskcc_2018/clinical-data?projection=DETAILED&clinicalDataType=SAMPLE'
req = urllib.request.Request(url_samples, headers={'Accept': 'application/json'})
with urllib.request.urlopen(req) as response:
    sample_data = json.loads(response.read().decode('utf-8'))

url_patients = 'https://www.cbioportal.org/api/studies/tmb_mskcc_2018/clinical-data?projection=DETAILED&clinicalDataType=PATIENT'
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

# Filter for Bladder Cancer
target_cancer = 'Bladder Cancer'
cancer_df = df[df['CANCER_TYPE'] == target_cancer].copy()

# Determine High TMB (Top 20%)
tmb_threshold = cancer_df['TMB_NONSYNONYMOUS'].quantile(0.8)
high_tmb_df = cancer_df[cancer_df['TMB_NONSYNONYMOUS'] >= tmb_threshold].copy()
high_tmb_sample_ids = high_tmb_df['SAMPLE_ID'].tolist()

print(f"Bladder Cancer High TMB (>={tmb_threshold:.1f}) cohort size: {len(high_tmb_df)}")

print("2. Fetching ARID1A mutations for High TMB Bladder Cancer cohort...")
# Gene ID for ARID1A is 8289
molecular_profile_id = 'tmb_mskcc_2018_mutations'
url_mutations = f'https://www.cbioportal.org/api/molecular-profiles/{molecular_profile_id}/mutations/fetch'
fetch_data = {
    "entrezGeneIds": [8289],
    "sampleIds": high_tmb_sample_ids
}
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

high_tmb_df['Gene_Status'] = high_tmb_df['SAMPLE_ID'].apply(lambda x: 'Mutant' if x in mutated_samples else 'Wild-Type')

# Status: 0 = censored, 1 = dead
def parse_status(status_str):
    if pd.isna(status_str): return np.nan
    s = str(status_str).upper()
    if '1' in s or 'DECEASED' in s or 'DEAD' in s: return 1
    if '0' in s or 'LIVING' in s or 'ALIVE' in s: return 0
    return np.nan

high_tmb_df['EVENT'] = high_tmb_df['OS_STATUS'].apply(parse_status)

# Simple Kaplan-Meier estimator
def kaplan_meier(df, time_col='OS_MONTHS', event_col='EVENT'):
    df_sorted = df.dropna(subset=[time_col, event_col]).sort_values(time_col)
    times = df_sorted[time_col].values
    events = df_sorted[event_col].values
    
    n_at_risk = len(times)
    survival_prob = 1.0
    
    km_times = [0.0]
    km_survival = [1.0]
    
    for t, e in zip(times, events):
        if e == 1:
            survival_prob = survival_prob * (1 - 1/n_at_risk)
        n_at_risk -= 1
        km_times.append(t)
        km_survival.append(survival_prob)
        
    return km_times, km_survival

mut_df = high_tmb_df[high_tmb_df['Gene_Status'] == 'Mutant']
wt_df = high_tmb_df[high_tmb_df['Gene_Status'] == 'Wild-Type']

mut_med_os = mut_df['OS_MONTHS'].median()
wt_med_os = wt_df['OS_MONTHS'].median()

print(f"High TMB + ARID1A Wild-Type: Median OS = {wt_med_os:.1f} months (N={len(wt_df)})")
print(f"High TMB + ARID1A Mutant: Median OS = {mut_med_os:.1f} months (N={len(mut_df)})")

# Plotting Kaplan Meier
plt.figure(figsize=(8, 6), dpi=300)

mut_t, mut_s = kaplan_meier(mut_df)
wt_t, wt_s = kaplan_meier(wt_df)

plt.step(wt_t, wt_s, where='post', color='#4DBBD5', linewidth=2.5, label=f'ARID1A Wild-Type\n(N={len(wt_df)}, mOS={wt_med_os:.1f} mo)')
plt.step(mut_t, mut_s, where='post', color='#E64B35', linewidth=2.5, label=f'ARID1A Mutant\n(N={len(mut_df)}, mOS={mut_med_os:.1f} mo)')

plt.title('Survival of High-TMB Bladder Cancer\nARID1A-Mediated Primary Resistance', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Overall Survival (Months)', fontsize=12, fontweight='bold')
plt.ylabel('Survival Probability', fontsize=12, fontweight='bold')
plt.ylim(0, 1.05)
plt.xlim(0, max(high_tmb_df['OS_MONTHS'].max() + 5, 20)) 

plt.legend(loc='upper right', frameon=False, fontsize=11)
sns.despine(top=True, right=True)
plt.grid(axis='y', linestyle=':', color='#DDDDDD')

output_path = 'bladder_high_tmb_arid1a_km_plot.png'
plt.tight_layout()
plt.savefig(output_path, transparent=False, facecolor='white')
print(f"Saved plot to {output_path}")
