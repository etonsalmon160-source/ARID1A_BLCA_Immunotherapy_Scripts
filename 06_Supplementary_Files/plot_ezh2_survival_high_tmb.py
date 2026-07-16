# plot_ezh2_survival_high_tmb.py
import urllib.request
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['pdf.fonttype'] = 42

study_id = 'tmb_mskcc_2018'

# Fetch clinical data
print("1. Fetching clinical data...")
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

blca_df = df[df['CANCER_TYPE'] == 'Bladder Cancer'].copy()

def parse_status(status_str):
    if pd.isna(status_str): return np.nan
    s = str(status_str).upper()
    if '1' in s or 'DECEASED' in s or 'DEAD' in s: return 1
    if '0' in s or 'LIVING' in s or 'ALIVE' in s: return 0
    return np.nan

blca_df['EVENT'] = blca_df['OS_STATUS'].apply(parse_status)
blca_df = blca_df.dropna(subset=['OS_MONTHS', 'EVENT'])

# Filter by pure immunotherapy (PD-1/PD-L1 monotherapy, excluding Combo)
blca_df = blca_df[blca_df['DRUG_TYPE'] == 'PD-1/PDL-1'].copy()

# High TMB cohort (tertile High TMB >= 11.74)
tmb_thresh = 11.74
sub_df = blca_df[blca_df['TMB_NONSYNONYMOUS'] >= tmb_thresh].copy()
sub_sample_ids = sub_df['SAMPLE_ID'].tolist()

print(f"Pure immunotherapy High TMB cohort size (>= 11.74): {len(sub_df)}")

# Fetch EZH2 (Entrez ID: 2146) mutations
print("2. Fetching EZH2 mutations...")
url_mut = f'https://www.cbioportal.org/api/molecular-profiles/{study_id}_mutations/mutations/fetch'
fetch_data = {"entrezGeneIds": [2146], "sampleIds": sub_sample_ids}
req = urllib.request.Request(url_mut, data=json.dumps(fetch_data).encode('utf-8'), method='POST')
req.add_header('Content-Type', 'application/json')
req.add_header('Accept', 'application/json')
with urllib.request.urlopen(req) as response:
    mut_data = json.loads(response.read().decode('utf-8'))

mutated_samples = set([m['sampleId'] for m in mut_data])
sub_df['EZH2_Status'] = sub_df['SAMPLE_ID'].apply(lambda x: 'Mutant' if x in mutated_samples else 'Wild-Type')

mut_df = sub_df[sub_df['EZH2_Status'] == 'Mutant']
wt_df = sub_df[sub_df['EZH2_Status'] == 'Wild-Type']

# Calculate log-rank test
results = logrank_test(mut_df['OS_MONTHS'], wt_df['OS_MONTHS'], event_observed_A=mut_df['EVENT'], event_observed_B=wt_df['EVENT'])
pval = results.p_value

# Plot Kaplan-Meier survival curves
plt.figure(figsize=(7, 6), dpi=300)
kmf_wt = KaplanMeierFitter()
kmf_mut = KaplanMeierFitter()

kmf_wt.fit(wt_df['OS_MONTHS'], event_observed=wt_df['EVENT'], label=f'EZH2 Wild-Type (N={len(wt_df)}, mOS={wt_df["OS_MONTHS"].median():.1f} mo)')
kmf_mut.fit(mut_df['OS_MONTHS'], event_observed=mut_df['EVENT'], label=f'EZH2 Mutant (N={len(mut_df)}, mOS={mut_df["OS_MONTHS"].median():.1f} mo)')

ax = plt.subplot(111)
kmf_wt.plot_survival_function(ax=ax, color='#4DBBD5', linewidth=2.5, ci_show=False)
kmf_mut.plot_survival_function(ax=ax, color='#E64B35', linewidth=2.5, ci_show=False)

plt.title('Overall Survival in High-TMB Bladder Cancer (TMB >= 11.7)\nby EZH2 Mutation (Pure PD-1/PD-L1 Monotherapy Cohort)', fontsize=12, fontweight='bold', pad=15)
plt.xlabel('Survival Time (Months)', fontsize=11, fontweight='bold')
plt.ylabel('Overall Survival Probability', fontsize=11, fontweight='bold')
plt.ylim(0, 1.05)
plt.xlim(0, max(sub_df['OS_MONTHS'].max() + 5, 20))

# Annotation of p-value and Hazard Ratio
from lifelines import CoxPHFitter
cph = CoxPHFitter()
sub_df['Is_Mutant'] = sub_df['EZH2_Status'].apply(lambda x: 1 if x == 'Mutant' else 0)
cph.fit(sub_df[['OS_MONTHS', 'EVENT', 'Is_Mutant']], duration_col='OS_MONTHS', event_col='EVENT')
hr = cph.summary.loc['Is_Mutant', 'exp(coef)']

stat_text = f"Log-rank P = {pval:.2e}\nCox HR = {hr:.2f}"
plt.text(0.05, 0.15, stat_text, transform=ax.transAxes, fontsize=10.5, fontweight='bold', bbox=dict(facecolor='white', alpha=0.8, edgecolor='#BDC3C7', boxstyle='round,pad=0.5'))

plt.legend(loc='upper right', frameon=False, fontsize=10)
sns.despine(top=True, right=True)
plt.grid(axis='y', linestyle=':', color='#DDDDDD')

output_path = 'bladder_high_tmb_ezh2_km_plot.png'
plt.tight_layout()
plt.savefig(output_path, transparent=False, facecolor='white')
print(f"Successfully generated survival plot at {output_path}")
