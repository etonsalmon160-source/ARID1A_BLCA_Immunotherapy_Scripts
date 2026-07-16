# plot_tert_multicohort_tmb_matched.py
# Multi-cohort TMB-matched survival analysis for TERT in bladder cancer
# Pure immunotherapy only (PD-1/PD-L1 monotherapy)
# Cohorts: MSK-IMPACT, IMvigor210, HCRN

import urllib.request
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
from scipy.spatial.distance import cdist
import ssl
import warnings
warnings.filterwarnings('ignore')

ssl._create_default_https_context = ssl._create_unverified_context

plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['pdf.fonttype'] = 42

TERT_ENTREZ = 7015

def fetch_cbioportal(url, data=None):
    req = urllib.request.Request(url)
    req.add_header('Accept', 'application/json')
    if data is not None:
        req.add_header('Content-Type', 'application/json')
        encoded_data = json.dumps(data).encode('utf-8')
        with urllib.request.urlopen(req, data=encoded_data, timeout=120) as response:
            return json.loads(response.read().decode('utf-8'))
    else:
        with urllib.request.urlopen(req, timeout=120) as response:
            return json.loads(response.read().decode('utf-8'))

def parse_status(status_str):
    if pd.isna(status_str): return np.nan
    s = str(status_str).upper()
    if '1' in s or 'DECEASED' in s or 'DEAD' in s: return 1
    if '0' in s or 'LIVING' in s or 'ALIVE' in s: return 0
    return np.nan

# ============================================================
# Cohort 1: MSK-IMPACT (tmb_mskcc_2018) - Pure PD-1/PDL-1
# ============================================================
print("=" * 60)
print("Cohort 1: MSK-IMPACT (tmb_mskcc_2018)")
print("=" * 60)

study1 = 'tmb_mskcc_2018'
url_s1 = f'https://www.cbioportal.org/api/studies/{study1}/clinical-data?projection=DETAILED&clinicalDataType=SAMPLE'
url_p1 = f'https://www.cbioportal.org/api/studies/{study1}/clinical-data?projection=DETAILED&clinicalDataType=PATIENT'

sample_data1 = fetch_cbioportal(url_s1)
patient_data1 = fetch_cbioportal(url_p1)

samples1 = {}
for item in sample_data1:
    sid = item.get('sampleId')
    if sid not in samples1:
        samples1[sid] = {'SAMPLE_ID': sid, 'PATIENT_ID': item.get('patientId')}
    samples1[sid][item.get('clinicalAttributeId')] = item.get('value')
df_s1 = pd.DataFrame(list(samples1.values()))

patients1 = {}
for item in patient_data1:
    pid = item.get('patientId')
    if pid not in patients1:
        patients1[pid] = {'PATIENT_ID': pid}
    patients1[pid][item.get('clinicalAttributeId')] = item.get('value')
df_p1 = pd.DataFrame(list(patients1.values()))

df1 = pd.merge(df_s1, df_p1, on='PATIENT_ID', how='inner')
df1['TMB'] = pd.to_numeric(df1.get('TMB_NONSYNONYMOUS', np.nan), errors='coerce')
df1['OS_MONTHS'] = pd.to_numeric(df1.get('OS_MONTHS', np.nan), errors='coerce')
df1['EVENT'] = df1['OS_STATUS'].apply(parse_status)

# Filter: Bladder Cancer + Pure PD-1/PDL-1
df1 = df1[df1['CANCER_TYPE'] == 'Bladder Cancer'].copy()
df1 = df1[df1['DRUG_TYPE'] == 'PD-1/PDL-1'].copy()
df1 = df1.dropna(subset=['TMB', 'OS_MONTHS', 'EVENT'])
df1['COHORT'] = 'MSK-IMPACT'

# Fetch TERT mutations
sids1 = df1['SAMPLE_ID'].tolist()
url_mut1 = f'https://www.cbioportal.org/api/molecular-profiles/{study1}_mutations/mutations/fetch'
mut_data1 = fetch_cbioportal(url_mut1, data={"entrezGeneIds": [TERT_ENTREZ], "sampleIds": sids1})
mut_sids1 = set([m['sampleId'] for m in mut_data1])
df1['TERT_Status'] = df1['SAMPLE_ID'].apply(lambda x: 'Mutant' if x in mut_sids1 else 'Wild-Type')

print(f"  Bladder Cancer pure ICB patients with OS: {len(df1)}")
print(f"  TERT Mutant: {(df1['TERT_Status']=='Mutant').sum()}, Wild-Type: {(df1['TERT_Status']=='Wild-Type').sum()}")
print(f"  Mean TMB (Mut): {df1[df1['TERT_Status']=='Mutant']['TMB'].mean():.2f}, Mean TMB (WT): {df1[df1['TERT_Status']=='Wild-Type']['TMB'].mean():.2f}")

# ============================================================
# Cohort 2: IMvigor210 (blca_iatlas_imvigor210_2017) - All Atezolizumab
# ============================================================
print("\n" + "=" * 60)
print("Cohort 2: IMvigor210 (Anti-PD-L1 Atezolizumab)")
print("=" * 60)

study2 = 'blca_iatlas_imvigor210_2017'
url_s2 = f'https://www.cbioportal.org/api/studies/{study2}/clinical-data?projection=DETAILED&clinicalDataType=SAMPLE'
url_p2 = f'https://www.cbioportal.org/api/studies/{study2}/clinical-data?projection=DETAILED&clinicalDataType=PATIENT'

sample_data2 = fetch_cbioportal(url_s2)
patient_data2 = fetch_cbioportal(url_p2)

samples2 = {}
for item in sample_data2:
    sid = item.get('sampleId')
    if sid not in samples2:
        samples2[sid] = {'SAMPLE_ID': sid, 'PATIENT_ID': item.get('patientId')}
    samples2[sid][item.get('clinicalAttributeId')] = item.get('value')
df_s2 = pd.DataFrame(list(samples2.values()))

patients2 = {}
for item in patient_data2:
    pid = item.get('patientId')
    if pid not in patients2:
        patients2[pid] = {'PATIENT_ID': pid}
    patients2[pid][item.get('clinicalAttributeId')] = item.get('value')
df_p2 = pd.DataFrame(list(patients2.values()))

df2 = pd.merge(df_s2, df_p2, on='PATIENT_ID', how='inner')
df2['TMB'] = pd.to_numeric(df2.get('TMB_NONSYNONYMOUS', np.nan), errors='coerce')
# If TMB_NONSYNONYMOUS is NaN, try MUTATION_COUNT
if df2['TMB'].isna().sum() > len(df2) * 0.5:
    df2['TMB'] = pd.to_numeric(df2.get('MUTATION_COUNT', np.nan), errors='coerce')
df2['OS_MONTHS'] = pd.to_numeric(df2.get('OS_MONTHS', np.nan), errors='coerce')
df2['EVENT'] = df2['OS_STATUS'].apply(parse_status)
df2 = df2.dropna(subset=['TMB', 'OS_MONTHS', 'EVENT'])
df2['COHORT'] = 'IMvigor210'

# IMvigor210 is entirely atezolizumab (anti-PD-L1) monotherapy
sids2 = df2['SAMPLE_ID'].tolist()
url_mut2 = f'https://www.cbioportal.org/api/molecular-profiles/{study2}_mutations/mutations/fetch'
mut_data2 = fetch_cbioportal(url_mut2, data={"entrezGeneIds": [TERT_ENTREZ], "sampleIds": sids2})
mut_sids2 = set([m['sampleId'] for m in mut_data2])
df2['TERT_Status'] = df2['SAMPLE_ID'].apply(lambda x: 'Mutant' if x in mut_sids2 else 'Wild-Type')

print(f"  IMvigor210 patients with TMB + OS: {len(df2)}")
print(f"  TERT Mutant: {(df2['TERT_Status']=='Mutant').sum()}, Wild-Type: {(df2['TERT_Status']=='Wild-Type').sum()}")
print(f"  Mean TMB (Mut): {df2[df2['TERT_Status']=='Mutant']['TMB'].mean():.2f}, Mean TMB (WT): {df2[df2['TERT_Status']=='Wild-Type']['TMB'].mean():.2f}")

# ============================================================
# Cohort 3: HCRN (blca_bcan_hcrn_2022) - Immunotherapy patients
# ============================================================
print("\n" + "=" * 60)
print("Cohort 3: HCRN (blca_bcan_hcrn_2022)")
print("=" * 60)

study3 = 'blca_bcan_hcrn_2022'
url_s3 = f'https://www.cbioportal.org/api/studies/{study3}/clinical-data?projection=DETAILED&clinicalDataType=SAMPLE'
url_p3 = f'https://www.cbioportal.org/api/studies/{study3}/clinical-data?projection=DETAILED&clinicalDataType=PATIENT'

sample_data3 = fetch_cbioportal(url_s3)
patient_data3 = fetch_cbioportal(url_p3)

samples3 = {}
for item in sample_data3:
    sid = item.get('sampleId')
    if sid not in samples3:
        samples3[sid] = {'SAMPLE_ID': sid, 'PATIENT_ID': item.get('patientId')}
    samples3[sid][item.get('clinicalAttributeId')] = item.get('value')
df_s3 = pd.DataFrame(list(samples3.values()))

patients3 = {}
for item in patient_data3:
    pid = item.get('patientId')
    if pid not in patients3:
        patients3[pid] = {'PATIENT_ID': pid}
    patients3[pid][item.get('clinicalAttributeId')] = item.get('value')
df_p3 = pd.DataFrame(list(patients3.values()))

df3 = pd.merge(df_s3, df_p3, on='PATIENT_ID', how='inner')

# HCRN: Use IMMUNO_SURVIVAL as immunotherapy-specific survival if available
# Filter: patients who received immunotherapy
df3_immuno = df3[df3.get('IMMUNOTHERAPY', pd.Series()) == 'Yes'].copy()

# Use MUTATION_COUNT as TMB proxy
df3_immuno['TMB'] = pd.to_numeric(df3_immuno.get('MUTATION_COUNT', np.nan), errors='coerce')
df3_immuno['OS_MONTHS'] = pd.to_numeric(df3_immuno.get('IMMUNO_SURVIVAL', np.nan), errors='coerce')

def parse_hcrn_status(s):
    if pd.isna(s): return np.nan
    s = str(s).upper()
    if 'DEAD' in s or 'DECEASED' in s: return 1
    if 'ALIVE' in s or 'LIVING' in s: return 0
    return np.nan

df3_immuno['EVENT'] = df3_immuno['SURVIVAL_STATUS'].apply(parse_hcrn_status)
df3_immuno = df3_immuno.dropna(subset=['TMB', 'OS_MONTHS', 'EVENT'])
df3_immuno['COHORT'] = 'HCRN'

sids3 = df3_immuno['SAMPLE_ID'].tolist()
url_mut3 = f'https://www.cbioportal.org/api/molecular-profiles/{study3}_mutations/mutations/fetch'
mut_data3 = fetch_cbioportal(url_mut3, data={"entrezGeneIds": [TERT_ENTREZ], "sampleIds": sids3})
mut_sids3 = set([m['sampleId'] for m in mut_data3])
df3_immuno['TERT_Status'] = df3_immuno['SAMPLE_ID'].apply(lambda x: 'Mutant' if x in mut_sids3 else 'Wild-Type')

print(f"  HCRN immunotherapy patients with TMB + OS: {len(df3_immuno)}")
print(f"  TERT Mutant: {(df3_immuno['TERT_Status']=='Mutant').sum()}, Wild-Type: {(df3_immuno['TERT_Status']=='Wild-Type').sum()}")
if (df3_immuno['TERT_Status']=='Mutant').sum() > 0:
    print(f"  Mean TMB (Mut): {df3_immuno[df3_immuno['TERT_Status']=='Mutant']['TMB'].mean():.2f}, Mean TMB (WT): {df3_immuno[df3_immuno['TERT_Status']=='Wild-Type']['TMB'].mean():.2f}")

# ============================================================
# Merge all cohorts
# ============================================================
print("\n" + "=" * 60)
print("Merging all cohorts")
print("=" * 60)

keep_cols = ['SAMPLE_ID', 'PATIENT_ID', 'TMB', 'OS_MONTHS', 'EVENT', 'TERT_Status', 'COHORT']
combined = pd.concat([
    df1[keep_cols],
    df2[keep_cols],
    df3_immuno[keep_cols]
], ignore_index=True)

print(f"Total combined patients: {len(combined)}")
print(f"  TERT Mutant: {(combined['TERT_Status']=='Mutant').sum()}")
print(f"  TERT Wild-Type: {(combined['TERT_Status']=='Wild-Type').sum()}")
print(f"  Mean TMB (Mut): {combined[combined['TERT_Status']=='Mutant']['TMB'].mean():.2f}")
print(f"  Mean TMB (WT): {combined[combined['TERT_Status']=='Wild-Type']['TMB'].mean():.2f}")
print(f"Cohort breakdown:")
print(combined.groupby(['COHORT', 'TERT_Status']).size())

# ============================================================
# TMB-Matched Analysis (Nearest-Neighbor Caliper Matching)
# ============================================================
print("\n" + "=" * 60)
print("TMB Propensity Score / Caliper Matching")
print("=" * 60)

# Log-transform TMB for better distribution matching
combined['log_TMB'] = np.log1p(combined['TMB'])

mut_group = combined[combined['TERT_Status'] == 'Mutant'].copy().reset_index(drop=True)
wt_group = combined[combined['TERT_Status'] == 'Wild-Type'].copy().reset_index(drop=True)

# Use 1:1 nearest-neighbor matching with caliper = 0.2 SD of log_TMB
pooled_sd = combined['log_TMB'].std()
caliper = 0.2 * pooled_sd

print(f"  Pooled SD of log(TMB): {pooled_sd:.4f}")
print(f"  Caliper (0.2 * SD): {caliper:.4f}")

# For each TERT mutant, find the nearest unmatched WT by log_TMB
matched_mut_idx = []
matched_wt_idx = []
wt_available = set(range(len(wt_group)))

for i in range(len(mut_group)):
    mut_tmb = mut_group.loc[i, 'log_TMB']
    best_dist = float('inf')
    best_j = None
    for j in wt_available:
        dist = abs(wt_group.loc[j, 'log_TMB'] - mut_tmb)
        if dist < best_dist:
            best_dist = dist
            best_j = j
    if best_j is not None and best_dist <= caliper:
        matched_mut_idx.append(i)
        matched_wt_idx.append(best_j)
        wt_available.remove(best_j)

matched_mut = mut_group.loc[matched_mut_idx].copy()
matched_wt = wt_group.loc[matched_wt_idx].copy()
matched_df = pd.concat([matched_mut, matched_wt], ignore_index=True)

print(f"\n  Matched pairs: {len(matched_mut_idx)}")
print(f"  Matched TERT Mutant: {len(matched_mut)}, Matched Wild-Type: {len(matched_wt)}")
print(f"  Mean TMB after matching (Mut): {matched_mut['TMB'].mean():.2f}")
print(f"  Mean TMB after matching (WT): {matched_wt['TMB'].mean():.2f}")

# Verify TMB balance with t-test
from scipy.stats import ttest_ind
t_stat, t_pval = ttest_ind(matched_mut['TMB'], matched_wt['TMB'])
print(f"  TMB balance t-test: t={t_stat:.3f}, P={t_pval:.3f} (P>0.05 = well balanced)")

# ============================================================
# Kaplan-Meier Survival Analysis on Matched Cohort
# ============================================================
print("\n" + "=" * 60)
print("Survival Analysis on TMB-Matched Cohort")
print("=" * 60)

# Log-rank test
results_matched = logrank_test(
    matched_mut['OS_MONTHS'], matched_wt['OS_MONTHS'],
    event_observed_A=matched_mut['EVENT'], event_observed_B=matched_wt['EVENT']
)
pval_matched = results_matched.p_value

# Cox HR
cph_matched = CoxPHFitter()
matched_df['Is_Mutant'] = matched_df['TERT_Status'].apply(lambda x: 1 if x == 'Mutant' else 0)
cph_matched.fit(matched_df[['OS_MONTHS', 'EVENT', 'Is_Mutant']], duration_col='OS_MONTHS', event_col='EVENT')
hr_matched = cph_matched.summary.loc['Is_Mutant', 'exp(coef)']
hr_ci_lower = cph_matched.summary.loc['Is_Mutant', 'exp(coef) lower 95%']
hr_ci_upper = cph_matched.summary.loc['Is_Mutant', 'exp(coef) upper 95%']

print(f"  Log-rank P = {pval_matched:.4f}")
print(f"  Cox HR = {hr_matched:.2f} (95% CI: {hr_ci_lower:.2f} - {hr_ci_upper:.2f})")
print(f"  Median OS (Mut): {matched_mut['OS_MONTHS'].median():.1f} mo")
print(f"  Median OS (WT): {matched_wt['OS_MONTHS'].median():.1f} mo")

# ============================================================
# Generate publication-grade figure (2 panels)
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), dpi=300, facecolor='white')

# --- Panel A: Unmatched (All combined) ---
ax1 = axes[0]
kmf_wt_all = KaplanMeierFitter()
kmf_mut_all = KaplanMeierFitter()

all_mut = combined[combined['TERT_Status'] == 'Mutant']
all_wt = combined[combined['TERT_Status'] == 'Wild-Type']

res_all = logrank_test(all_mut['OS_MONTHS'], all_wt['OS_MONTHS'],
                       event_observed_A=all_mut['EVENT'], event_observed_B=all_wt['EVENT'])
cph_all = CoxPHFitter()
combined_copy = combined.copy()
combined_copy['Is_Mutant'] = combined_copy['TERT_Status'].apply(lambda x: 1 if x == 'Mutant' else 0)
cph_all.fit(combined_copy[['OS_MONTHS', 'EVENT', 'Is_Mutant']], duration_col='OS_MONTHS', event_col='EVENT')
hr_all = cph_all.summary.loc['Is_Mutant', 'exp(coef)']

kmf_wt_all.fit(all_wt['OS_MONTHS'], event_observed=all_wt['EVENT'],
               label=f'TERT Wild-Type (N={len(all_wt)}, mOS={all_wt["OS_MONTHS"].median():.1f} mo)')
kmf_mut_all.fit(all_mut['OS_MONTHS'], event_observed=all_mut['EVENT'],
                label=f'TERT Mutant (N={len(all_mut)}, mOS={all_mut["OS_MONTHS"].median():.1f} mo)')

kmf_wt_all.plot_survival_function(ax=ax1, color='#4DBBD5', linewidth=2.5, ci_show=False)
kmf_mut_all.plot_survival_function(ax=ax1, color='#E64B35', linewidth=2.5, ci_show=False)

ax1.set_title('A. Multi-Cohort Combined (Unmatched)\nBladder Cancer Pure ICB Monotherapy', fontsize=12, fontweight='bold', pad=15)
ax1.set_xlabel('Survival Time (Months)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Overall Survival Probability', fontsize=11, fontweight='bold')
ax1.set_ylim(0, 1.05)

stat_text1 = f"Log-rank P = {res_all.p_value:.2e}\nCox HR = {hr_all:.2f}"
ax1.text(0.05, 0.15, stat_text1, transform=ax1.transAxes, fontsize=10.5, fontweight='bold',
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='#BDC3C7', boxstyle='round,pad=0.5'))
ax1.legend(loc='upper right', frameon=False, fontsize=9.5)
sns.despine(ax=ax1, top=True, right=True)
ax1.grid(axis='y', linestyle=':', color='#DDDDDD')

# --- Panel B: TMB-Matched ---
ax2 = axes[1]
kmf_wt_m = KaplanMeierFitter()
kmf_mut_m = KaplanMeierFitter()

kmf_wt_m.fit(matched_wt['OS_MONTHS'], event_observed=matched_wt['EVENT'],
             label=f'TERT Wild-Type (N={len(matched_wt)}, mOS={matched_wt["OS_MONTHS"].median():.1f} mo)')
kmf_mut_m.fit(matched_mut['OS_MONTHS'], event_observed=matched_mut['EVENT'],
              label=f'TERT Mutant (N={len(matched_mut)}, mOS={matched_mut["OS_MONTHS"].median():.1f} mo)')

kmf_wt_m.plot_survival_function(ax=ax2, color='#4DBBD5', linewidth=2.5, ci_show=False)
kmf_mut_m.plot_survival_function(ax=ax2, color='#E64B35', linewidth=2.5, ci_show=False)

ax2.set_title('B. TMB-Matched Analysis\n(Caliper Nearest-Neighbor 1:1 Matching)', fontsize=12, fontweight='bold', pad=15)
ax2.set_xlabel('Survival Time (Months)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Overall Survival Probability', fontsize=11, fontweight='bold')
ax2.set_ylim(0, 1.05)

stat_text2 = f"Log-rank P = {pval_matched:.2e}\nCox HR = {hr_matched:.2f} ({hr_ci_lower:.2f}-{hr_ci_upper:.2f})\nTMB balance P = {t_pval:.3f}"
ax2.text(0.05, 0.10, stat_text2, transform=ax2.transAxes, fontsize=10, fontweight='bold',
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='#BDC3C7', boxstyle='round,pad=0.5'))
ax2.legend(loc='upper right', frameon=False, fontsize=9.5)
sns.despine(ax=ax2, top=True, right=True)
ax2.grid(axis='y', linestyle=':', color='#DDDDDD')

plt.tight_layout()
output_path = 'bladder_tert_multicohort_tmb_matched_km.png'
plt.savefig(output_path, transparent=False, facecolor='white', bbox_inches='tight')
print(f"\nSaved plot to {output_path}")
plt.close()
