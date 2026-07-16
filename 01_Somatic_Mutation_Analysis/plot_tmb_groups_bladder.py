import urllib.request
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ======== 顶刊级绘图配置 ========
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['pdf.fonttype'] = 42

print("1. Fetching Bladder Cancer TMB data from MSK-IMPACT...")
url_samples = 'https://www.cbioportal.org/api/studies/tmb_mskcc_2018/clinical-data?projection=DETAILED&clinicalDataType=SAMPLE'
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

# Filter Bladder Cancer
df_samples['TMB_NONSYNONYMOUS'] = pd.to_numeric(df_samples.get('TMB_NONSYNONYMOUS', np.nan), errors='coerce')
blca_df = df_samples[df_samples['CANCER_TYPE'] == 'Bladder Cancer'].dropna(subset=['TMB_NONSYNONYMOUS']).copy()

# Calculate Tertiles (33.3% and 66.7% cutoffs)
q33 = blca_df['TMB_NONSYNONYMOUS'].quantile(1/3)
q67 = blca_df['TMB_NONSYNONYMOUS'].quantile(2/3)

print(f"Quantile Cutoffs for Bladder Cancer TMB:")
print(f"  33.3% Cutoff (Low vs Medium): {q33:.2f} Mut/Mb")
print(f"  66.7% Cutoff (Medium vs High): {q67:.2f} Mut/Mb")

low_tmb = blca_df[blca_df['TMB_NONSYNONYMOUS'] < q33]
med_tmb = blca_df[(blca_df['TMB_NONSYNONYMOUS'] >= q33) & (blca_df['TMB_NONSYNONYMOUS'] < q67)]
high_tmb = blca_df[blca_df['TMB_NONSYNONYMOUS'] >= q67]

print(f"Sample Sizes:")
print(f"  TMB-Low: {len(low_tmb)} patients")
print(f"  TMB-Medium: {len(med_tmb)} patients")
print(f"  TMB-High: {len(high_tmb)} patients")

# Plotting TMB distribution and shading groups
plt.figure(figsize=(10, 6), dpi=300, facecolor='white')

# KDE curve
sns.kdeplot(blca_df['TMB_NONSYNONYMOUS'], color='black', linewidth=2.5, label='_nolegend_')

# Get density values for shading
x_vals, y_vals = plt.gca().lines[0].get_data()

# Shade regions
plt.fill_between(x_vals, y_vals, where=(x_vals < q33) & (x_vals >= 0), color='#8491B4', alpha=0.6, label=f'TMB-Low (<{q33:.1f} Mut/Mb, N={len(low_tmb)})')
plt.fill_between(x_vals, y_vals, where=(x_vals >= q33) & (x_vals < q67), color='#F39B7F', alpha=0.6, label=f'TMB-Medium ({q33:.1f}-{q67:.1f} Mut/Mb, N={len(med_tmb)})')
plt.fill_between(x_vals, y_vals, where=(x_vals >= q67), color='#E64B35', alpha=0.6, label=f'TMB-High (>={q67:.1f} Mut/Mb, N={len(high_tmb)})')

# Add dividing lines
plt.axvline(q33, color='black', linestyle='--', linewidth=1.5)
plt.axvline(q67, color='black', linestyle='--', linewidth=1.5)

# Annotate lines
plt.text(q33 - 0.5, plt.gca().get_ylim()[1]*0.8, f'33.3% cutoff\n({q33:.1f} Mut/Mb)', ha='right', va='center', fontsize=10, fontweight='bold', color='#475569')
plt.text(q67 + 0.5, plt.gca().get_ylim()[1]*0.8, f'66.7% cutoff\n({q67:.1f} Mut/Mb)', ha='left', va='center', fontsize=10, fontweight='bold', color='#475569')

plt.title('Tumor Mutational Burden (TMB) Stratification in Bladder Cancer\n(Tertile Splitting: Low, Medium, High TMB)', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('TMB (Nonsynonymous Mutations per Mb)', fontsize=12, fontweight='bold')
plt.ylabel('Density', fontsize=12, fontweight='bold')
plt.xlim(0, blca_df['TMB_NONSYNONYMOUS'].quantile(0.98)) # Clip long right tail for better visualization

plt.legend(loc='upper right', frameon=False, fontsize=10)
sns.despine(top=True, right=True)
plt.grid(axis='y', linestyle=':', color='#DDDDDD', alpha=0.5)

output_path = 'bladder_tmb_groups_distribution.png'
plt.tight_layout()
plt.savefig(output_path, transparent=False, facecolor='white')
print(f"Saved plot to {output_path}")
