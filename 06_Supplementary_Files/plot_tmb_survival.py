import urllib.request
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

# ======== 顶刊级绘图配置 (Nature / Cell Style) ========
# 使用通用的无衬线字体 (类似 Arial / Helvetica)
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['pdf.fonttype'] = 42 # 保证能在AI中编辑
plt.rcParams['ps.fonttype'] = 42

# Nature/Lancet 常用高阶配色 (ggsci 风格)
COLOR_NEG = '#E64B35' # Nature Red
COLOR_POS = '#4DBBD5' # Nature Blue
COLOR_NEU = '#7E6148' # Muted Brown/Grey
# =======================================================

print("Fetching clinical data from cBioPortal...")
url_samples = 'https://www.cbioportal.org/api/studies/tmb_mskcc_2018/clinical-data?projection=DETAILED&clinicalDataType=SAMPLE'
req = urllib.request.Request(url_samples)
req.add_header('Accept', 'application/json')
with urllib.request.urlopen(req) as response:
    sample_data = json.loads(response.read().decode('utf-8'))

url_patients = 'https://www.cbioportal.org/api/studies/tmb_mskcc_2018/clinical-data?projection=DETAILED&clinicalDataType=PATIENT'
req = urllib.request.Request(url_patients)
req.add_header('Accept', 'application/json')
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

def parse_status(status_str):
    if pd.isna(status_str): return np.nan
    s = str(status_str).upper()
    if '1' in s or 'DECEASED' in s or 'DEAD' in s: return 1
    if '0' in s or 'LIVING' in s or 'ALIVE' in s: return 0
    return np.nan

df['EVENT'] = df['OS_STATUS'].apply(parse_status)

from lifelines.statistics import logrank_test

results = []
for ctype in df['CANCER_TYPE'].dropna().unique():
    subset = df[(df['CANCER_TYPE'] == ctype) & df['TMB_NONSYNONYMOUS'].notna() & df['OS_MONTHS'].notna() & df['EVENT'].notna()]
    if len(subset) < 20:
        continue
    
    threshold = subset['TMB_NONSYNONYMOUS'].quantile(0.8)
    high_tmb = subset[subset['TMB_NONSYNONYMOUS'] >= threshold]
    low_tmb = subset[subset['TMB_NONSYNONYMOUS'] < threshold]
    
    if len(high_tmb) < 5 or len(low_tmb) < 5:
        continue
        
    high_os = high_tmb['OS_MONTHS'].median()
    low_os = low_tmb['OS_MONTHS'].median()
    
    # Run Log-Rank test
    lr_results = logrank_test(
        high_tmb['OS_MONTHS'], low_tmb['OS_MONTHS'],
        event_observed_A=high_tmb['EVENT'], event_observed_B=low_tmb['EVENT']
    )
    p_val = lr_results.p_value
    
    results.append({
        'Cancer_Type': ctype,
        'N': len(subset),
        'TMB_Threshold': threshold,
        'High_TMB_OS': high_os,
        'Low_TMB_OS': low_os,
        'OS_Diff': high_os - low_os,
        'P_Value': p_val
    })

results_df = pd.DataFrame(results).sort_values('OS_Diff')
# Shorten long cancer types for a cleaner look
results_df['Cancer_Type'] = results_df['Cancer_Type'].str.replace('Cancer of Unknown Primary', 'CUP')
results_df['Cancer_Type'] = results_df['Cancer_Type'].str.replace('Non-Small Cell Lung Cancer', 'NSCLC')
results_df['Cancer_Type'] = results_df['Cancer_Type'].str.replace('Renal Cell Carcinoma', 'RCC')
results_df['Cancer_Type'] = results_df['Cancer_Type'].str.replace('Esophagogastric Cancer', 'Esophagogastric')

# ======== Start Plotting (Lollipop Plot) ========
fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

y_range = np.arange(len(results_df))
colors = [COLOR_NEG if x < 0 else COLOR_POS if x > 0 else COLOR_NEU for x in results_df['OS_Diff']]

# Draw horizontal lines connecting to zero (Lollipop stick)
ax.hlines(y=y_range, xmin=0, xmax=results_df['OS_Diff'], color=colors, alpha=0.8, linewidth=2.5)

# Draw the dots
ax.scatter(results_df['OS_Diff'], y_range, color=colors, s=120, zorder=3, edgecolors='white', linewidths=1.5)

# Axis styling
ax.set_yticks(y_range)
ax.set_yticklabels(results_df['Cancer_Type'], fontsize=11, color='#333333')
ax.set_xlabel('$\Delta$ Median Overall Survival (High vs. Low TMB, Months)', fontsize=12, fontweight='bold', color='#333333')
ax.axvline(0, color='black', linewidth=1.2, linestyle='-', zorder=1)

# 动态扩展 x 轴的范围，防止文字遮挡边缘的 y 轴标签
x_min, x_max = results_df['OS_Diff'].min(), results_df['OS_Diff'].max()
ax.set_xlim(x_min - 3.5, x_max + 3.5)

# 数据标注
for i, row in results_df.reset_index(drop=True).iterrows():
    val = row['OS_Diff']
    p_val = row['P_Value']
    
    # Format p-value
    if p_val < 0.001:
        p_str = "p < 0.001"
    else:
        p_str = f"p = {p_val:.3f}"
        
    if val != 0:
        label = f"{val:+.1f} mo ({p_str})"
    else:
        label = f"0.0 mo ({p_str})"
        
    x_pos = val - 0.25 if val < 0 else val + 0.25
    ha = 'right' if val < 0 else 'left'
    text_color = COLOR_NEG if val < 0 else '#555555'
    font_weight = 'bold' if p_val < 0.05 else 'normal'
    
    ax.text(x_pos, i, label, va='center', ha=ha, fontsize=9, fontweight=font_weight, color=text_color)

# Aesthetic tweaks
sns.despine(left=True, top=True, right=True, bottom=False)
ax.tick_params(axis='y', left=False) # remove y ticks
ax.tick_params(axis='x', color='#333333')
ax.spines['bottom'].set_linewidth(1.2)
ax.spines['bottom'].set_color('#333333')

# Add subtle vertical grid for x-axis
ax.grid(axis='x', linestyle=':', color='#DDDDDD', zorder=0)

# Optional minimalist title
plt.title('Immunotherapy Survival Outcomes by TMB Status\n(MSK-IMPACT Cohort, N=1661)', fontsize=14, pad=15, loc='left', fontweight='bold', color='#222222')

plt.tight_layout()
output_path = 'tmb_os_lollipop_plot.png'
plt.savefig(output_path, bbox_inches='tight', transparent=False, facecolor='white')
print(f"Plot saved to {output_path}")
