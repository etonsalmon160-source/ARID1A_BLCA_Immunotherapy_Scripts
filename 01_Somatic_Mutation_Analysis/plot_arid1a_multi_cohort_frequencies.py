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

cohorts = {
    'TCGA': 'blca_tcga_pan_can_atlas_2018',
    'MSK-IMPACT': 'tmb_mskcc_2018',
    'BCAN/HCRN': 'blca_bcan_hcrn_2022',
    'IMvigor210': 'blca_iatlas_imvigor210_2017'
}

gene_id = 8289  # ARID1A
results = []

print("1. Querying multi-cohort data from cBioPortal...")
for name, study_id in cohorts.items():
    print(f"Fetching {name} ({study_id})...")
    # Fetch clinical samples
    url_samples = f'https://www.cbioportal.org/api/studies/{study_id}/clinical-data?projection=DETAILED&clinicalDataType=SAMPLE'
    req = urllib.request.Request(url_samples, headers={'Accept': 'application/json'})
    with urllib.request.urlopen(req) as r:
        samples = json.loads(r.read().decode('utf-8'))
    
    s_dict = {}
    for s in samples:
        sid = s['sampleId']
        if sid not in s_dict: 
            s_dict[sid] = {'SAMPLE_ID': sid}
        s_dict[sid][s['clinicalAttributeId']] = s['value']
    df_s = pd.DataFrame(list(s_dict.values()))
    
    # Filter bladder cancer for MSK-IMPACT
    if name == 'MSK-IMPACT':
        df_s = df_s[df_s['CANCER_TYPE'] == 'Bladder Cancer']
    
    s_ids = df_s['SAMPLE_ID'].tolist()
    total_samples = len(s_ids)
    
    # Fetch mutations
    url_mut = f'https://www.cbioportal.org/api/molecular-profiles/{study_id}_mutations/mutations/fetch'
    fetch_data = {'entrezGeneIds': [gene_id], 'sampleIds': s_ids}
    req = urllib.request.Request(url_mut, data=json.dumps(fetch_data).encode('utf-8'), method='POST')
    req.add_header('Content-Type', 'application/json')
    req.add_header('Accept', 'application/json')
    
    try:
        with urllib.request.urlopen(req) as r:
            muts = json.loads(r.read().decode('utf-8'))
        mutated_samples = len(set([m['sampleId'] for m in muts]))
    except Exception as e:
        mutated_samples = 0
        print(f'Error fetching mutations for {name}: {e}')
    
    freq = (mutated_samples / total_samples) * 100 if total_samples > 0 else 0
    results.append({
        'Cohort': name,
        'Total': total_samples,
        'Mutated': mutated_samples,
        'Frequency': freq
    })
    print(f" -> {name}: N={total_samples}, Mutants={mutated_samples}, Freq={freq:.2f}%")

df_res = pd.DataFrame(results)

# ======== 🎨 Plotting Panel A (Green Bar Chart) ========
plt.figure(figsize=(5.5, 4.5), dpi=300, facecolor='white')
ax = plt.gca()

# Dark green color matching HNSCC FAT1 paper Panel A
bar_color = '#1E8449'  # Forest green

bars = plt.bar(df_res['Cohort'], df_res['Frequency'], color=bar_color, width=0.45, edgecolor='black', linewidth=0.8)

# Add frequency values on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.5, f'{height:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.title('ARID1A Mutation Frequency in Bladder Cancer', fontsize=12, fontweight='bold', pad=15)
plt.ylabel('Mutation frequency (%)', fontsize=11, fontweight='bold')
plt.ylim(0, max(df_res['Frequency']) + 5)
plt.tick_params(axis='both', which='major', labelsize=10.5)
plt.xticks(rotation=15)
sns.despine(top=True, right=True)
plt.grid(axis='y', linestyle=':', color='#DDDDDD', alpha=0.7)

# Add "A" panel identifier
plt.text(-0.2, 1.05, 'A', transform=ax.transAxes, fontsize=20, fontweight='bold', va='top', ha='right')

plt.tight_layout()

# Save paths
output_path = r'[YOUR_WORKING_DIRECTORY]\arid1a_multi_cohort_mutation_frequencies.png'
plt.savefig(output_path, facecolor='white', edgecolor='none', transparent=False, bbox_inches='tight')
plt.close()
print(f"Successfully generated and saved Panel A to {output_path}")
