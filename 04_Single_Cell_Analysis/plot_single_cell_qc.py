import os
import gzip
import urllib.request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ======== 🎨 顶刊级绘图配置 ========
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['pdf.fonttype'] = 42

urls_gse135337 = {
    'BC1': "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM4006nnn/GSM4006644/suppl/GSM4006644_BC1_gene_cell_exprs_table.txt.gz",
    'BC2': "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM4006nnn/GSM4006645/suppl/GSM4006645_BC2_gene_cell_exprs_table.txt.gz",
    'BC3': "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM4006nnn/GSM4006646/suppl/GSM4006646_BC3_gene_cell_exprs_table.txt.gz",
    'BC4': "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM4006nnn/GSM4006647/suppl/GSM4006647_BC4_gene_cell_exprs_table.txt.gz",
    'BC5': "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM4006nnn/GSM4006648/suppl/GSM4006648_BC5_gene_cell_exprs_table.txt.gz",
    'BC6': "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM4751nnn/GSM4751267/suppl/GSM4751267_BC6_gene_cell_exprs_table.txt.gz",
    'BC7': "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM4751nnn/GSM4751268/suppl/GSM4751268_BC7_gene_cell_exprs_table.txt.gz"
}
url_gse145137 = "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM4307nnn/GSM4307111/suppl/GSM4307111_GEO_processed_BC159-T_3_log2TPM_matrix_final.txt.gz"

dest_dir = "[YOUR_WORKING_DIRECTORY]\\scratch"
os.makedirs(dest_dir, exist_ok=True)

print("Gathering QC metrics across all cells for 8 patients...")
qc_data = []

# --- Process GSE135337 (BC1-BC7) ---
for name, url in urls_gse135337.items():
    local_path = os.path.join(dest_dir, f"{name}_matrix.txt.gz")
    if not os.path.exists(local_path):
        print(f"Downloading {name} matrix...")
        urllib.request.urlretrieve(url, local_path)
        
    print(f"Streaming cells for {name}...")
    with gzip.open(local_path, 'rt') as f:
        header = f.readline().strip().split('\t')
        barcodes = header[2:]
        n_cells = len(barcodes)
        
        total_counts = np.zeros(n_cells)
        mt_counts = np.zeros(n_cells)
        detected_genes = np.zeros(n_cells)
        
        for line in f:
            parts = line.strip().split('\t')
            symbol = parts[1]
            expr_vals = np.array([float(x) for x in parts[2:]])
            
            total_counts += expr_vals
            detected_genes += (expr_vals > 0)
            
            if symbol.upper().startswith('MT-') or symbol.upper().startswith('MT.'):
                mt_counts += expr_vals
                
    pct_counts_mt = (mt_counts / (total_counts + 1e-6)) * 100
    
    # Store all cells' QC metrics
    for i in range(n_cells):
        qc_data.append({
            'Patient': name,
            'n_counts': total_counts[i],
            'n_genes': detected_genes[i],
            'pct_counts_mt': pct_counts_mt[i],
            'Status': 'Passed' if (total_counts[i] >= 500 and total_counts[i] <= 15000 and 
                                   detected_genes[i] >= 200 and detected_genes[i] <= 4000 and 
                                   pct_counts_mt[i] <= 20.0) else 'Filtered'
        })

# --- Process GSE145137 (BC159) ---
local_path_bc159 = os.path.join(dest_dir, "GSM4307111_matrix.txt.gz")
if not os.path.exists(local_path_bc159):
    print("Downloading BC159 matrix...")
    urllib.request.urlretrieve(url_gse145137, local_path_bc159)
    
print("Streaming cells for BC159...")
with gzip.open(local_path_bc159, 'rt') as f:
    header = f.readline().strip().split('\t')
    barcodes = header[1:]
    n_cells = len(barcodes)
    
    total_counts = np.zeros(n_cells)
    mt_counts = np.zeros(n_cells)
    detected_genes = np.zeros(n_cells)
    
    for line in f:
        parts = line.strip().split('\t')
        symbol = parts[0]
        expr_vals = np.array([float(x) for x in parts[1:]])
        
        total_counts += expr_vals
        detected_genes += (expr_vals > 0)
        
        if symbol.upper().startswith('MT-') or symbol.upper().startswith('MT.'):
            mt_counts += expr_vals
            
pct_counts_mt = (mt_counts / (total_counts + 1e-6)) * 100

for i in range(n_cells):
    qc_data.append({
        'Patient': 'BC159',
        'n_counts': total_counts[i],
        'n_genes': detected_genes[i],
        'pct_counts_mt': pct_counts_mt[i],
        'Status': 'Passed' if (detected_genes[i] >= 200 and pct_counts_mt[i] <= 20.0) else 'Filtered'
    })

df_qc_all = pd.DataFrame(qc_data)

# Print summary
print("Global QC Summary:")
print(df_qc_all['Status'].value_counts())

# Generate publication-grade violin plots showing quality control metrics
print("Plotting publication-grade QC composite figure...")
fig, axes = plt.subplots(3, 1, figsize=(12, 14), dpi=300, sharex=False, facecolor='white')

# Theme palettes
colors = ['#E74C3C', '#3498DB', '#2ECC71', '#9B59B6', '#1ABC9C', '#16A085', '#2C3E50', '#F1C40F']
sns.set_style("whitegrid")

# --- Panel A: UMI Counts per Cell ---
sns.violinplot(
    data=df_qc_all, x='Patient', y='n_counts', ax=axes[0], 
    palette=colors, hue='Patient', legend=False, inner='quartile', density_norm='width'
)
axes[0].set_title('Cell UMI Counts (Library Size)', fontsize=13, fontweight='bold', pad=8)
axes[0].set_ylabel('UMI Counts', fontsize=11)
axes[0].set_xlabel('')
# Add dashed line indicating threshold (GSE135337 lower and upper bounds)
axes[0].axhline(500, color='red', linestyle='--', linewidth=1.0, label='Min Threshold (500)')
axes[0].axhline(15000, color='red', linestyle='--', linewidth=1.0, label='Max Threshold (15,000)')
axes[0].legend(loc='upper right', frameon=True, facecolor='white', edgecolor='none', fontsize=9)

# --- Panel B: Number of Detected Genes per Cell ---
sns.violinplot(
    data=df_qc_all, x='Patient', y='n_genes', ax=axes[1], 
    palette=colors, hue='Patient', legend=False, inner='quartile', density_norm='width'
)
axes[1].set_title('Number of Detected Genes', fontsize=13, fontweight='bold', pad=8)
axes[1].set_ylabel('Gene Count', fontsize=11)
axes[1].set_xlabel('')
axes[1].axhline(200, color='red', linestyle='--', linewidth=1.0, label='Min Genes (200)')
axes[1].axhline(4000, color='red', linestyle='--', linewidth=1.0, label='Max Genes (4,000)')
axes[1].legend(loc='upper right', frameon=True, facecolor='white', edgecolor='none', fontsize=9)

# --- Panel C: Mitochondrial Percentage per Cell ---
sns.violinplot(
    data=df_qc_all, x='Patient', y='pct_counts_mt', ax=axes[2], 
    palette=colors, hue='Patient', legend=False, inner='quartile', density_norm='width'
)
axes[2].set_title('Mitochondrial Gene Percentage', fontsize=13, fontweight='bold', pad=8)
axes[2].set_ylabel('Mitochondrial %', fontsize=11)
axes[2].set_xlabel('Patient Specimen', fontsize=11)
axes[2].axhline(20.0, color='red', linestyle='--', linewidth=1.0, label='Max MT% Threshold (20%)')
axes[2].legend(loc='upper right', frameon=True, facecolor='white', edgecolor='none', fontsize=9)

# Clean styling
for ax in axes:
    ax.tick_params(axis='both', which='major', labelsize=10)
    sns.despine(ax=ax, top=True, right=True)

plt.tight_layout()
output_path = '[YOUR_WORKING_DIRECTORY]\\bladder_single_cell_qc_violins.png'
plt.savefig(output_path, transparent=False, facecolor='white', bbox_inches='tight')
plt.close()
print(f"Successfully generated and saved publication-grade QC plot to {output_path}")
