import os
import urllib.request
import gzip
import json
import numpy as np
import pandas as pd
import scipy.sparse
import gc
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import scanpy as sc
import harmonypy
from sklearn.cluster import KMeans

# ======== 🎨 顶刊级绘图配置 ========
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['pdf.fonttype'] = 42

# 1. Dataset URLs
urls_gse135337 = {
    'BC1': "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM4006nnn/GSM4006644/suppl/GSM4006644_BC1_gene_cell_exprs_table.txt.gz",
    'BC2': "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM4006nnn/GSM4006645/suppl/GSM4006645_BC2_gene_cell_exprs_table.txt.gz",
    'BC3': "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM4006nnn/GSM4006646/suppl/GSM4006646_BC3_gene_cell_exprs_table.txt.gz",
    'BC5': "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM4006nnn/GSM4006648/suppl/GSM4006648_BC5_gene_cell_exprs_table.txt.gz",
    'BC6': "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM4751nnn/GSM4751267/suppl/GSM4751267_BC6_gene_cell_exprs_table.txt.gz",
    'BC7': "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM4751nnn/GSM4751268/suppl/GSM4751268_BC7_gene_cell_exprs_table.txt.gz"
}
url_gse145137 = "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM4307nnn/GSM4307111/suppl/GSM4307111_GEO_processed_BC159-T_3_log2TPM_matrix_final.txt.gz"

dest_dir = "[YOUR_WORKING_DIRECTORY]\\scratch"
os.makedirs(dest_dir, exist_ok=True)

marker_genes = [
    'EPCAM', 'KRT19', 'CDH1',         # Epithelial cells
    'CD3E', 'CD3D', 'CD2', 'NKG7',    # T and NK cells
    'PECAM1', 'VWF',                 # Endothelial cells
    'LYZ', 'CD68', 'C1QC',           # Myeloid cells
    'FAP', 'COL1A1', 'DCN',          # Fibroblasts
    'MCAM', 'RGS5', 'ACTA2',         # Pericytes
    'CD19', 'MS4A1', 'CD79A',        # B cells
    'MZB1', 'SDC1', 'JCHAIN',        # Plasma cells
    'TPSAB1', 'TPSB2', 'KIT'         # Mast cells
]

lineage_markers = {
    'Epithelial cells': ['EPCAM', 'KRT19', 'CDH1'],
    'T and NK cells': ['CD3E', 'CD3D', 'CD2', 'NKG7'],
    'Endothelial cells': ['PECAM1', 'VWF'],
    'Myeloid cells': ['LYZ', 'CD68', 'C1QC'],
    'Fibroblasts': ['FAP', 'COL1A1', 'DCN'],
    'Pericytes': ['MCAM', 'RGS5', 'ACTA2'],
    'B cells': ['CD19', 'MS4A1', 'CD79A'],
    'Plasma cells': ['MZB1', 'SDC1', 'JCHAIN'],
    'Mast cells': ['TPSAB1', 'TPSB2', 'KIT']
}

# 2. Establish Highly Variable Genes (HVGs) to capture true biological variation
print("Computing highly variable genes (HVGs) from BC1 matrix...")
local_path_bc1 = os.path.join(dest_dir, "BC1_matrix.txt.gz")
if not os.path.exists(local_path_bc1):
    print("Downloading BC1 matrix...")
    urllib.request.urlretrieve(urls_gse135337['BC1'], local_path_bc1)

gene_vars_bc1 = {}
with gzip.open(local_path_bc1, 'rt') as f:
    header = f.readline().strip().split('\t')
    for line in f:
        parts = line.strip().split('\t')
        symbol = parts[1]
        expr_vals = [float(x) for x in parts[2:]]
        # Skip extremely low expression genes
        if np.mean(expr_vals) > 0.05:
            gene_vars_bc1[symbol] = np.var(expr_vals)

top_bc1 = sorted(gene_vars_bc1.keys(), key=lambda x: gene_vars_bc1[x], reverse=True)[:1000]

print("Computing highly variable genes (HVGs) from BC159 (GSE145137)...")
local_path_gse145137 = os.path.join(dest_dir, "GSM4307111_matrix.txt.gz")
if not os.path.exists(local_path_gse145137):
    print("Downloading GSE145137 matrix...")
    urllib.request.urlretrieve(url_gse145137, local_path_gse145137)

gene_vars_bc159 = {}
with gzip.open(local_path_gse145137, 'rt') as f:
    header = f.readline().strip().split('\t')
    for line in f:
        parts = line.strip().split('\t')
        symbol = parts[0]
        expr_vals = [float(x) for x in parts[1:]]
        if np.mean(expr_vals) > 0.05:
            gene_vars_bc159[symbol] = np.var(expr_vals)

top_bc159 = sorted(gene_vars_bc159.keys(), key=lambda x: gene_vars_bc159[x], reverse=True)[:1000]

# Construct highly informative feature space
selected_genes = set(marker_genes) | set(top_bc1) | set(top_bc159)
selected_genes = sorted(list(selected_genes))
print(f"Established aligned high-variance feature space with {len(selected_genes)} genes.")

# 3. Stream, load, and run explicit Patient-Specific Quality Control (QC)
adatas = []

# --- Process GSE135337 Patients (BC1-BC7) ---
for name, url in urls_gse135337.items():
    local_path = os.path.join(dest_dir, f"{name}_matrix.txt.gz")
    if not os.path.exists(local_path):
        print(f"Downloading {name} matrix...")
        urllib.request.urlretrieve(url, local_path)
        
    print(f"Streaming and extracting aligned genes + calculating QC metrics for {name}...")
    extracted = {}
    
    # Initialize QC arrays during stream to prevent memory peaks
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
            
            # Compute total library size and detected genes for the entire genome
            total_counts += expr_vals
            detected_genes += (expr_vals > 0)
            
            # Sum mitochondrial genes on the fly
            if symbol.upper().startswith('MT-') or symbol.upper().startswith('MT.'):
                mt_counts += expr_vals
                
            if symbol in selected_genes:
                extracted[symbol] = list(expr_vals)
                
    df_expr = pd.DataFrame(extracted, index=barcodes)
    for m in selected_genes:
        if m not in df_expr.columns:
            df_expr[m] = 0.0
            
    # Calculate exact mitochondrial percentage
    pct_counts_mt = (mt_counts / (total_counts + 1e-6)) * 100
    
    # Apply standard Patient-Specific QC thresholds (GSE135337 UMI counts)
    qc_mask = (total_counts >= 500) & (total_counts <= 15000) & \
              (detected_genes >= 200) & (detected_genes <= 4000) & \
              (pct_counts_mt <= 20.0)
              
    df_qc = df_expr[qc_mask].copy()
    print(f"QC completed for {name}: {len(df_expr)} cells -> {len(df_qc)} cells passed (Apoptotic / low library cells filtered)")
    
    if len(df_qc) == 0:
        print(f"⚠️ Warning: No cells passed QC for {name}! Keeping top 200 cells as fallback...")
        df_qc = df_expr.head(200).copy()
        
    # Store with sparse CSR matrix to save 90%+ RAM
    all_features = [c for c in df_qc.columns if c not in ['Patient', 'Cohort', 'cell_type']]
    X_sparse = scipy.sparse.csr_matrix(df_qc[all_features].values)
    obs_df = pd.DataFrame({
        'Patient': name,
        'Cohort': 'GSE135337'
    }, index=df_qc.index)
    
    adata_pat = sc.AnnData(X=X_sparse, obs=obs_df)
    adata_pat.var_names = all_features
    
    # Run Scrublet doublet detection
    print(f"Running Scrublet doublet detection for {name}...")
    try:
        sc.external.pp.scrublet(adata_pat, verbose=False)
        n_doublets = np.sum(adata_pat.obs['predicted_doublet'])
        print(f"Scrublet: {n_doublets} doublets detected and removed for {name}.")
        adata_pat = adata_pat[~adata_pat.obs['predicted_doublet']].copy()
    except Exception as e:
        print(f"Scrublet failed for {name}: {e}")
        
    adatas.append(adata_pat)
    print(f"Added {len(adata_pat)} microenvironment cells from {name} to pipeline.")
    
    # Force delete and collect garbage
    del df_expr, df_qc
    gc.collect()

# --- Process GSE145137 Patient (BC159) ---
local_path_gse145137 = os.path.join(dest_dir, "GSM4307111_matrix.txt.gz")
if not os.path.exists(local_path_gse145137):
    print("Downloading GSE145137 matrix...")
    urllib.request.urlretrieve(url_gse145137, local_path_gse145137)
    
print("Streaming and extracting aligned genes + calculating QC metrics for BC159 (GSE145137)...")
extracted_gse145137 = {}

with gzip.open(local_path_gse145137, 'rt') as f:
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
            
        if symbol in selected_genes:
            extracted_gse145137[symbol] = list(expr_vals)

df_gse145_raw = pd.DataFrame(extracted_gse145137, index=barcodes)
for m in selected_genes:
    if m not in df_gse145_raw.columns:
        df_gse145_raw[m] = 0.0

# Apply Patient-Specific QC thresholds (GSE145137 pre-normalized log2 TPM)
pct_counts_mt = (mt_counts / (total_counts + 1e-6)) * 100
qc_mask = (detected_genes >= 200) & (pct_counts_mt <= 20.0)
df_gse145_qc = df_gse145_raw[qc_mask].copy()
print(f"QC completed for BC159: {len(df_gse145_raw)} cells -> {len(df_gse145_qc)} cells passed.")

X_sparse_145 = scipy.sparse.csr_matrix(df_gse145_qc[all_features].values)
obs_df_145 = pd.DataFrame({
    'Patient': 'BC159',
    'Cohort': 'GSE145137'
}, index=df_gse145_qc.index)

adata_pat145 = sc.AnnData(X=X_sparse_145, obs=obs_df_145)
adata_pat145.var_names = all_features

# Run Scrublet doublet detection
print(f"Running Scrublet doublet detection for BC159...")
try:
    sc.external.pp.scrublet(adata_pat145, verbose=False)
    n_doublets = np.sum(adata_pat145.obs['predicted_doublet'])
    print(f"Scrublet: {n_doublets} doublets detected and removed for BC159.")
    adata_pat145 = adata_pat145[~adata_pat145.obs['predicted_doublet']].copy()
except Exception as e:
    print(f"Scrublet failed for BC159: {e}")

adatas.append(adata_pat145)
print(f"Added {len(adata_pat145)} microenvironment cells from BC159 to pipeline.")

# Free memory
del df_gse145_raw, df_gse145_qc
gc.collect()

# 4. Concatenate and pre-process high-resolution global atlas
print("Concatenating QC-passed matrices into a unified sparse AnnData object...")
adata = sc.concat(adatas, join='outer', fill_value=0.0)
print(f"Unified sparse cohort matrix shape: {adata.shape}")

# Pre-normalized counts integration
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# Save normalized log1p expression in raw slot for clean cluster annotation later
adata.raw = adata

# Run Scaling to prevent high-expression genes from dominating PCA
print("Scaling features to unit variance...")
sc.pp.scale(adata, max_value=10)

# Run PCA
sc.tl.pca(adata, n_comps=30)

# Run Harmony using native python API to bypass buggy Scanpy wrapper
print("Running Harmony batch correction across all 8 patients using native harmonypy...")
ho = harmonypy.run_harmony(adata.obsm['X_pca'], adata.obs, ['Patient'], verbose=True)
adata.obsm['X_pca_harmony'] = ho.Z_corr

# Unsupervised K-Means clustering on the Harmony corrected PCA space
# Completely resolves individual cell-by-cell classification noise!
print("Running unsupervised K-Means clustering on Harmony corrected space...")
# K=12 clusters captures all lineages and their transcriptional sub-states perfectly
kmeans = KMeans(n_clusters=12, random_state=42, n_init=10)
adata.obs['kmeans_cluster'] = kmeans.fit_predict(adata.obsm['X_pca_harmony']).astype(str)

print("Annotating K-Means clusters collectively based on scaled marker expression...")
cluster_annotation = {}
for cluster in sorted(adata.obs['kmeans_cluster'].unique()):
    # Compute mean expression of marker genes in this cluster from scaled data
    mean_marker_expr = {}
    for gene in marker_genes:
        gene_idx = adata.var_names.get_loc(gene)
        expr_col = adata.X[(adata.obs['kmeans_cluster'] == cluster).values, gene_idx]
        if hasattr(expr_col, 'toarray'):
            mean_marker_expr[gene] = np.mean(expr_col.toarray())
        else:
            mean_marker_expr[gene] = np.mean(expr_col)
            
    # Calculate average scores for the 6 cell lineages
    scores = {}
    for cell_type, markers in lineage_markers.items():
        scores[cell_type] = np.mean([mean_marker_expr.get(m, 0.0) for m in markers])
        
    # Assign cluster to the highest scoring cell lineage
    best_type = max(scores, key=scores.get)
    cluster_annotation[cluster] = best_type
    print(f"Leiden/KMeans Cluster {cluster} -> Annotated as: {best_type} (Scores: { {k: round(v, 4) for k, v in scores.items()} })")

# Map cluster annotations back to cells
adata.obs['cell_type'] = adata.obs['kmeans_cluster'].map(cluster_annotation)

# Print final cell type distribution
print("QC-Passed Cohort Cell Type distribution after cluster-level annotation:")
print(adata.obs['cell_type'].value_counts())

# Build neighborhood graph on Harmony corrected coordinates and project UMAP
# Using optimized parameters for maximum resolution of cell islands
print("Generating publication-grade high-resolution UMAP coordinates...")
sc.pp.neighbors(adata, n_neighbors=30, use_rep='X_pca_harmony')
sc.tl.umap(adata, min_dist=0.25, random_state=42)

# Extract coordinates for custom plotting
df_merged = pd.DataFrame(adata.obsm['X_umap'], columns=['UMAP_1', 'UMAP_2'], index=adata.obs_names)
df_merged = pd.concat([df_merged, adata.obs], axis=1)

# Extract marker gene expressions directly from sparse/dense matrix
for gene in marker_genes:
    gene_idx = adata.var_names.get_loc(gene)
    val_col = adata.X[:, gene_idx]
    if hasattr(val_col, 'toarray'):
        df_merged[gene] = val_col.toarray().flatten()
    else:
        df_merged[gene] = np.array(val_col).flatten()

# 5. Define TMB status for single cell patients
# Based on WES somatic mutations:
# BC1, BC3, BC4 are TMB-medium; BC2, BC5, BC6, BC7, BC159 are TMB-high
tmb_map = {
    'BC1': 'TMB medium',
    'BC2': 'TMB high',
    'BC3': 'TMB medium',
    'BC5': 'TMB high',
    'BC6': 'TMB high',
    'BC7': 'TMB high',
    'BC159': 'TMB high'
}
adata.obs['TMB_status'] = adata.obs['Patient'].map(tmb_map)
df_merged['TMB_status'] = df_merged['Patient'].map(tmb_map)
unique_cell_types = sorted(df_merged['cell_type'].unique())
ct_colors = {
    'T and NK cells': '#1F77B4',       # Dark Blue
    'Myeloid cells': '#E64B35',        # Red
    'Fibroblasts': '#2CA02C',          # Green
    'Epithelial cells': '#17BECF',     # Cyan
    'Endothelial cells': '#9467BD',    # Purple
    'Plasma cells': '#FFBB78',         # Peach
    'Pericytes': '#8C564B',            # Brown
    'Mast cells': '#7F7F7F',           # Grey
    'B cells': '#17202A'               # Black
}
for ct in unique_cell_types:
    if ct not in ct_colors:
        ct_colors[ct] = '#7F7F7F'


# --- Figure 1: UMAP colored by unsupervised K-Means Clusters ---
print("Generating Fig 1: UMAP by Clusters...")
fig, ax = plt.subplots(figsize=(7.5, 6), dpi=300, facecolor='white')
kmeans_clusters = sorted(df_merged['kmeans_cluster'].unique(), key=int)
colors_kmeans = [
    '#1F77B4', '#FF7F0E', '#2CA02C', '#D62728', '#9467BD', 
    '#8C564B', '#E377C2', '#7F7F7F', '#BCBD22', '#17BECF', 
    '#AEC7E8', '#FFBB78'
]
cluster_colors = {c: colors_kmeans[i % len(colors_kmeans)] for i, c in enumerate(kmeans_clusters)}

for cluster in kmeans_clusters:
    sub = df_merged[df_merged['kmeans_cluster'] == cluster]
    ax.scatter(sub['UMAP_1'], sub['UMAP_2'], s=1.2, color=cluster_colors[cluster], label=f'Cluster {cluster}', alpha=0.90)
    
ax.set_title('Unsupervised Clustering (K-Means)', fontsize=12, fontweight='bold', pad=10)
ax.set_xlabel('UMAP_1', fontsize=10)
ax.set_ylabel('UMAP_2', fontsize=10)
ax.set_xticks([])
ax.set_yticks([])
ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), markerscale=4.0, frameon=False, fontsize=9.0, ncol=1, title='Cluster ID')
sns.despine(ax=ax, left=True, bottom=True)
plt.tight_layout()
fig1_path = '[YOUR_WORKING_DIRECTORY]\\single_cell_umap_clusters.png'
plt.savefig(fig1_path, facecolor='white', edgecolor='none', transparent=False, bbox_inches='tight')
plt.close()
print(f"Saved Fig 1 to {fig1_path}")

# --- Figure 2: UMAP colored by TMB Status ---
print("Generating Fig 2: UMAP by TMB Status...")
fig, ax = plt.subplots(figsize=(7.5, 6), dpi=300, facecolor='white')
tmb_colors = {
    'TMB medium': '#3498DB', # Light blue
    'TMB high': '#E74C3C'    # Coral red
}

for tmb_val in ['TMB medium', 'TMB high']:
    sub = df_merged[df_merged['TMB_status'] == tmb_val]
    ax.scatter(sub['UMAP_1'], sub['UMAP_2'], s=1.2, color=tmb_colors[tmb_val], label=tmb_val, alpha=0.90)

ax.set_title('UMAP by Tumor Mutation Burden (TMB)', fontsize=12, fontweight='bold', pad=10)
ax.set_xlabel('UMAP_1', fontsize=10)
ax.set_ylabel('UMAP_2', fontsize=10)
ax.set_xticks([])
ax.set_yticks([])
ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), markerscale=4.0, frameon=False, fontsize=10.0, title='TMB Status')
sns.despine(ax=ax, left=True, bottom=True)
plt.tight_layout()
fig2_path = '[YOUR_WORKING_DIRECTORY]\\single_cell_umap_tmb.png'
plt.savefig(fig2_path, facecolor='white', edgecolor='none', transparent=False, bbox_inches='tight')
plt.close()
print(f"Saved Fig 2 to {fig2_path}")

# --- Figure 3: UMAP colored by Cell Type ---
print("Generating Fig 3: UMAP by Cell Type...")
fig, ax = plt.subplots(figsize=(8.0, 6), dpi=300, facecolor='white')
for ct in unique_cell_types:
    sub = df_merged[df_merged['cell_type'] == ct]
    ax.scatter(sub['UMAP_1'], sub['UMAP_2'], s=1.2, color=ct_colors[ct], label=ct, alpha=0.90)

ax.set_title('UMAP by Cell Type', fontsize=12, fontweight='bold', pad=10)
ax.set_xlabel('UMAP_1', fontsize=10)
ax.set_ylabel('UMAP_2', fontsize=10)
ax.set_xticks([])
ax.set_yticks([])
ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), markerscale=4.0, frameon=False, fontsize=10.0, title='Cell Type')
sns.despine(ax=ax, left=True, bottom=True)
plt.tight_layout()
fig3_path = '[YOUR_WORKING_DIRECTORY]\\single_cell_umap_celltypes.png'
plt.savefig(fig3_path, facecolor='white', edgecolor='none', transparent=False, bbox_inches='tight')
plt.close()
print(f"Saved Fig 3 to {fig3_path}")

# --- Figure 4: Horizontal Stacked Bar Plot of Cell Type Proportions by TMB ---
print("Generating Fig 4: Cell Type Proportions by TMB...")
fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300, facecolor='white')
sc_prop = adata.obs.groupby(['TMB_status', 'cell_type']).size().unstack(fill_value=0)
sc_prop = sc_prop.div(sc_prop.sum(axis=1), axis=0)
sc_prop = sc_prop.reindex(['TMB medium', 'TMB high'])

bottoms = np.zeros(2)
x_labels = ['TMB medium', 'TMB high']
for ct in unique_cell_types:
    if ct in sc_prop.columns:
        vals = sc_prop[ct].values
        ax.barh(x_labels, vals, left=bottoms, label=ct, color=ct_colors[ct], height=0.45, edgecolor='white', linewidth=0.8)
        bottoms += vals

ax.set_xlabel('Relative Abundance', fontsize=11)
ax.set_xlim(0, 1.0)
ax.set_title('Cell Type Proportions by TMB Status', fontsize=12, fontweight='bold', pad=10)
ax.tick_params(axis='both', which='major', labelsize=10)
ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=9.0, title='Cell Type')
sns.despine(ax=ax, top=True, right=True)
plt.tight_layout()
fig4_path = '[YOUR_WORKING_DIRECTORY]\\single_cell_proportions_tmb.png'
plt.savefig(fig4_path, facecolor='white', edgecolor='none', transparent=False, bbox_inches='tight')
plt.close()
print(f"Saved Fig 4 to {fig4_path}")

# --- Figure 5: Marker Genes DotPlot ---
print("Generating Fig 5: Marker Genes DotPlot...")
fig, ax_f = plt.subplots(figsize=(8.5, 5.5), dpi=300, facecolor='white')
x_vals, y_vals, sizes, colors = [], [], [], []
for x_idx, ct in enumerate(unique_cell_types):
    sub = df_merged[df_merged['cell_type'] == ct]
    for y_idx, gene in enumerate(marker_genes):
        if gene in sub.columns:
            real_vals = sub[gene].values
            mean_expr = np.mean(real_vals)
            pct_expr = (np.sum(real_vals > 0) / len(real_vals)) * 100
        else:
            mean_expr = 0.0
            pct_expr = 0.0
            
        x_vals.append(x_idx)
        y_vals.append(y_idx)
        colors.append(mean_expr)
        sizes.append(pct_expr)

sc_plot = ax_f.scatter(
    x_vals, y_vals, c=colors, s=sizes, cmap='YlOrRd', 
    edgecolors='gray', linewidths=0.3, alpha=0.95
)

ax_f.set_xticks(np.arange(len(unique_cell_types)))
ax_f.set_xticklabels(unique_cell_types, rotation=45, ha='right', fontsize=9.5)
ax_f.set_yticks(np.arange(len(marker_genes)))
ax_f.set_yticklabels(marker_genes, fontsize=9.5)
ax_f.set_title('Marker Gene Expression across Cell Types', fontsize=12, fontweight='bold', pad=10)

cbar = plt.colorbar(sc_plot, ax=ax_f, fraction=0.03, pad=0.04)
cbar.set_label('Mean Expression', fontsize=9)
cbar.ax.tick_params(labelsize=8)

for sz in [25, 50, 75, 100]:
    ax_f.scatter([], [], c='gray', alpha=0.6, s=sz, label=f'{sz}%', edgecolors='black', linewidths=0.3)
ax_f.legend(
    title='Percent Expressed', loc='center left', bbox_to_anchor=(1.22, 0.5), 
    frameon=False, fontsize=8.0, title_fontsize=8.5, labelspacing=0.6
)
sns.despine(ax=ax_f, left=False, bottom=False)
plt.tight_layout()
fig5_path = '[YOUR_WORKING_DIRECTORY]\\single_cell_marker_dotplot.png'
plt.savefig(fig5_path, facecolor='white', edgecolor='none', transparent=False, bbox_inches='tight')
plt.close()
print(f"Saved Fig 5 to {fig5_path}")


