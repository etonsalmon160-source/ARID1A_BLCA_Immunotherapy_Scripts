import os
import urllib.request
import gzip
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

# Define expanded set of genes including T-cell subpopulation and exhaustion markers
marker_genes = [
    'EPCAM', 'KRT19', 'CDH1',         # Epithelial
    'CD3E', 'CD3D', 'CD2', 'NKG7',    # T and NK cells
    'PECAM1', 'VWF',                 # Endothelial
    'LYZ', 'CD68', 'C1QC',           # Myeloid
    'FAP', 'COL1A1', 'DCN',          # Fibroblasts
    'MCAM', 'RGS5', 'ACTA2',         # Pericytes
    'CD19', 'MS4A1', 'CD79A',        # B cells
    'MZB1', 'SDC1', 'JCHAIN',        # Plasma cells
    'TPSAB1', 'TPSB2', 'KIT'         # Mast cells
]

t_cell_subset_markers = {
    'CD8A': 'CD8A', 'CD8B': 'CD8B',
    'CD4': 'CD4', 'FOXP3': 'FOXP3', 'IL2RA': 'IL2RA',
    'NKG7': 'NKG7', 'GNLY': 'GNLY', 'GZMB': 'GZMB', 'PRF1': 'PRF1',
    'PDCD1': 'PDCD1', 'CTLA4': 'CTLA4', 'HAVCR2': 'HAVCR2',
    'LAG3': 'LAG3', 'TIGIT': 'TIGIT', 'TOX': 'TOX'
}

all_study_genes = sorted(list(set(marker_genes) | set(t_cell_subset_markers.keys())))

# 2. Establish HVGs
print("Computing highly variable genes (HVGs) from BC1...")
local_path_bc1 = os.path.join(dest_dir, "BC1_matrix.txt.gz")
gene_vars_bc1 = {}
with gzip.open(local_path_bc1, 'rt') as f:
    header = f.readline().strip().split('\t')
    for line in f:
        parts = line.strip().split('\t')
        symbol = parts[1]
        expr_vals = [float(x) for x in parts[2:]]
        if np.mean(expr_vals) > 0.05:
            gene_vars_bc1[symbol] = np.var(expr_vals)
top_bc1 = sorted(gene_vars_bc1.keys(), key=lambda x: gene_vars_bc1[x], reverse=True)[:1000]

print("Computing highly variable genes (HVGs) from BC159...")
local_path_gse145137 = os.path.join(dest_dir, "GSM4307111_matrix.txt.gz")
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

selected_genes = set(all_study_genes) | set(top_bc1) | set(top_bc159)
selected_genes = sorted(list(selected_genes))
print(f"Established aligned high-variance feature space with {len(selected_genes)} genes.")

# 3. Stream and Load data
adatas = []
for name, url in urls_gse135337.items():
    local_path = os.path.join(dest_dir, f"{name}_matrix.txt.gz")
    print(f"Streaming and extracting aligned genes for {name}...")
    extracted = {}
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
            if symbol in selected_genes:
                extracted[symbol] = list(expr_vals)
                
    df_expr = pd.DataFrame(extracted, index=barcodes)
    for m in selected_genes:
        if m not in df_expr.columns:
            df_expr[m] = 0.0
            
    pct_counts_mt = (mt_counts / (total_counts + 1e-6)) * 100
    qc_mask = (total_counts >= 500) & (total_counts <= 15000) & \
              (detected_genes >= 200) & (detected_genes <= 4000) & \
              (pct_counts_mt <= 20.0)
    df_qc = df_expr[qc_mask].copy()
    
    all_features = [c for c in df_qc.columns if c not in ['Patient', 'Cohort', 'cell_type']]
    X_sparse = scipy.sparse.csr_matrix(df_qc[all_features].values)
    obs_df = pd.DataFrame({'Patient': name, 'Cohort': 'GSE135337'}, index=df_qc.index)
    adata_pat = sc.AnnData(X=X_sparse, obs=obs_df)
    adata_pat.var_names = all_features
    
    try:
        sc.external.pp.scrublet(adata_pat, verbose=False)
        adata_pat = adata_pat[~adata_pat.obs['predicted_doublet']].copy()
    except Exception as e:
        pass
        
    adatas.append(adata_pat)
    del df_expr, df_qc
    gc.collect()

# Process BC159
print("Streaming and extracting aligned genes for BC159...")
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

pct_counts_mt = (mt_counts / (total_counts + 1e-6)) * 100
qc_mask = (detected_genes >= 200) & (pct_counts_mt <= 20.0)
df_gse145_qc = df_gse145_raw[qc_mask].copy()

X_sparse_145 = scipy.sparse.csr_matrix(df_gse145_qc[all_features].values)
obs_df_145 = pd.DataFrame({'Patient': 'BC159', 'Cohort': 'GSE145137'}, index=df_gse145_qc.index)
adata_pat145 = sc.AnnData(X=X_sparse_145, obs=obs_df_145)
adata_pat145.var_names = all_features

try:
    sc.external.pp.scrublet(adata_pat145, verbose=False)
    adata_pat145 = adata_pat145[~adata_pat145.obs['predicted_doublet']].copy()
except Exception as e:
    pass

adatas.append(adata_pat145)
del df_gse145_raw, df_gse145_qc
gc.collect()

# 4. Concatenate and Pre-process
print("Concatenating matrices...")
adata = sc.concat(adatas, join='outer', fill_value=0.0)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
adata.raw = adata

# Scale
sc.pp.scale(adata, max_value=10)
sc.tl.pca(adata, n_comps=30)

# Harmony batch correction
print("Running Harmony...")
ho = harmonypy.run_harmony(adata.obsm['X_pca'], adata.obs, ['Patient'], verbose=True)
adata.obsm['X_pca_harmony'] = ho.Z_corr

# First-level clustering
kmeans_global = KMeans(n_clusters=12, random_state=42, n_init=10)
adata.obs['global_cluster'] = kmeans_global.fit_predict(adata.obsm['X_pca_harmony']).astype(str)

# Calculate scores to find T/NK cluster
print("Identifying T/NK cell cluster...")
t_scores = {}
for cluster in adata.obs['global_cluster'].unique():
    t_expr = []
    for g in ['CD3D', 'CD3E', 'CD2', 'NKG7']:
        g_idx = adata.var_names.get_loc(g)
        expr_col = adata.X[(adata.obs['global_cluster'] == cluster).values, g_idx]
        t_expr.append(np.mean(expr_col.toarray()) if hasattr(expr_col, 'toarray') else np.mean(expr_col))
    t_scores[cluster] = np.mean(t_expr)

best_t_cluster = max(t_scores, key=t_scores.get)
print(f"Global Cluster {best_t_cluster} identified as T and NK cells with score {t_scores[best_t_cluster]:.4f}.")

# Subset T and NK cells for high-resolution sub-clustering!
# Also fallback to cells with CD3D > 0.5 or CD3E > 0.5 or NKG7 > 0.5 to capture all T/NK cells
t_cells_mask = (adata.obs['global_cluster'] == best_t_cluster).values
cd3_expr = adata.raw[:, 'CD3D'].X.toarray().flatten() + adata.raw[:, 'CD3E'].X.toarray().flatten() + adata.raw[:, 'NKG7'].X.toarray().flatten()
t_cells_fallback = cd3_expr > 0.5
t_mask = t_cells_mask | t_cells_fallback

adata_t = adata[t_mask].copy()
print(f"Subsetted {len(adata_t)} T and NK cells for high-resolution sub-clustering.")

# Recompute PCA and Harmony on T-cell subset to resolve fine exhaustion sub-states
# To eliminate patient-specific batch effects from global highly variable genes, we run PCA on targeted T/NK biological markers
t_features = [g for g in all_study_genes if g in adata_t.var_names]
adata_t_sub = adata_t[:, t_features].copy()
sc.tl.pca(adata_t_sub, n_comps=min(12, len(t_features)-1))
ho_t = harmonypy.run_harmony(adata_t_sub.obsm['X_pca'], adata_t_sub.obs, ['Patient'], verbose=False)
adata_t.obsm['X_pca_harmony'] = ho_t.Z_corr

# Run K-Means sub-clustering (K=5 sub-states of T & NK cells)
kmeans_t = KMeans(n_clusters=5, random_state=42, n_init=10)
adata_t.obs['sub_cluster'] = kmeans_t.fit_predict(adata_t.obsm['X_pca_harmony']).astype(str)

# Annotate T-cell sub-clusters using key subset and exhaustion markers
print("Annotating T-cell sub-clusters...")
t_sub_markers = {
    'CD8+ Exhausted T cells': ['CD8A', 'CD8B', 'PDCD1', 'CTLA4', 'TIGIT', 'TOX'],
    'CD8+ Cytotoxic T cells': ['CD8A', 'CD8B', 'GNLY', 'GZMB', 'PRF1'],
    'CD4+ Tregs': ['CD4', 'FOXP3', 'IL2RA'],
    'CD4+ Helper T cells': ['CD4'],
    'NK cells': ['NKG7', 'GNLY']
}

t_cluster_annotation = {}
for cluster in sorted(adata_t.obs['sub_cluster'].unique()):
    cluster_cells = (adata_t.obs['sub_cluster'] == cluster).values
    def get_mean(gene):
        if gene in adata_t.raw.var_names:
            g_idx = adata_t.raw.var_names.get_loc(gene)
            col = adata_t.raw[cluster_cells, g_idx].X
            return np.mean(col.toarray()) if hasattr(col, 'toarray') else np.mean(col)
        return 0.0
    
    cd8_val = (get_mean('CD8A') + get_mean('CD8B')) / 2
    cd4_val = get_mean('CD4')
    nkg7_val = (get_mean('NKG7') + get_mean('GNLY')) / 2
    cd3_val = (get_mean('CD3D') + get_mean('CD3E')) / 2
    
    if nkg7_val > 1.2 and cd3_val < 0.6:
        ann = 'NK cells'
    elif cd4_val > cd8_val:
        foxp3_val = get_mean('FOXP3')
        il2ra_val = get_mean('IL2RA')
        if foxp3_val > 0.15 or il2ra_val > 0.15:
            ann = 'CD4+ Tregs'
        else:
            ann = 'CD4+ Helper T cells'
    else:
        pd1_val = get_mean('PDCD1')
        tox_val = get_mean('TOX')
        ctla4_val = get_mean('CTLA4')
        tigit_val = get_mean('TIGIT')
        exhaustion_score = np.mean([pd1_val, ctla4_val, tigit_val, tox_val])
        
        if pd1_val > 0.08 or exhaustion_score > 0.08:
            ann = 'CD8+ Exhausted T cells'
        else:
            ann = 'CD8+ Cytotoxic T cells'
            
    t_cluster_annotation[cluster] = ann
    print(f"T-SubCluster {cluster} -> CD3: {cd3_val:.3f}, CD8: {cd8_val:.3f}, CD4: {cd4_val:.3f}, NKG7: {nkg7_val:.3f} -> Annotated as: {ann}")


# Map annotations back to cells
adata_t.obs['t_cell_subset'] = adata_t.obs['sub_cluster'].map(t_cluster_annotation)
print("T-cell Subsets Distribution:")
print(adata_t.obs['t_cell_subset'].value_counts())

# Project T-cell subsets onto a fine UMAP space
sc.pp.neighbors(adata_t, n_neighbors=15, use_rep='X_pca_harmony')
sc.tl.umap(adata_t, min_dist=0.3, random_state=42)

df_t = pd.DataFrame(adata_t.obsm['X_umap'], columns=['UMAP_1', 'UMAP_2'], index=adata_t.obs_names)
df_t = pd.concat([df_t, adata_t.obs], axis=1)

# Extract marker expressions
for gene in t_cell_subset_markers.keys():
    gene_idx = adata_t.var_names.get_loc(gene)
    val_col = adata_t.X[:, gene_idx]
    df_t[gene] = val_col.toarray().flatten() if hasattr(val_col, 'toarray') else np.array(val_col).flatten()

# TMB definitions
tmb_map = {
    'BC1': 'TMB medium', 'BC2': 'TMB high', 'BC3': 'TMB medium',
    'BC5': 'TMB high', 'BC6': 'TMB high', 'BC7': 'TMB high', 'BC159': 'TMB high'
}
df_t['TMB_status'] = df_t['Patient'].map(tmb_map)
adata_t.obs['TMB_status'] = adata_t.obs['Patient'].map(tmb_map)

# ARID1A mutation definitions (from WES somatic mutations)
# BC1: WT, BC2: WT, BC3: WT, BC4: WT, BC5: WT, BC6: WT, BC7: WT, BC159: Mutant (ARID1A p.Q685*)
# Wait, let's map patient ARID1A status
arid1a_map = {
    'BC1': 'WT', 'BC2': 'WT', 'BC3': 'WT',
    'BC5': 'WT', 'BC6': 'WT', 'BC7': 'WT', 'BC159': 'ARID1A Mutant'
}
df_t['ARID1A_status'] = df_t['Patient'].map(arid1a_map)
adata_t.obs['ARID1A_status'] = adata_t.obs['Patient'].map(arid1a_map)

# --- Figure 1: UMAP of T-cell Subsets (including Exhausted CD8+ T cells) ---
print("Generating Fig 1: T-cell Subsets UMAP...")
fig, ax = plt.subplots(figsize=(7.5, 6), dpi=300, facecolor='white')
t_subsets = sorted(df_t['t_cell_subset'].unique())
colors_t = {
    'CD8+ Exhausted T cells': '#E74C3C',    # Bright Coral Red (Exhausted)
    'CD8+ Cytotoxic T cells': '#2ECC71',    # Green
    'CD4+ Tregs': '#F1C40F',               # Yellow
    'CD4+ Helper T cells': '#3498DB',       # Light Blue
    'NK cells': '#9B59B6'                  # Purple
}

for subset in t_subsets:
    sub = df_t[df_t['t_cell_subset'] == subset]
    ax.scatter(sub['UMAP_1'], sub['UMAP_2'], s=3.5, color=colors_t.get(subset, '#7F7F7F'), label=subset, alpha=0.90)

ax.set_title('UMAP of T & NK Cell Subpopulations', fontsize=12, fontweight='bold', pad=10)
ax.set_xlabel('UMAP_1', fontsize=10)
ax.set_ylabel('UMAP_2', fontsize=10)
ax.set_xticks([])
ax.set_yticks([])
ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), markerscale=3.0, frameon=False, fontsize=10.0, title='T-cell Subsets')
sns.despine(ax=ax, left=True, bottom=True)
plt.tight_layout()
fig1_path = '[YOUR_WORKING_DIRECTORY]\\t_cell_umap_subsets.png'
plt.savefig(fig1_path, facecolor='white', edgecolor='none', transparent=False, bbox_inches='tight')
plt.close()
print(f"Saved T-cell sub-UMAP to {fig1_path}")

# --- Figure 2: Exhaustion Marker Genes DotPlot ---
print("Generating Fig 2: Exhaustion DotPlot...")
fig, ax_f = plt.subplots(figsize=(5.2, 5.5), dpi=300, facecolor='white')
x_vals, y_vals, sizes, colors = [], [], [], []
exhaustion_marker_list = ['CD8A', 'CD8B', 'CD4', 'FOXP3', 'IL2RA', 'NKG7', 'GNLY', 'PDCD1', 'CTLA4', 'HAVCR2', 'LAG3', 'TIGIT', 'TOX']

for x_idx, subset in enumerate(t_subsets):
    sub = df_t[df_t['t_cell_subset'] == subset]
    for y_idx, gene in enumerate(exhaustion_marker_list):
        real_vals = sub[gene].values
        mean_expr = np.mean(real_vals)
        pct_expr = (np.sum(real_vals > 0) / len(real_vals)) * 100
        
        x_vals.append(x_idx)
        y_vals.append(y_idx)
        colors.append(mean_expr)
        sizes.append(pct_expr)

sc_plot = ax_f.scatter(
    x_vals, y_vals, c=colors, s=np.array(sizes) * 1.8, cmap='YlOrRd', 
    edgecolors='gray', linewidths=0.3, alpha=0.95
)

ax_f.set_xticks(np.arange(len(t_subsets)))
ax_f.set_xticklabels(t_subsets, rotation=30, ha='right', fontsize=9.5)
ax_f.set_yticks(np.arange(len(exhaustion_marker_list)))
ax_f.set_yticklabels(exhaustion_marker_list, fontsize=9.5)
ax_f.set_title('Exhaustion & Subpopulation Markers\nacross T-cell Subsets', fontsize=11, fontweight='bold', pad=10)

cbar = plt.colorbar(sc_plot, ax=ax_f, fraction=0.03, pad=0.05)
cbar.set_label('Mean Expression', fontsize=9)
cbar.ax.tick_params(labelsize=8)

for sz in [25, 50, 75, 100]:
    ax_f.scatter([], [], c='gray', alpha=0.6, s=sz * 1.8, label=f'{sz}%', edgecolors='black', linewidths=0.3)
ax_f.legend(
    title='Percent Expressed', loc='center left', bbox_to_anchor=(1.30, 0.5), 
    frameon=False, fontsize=8.0, title_fontsize=8.5, labelspacing=0.6
)
sns.despine(ax=ax_f, left=False, bottom=False)
plt.tight_layout()
fig2_path = '[YOUR_WORKING_DIRECTORY]\\t_cell_exhaustion_dotplot.png'
plt.savefig(fig2_path, facecolor='white', edgecolor='none', transparent=False, bbox_inches='tight')
plt.close()
print(f"Saved Exhaustion DotPlot to {fig2_path}")

# --- Figure 3: Proportions of T-cell Subsets by TMB ---
print("Generating Fig 3: T-cell Subset Proportions by TMB...")
fig, ax = plt.subplots(figsize=(7.5, 4.0), dpi=300, facecolor='white')
t_prop = adata_t.obs.groupby(['TMB_status', 't_cell_subset']).size().unstack(fill_value=0)
t_prop = t_prop.div(t_prop.sum(axis=1), axis=0)
t_prop = t_prop.reindex(['TMB medium', 'TMB high'])

bottoms = np.zeros(2)
x_labels = ['TMB medium', 'TMB high']
for subset in t_subsets:
    if subset in t_prop.columns:
        vals = t_prop[subset].values
        ax.barh(x_labels, vals, left=bottoms, label=subset, color=colors_t.get(subset, '#7F7F7F'), height=0.40, edgecolor='white', linewidth=0.8)
        bottoms += vals

ax.set_xlabel('Relative Abundance', fontsize=11)
ax.set_xlim(0, 1.0)
ax.set_title('T-cell Subset Abundance by TMB Status', fontsize=12, fontweight='bold', pad=10)
ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=9.0, title='T-cell Subsets')
sns.despine(ax=ax, top=True, right=True)
plt.tight_layout()
fig3_path = '[YOUR_WORKING_DIRECTORY]\\t_cell_proportions_tmb.png'
plt.savefig(fig3_path, facecolor='white', edgecolor='none', transparent=False, bbox_inches='tight')
plt.close()
print(f"Saved T-cell TMB proportions to {fig3_path}")

# --- Figure 4: Proportions of T-cell Subsets in ARID1A Mutant vs WT ---
print("Generating Fig 4: T-cell Subset Proportions by ARID1A Status...")
fig, ax = plt.subplots(figsize=(7.5, 4.0), dpi=300, facecolor='white')
t_prop_arid = adata_t.obs.groupby(['ARID1A_status', 't_cell_subset']).size().unstack(fill_value=0)
t_prop_arid = t_prop_arid.div(t_prop_arid.sum(axis=1), axis=0)
t_prop_arid = t_prop_arid.reindex(['WT', 'ARID1A Mutant'])

bottoms = np.zeros(2)
x_labels = ['ARID1A Wildtype', 'ARID1A Mutant (BC159)']
for subset in t_subsets:
    if subset in t_prop_arid.columns:
        vals = t_prop_arid[subset].values
        ax.barh(x_labels, vals, left=bottoms, label=subset, color=colors_t.get(subset, '#7F7F7F'), height=0.40, edgecolor='white', linewidth=0.8)
        bottoms += vals

ax.set_xlabel('Relative Abundance', fontsize=11)
ax.set_xlim(0, 1.0)
ax.set_title('T-cell Subset Abundance by ARID1A Status (High TMB Cohort)', fontsize=12, fontweight='bold', pad=10)
ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=9.0, title='T-cell Subsets')
sns.despine(ax=ax, top=True, right=True)
plt.tight_layout()
fig4_path = '[YOUR_WORKING_DIRECTORY]\\t_cell_proportions_arid1a.png'
plt.savefig(fig4_path, facecolor='white', edgecolor='none', transparent=False, bbox_inches='tight')
plt.close()
print(f"Saved T-cell ARID1A proportions to {fig4_path}")

print("T-cell subsetting and exhaustion analysis completed successfully!")
