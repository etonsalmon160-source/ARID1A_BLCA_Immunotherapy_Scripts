import os
import zarr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tifffile

def main():
    base_dir = "/home/eto/bladder_spatial/data/extracted/LCCC_553_Xenium"
    output_dir = "/home/eto/bladder_spatial/output"
    
    cells_df = pd.read_csv(os.path.join(output_dir, "all_cells_meta.csv"))
    rx_min, ry_min, rx_max, ry_max = 5252.2, 6599.0, 5652.2, 6999.0
    roi_cells_df = cells_df[
        (cells_df['x'] >= rx_min) & (cells_df['x'] < rx_max) &
        (cells_df['y'] >= ry_min) & (cells_df['y'] < ry_max)
    ]
    
    trans_store = zarr.ZipStore(os.path.join(base_dir, "transcripts.zarr.zip"), mode='r')
    trans_grp = zarr.group(store=trans_store)
    grids_grp = trans_grp['grids/0']
    gene_names = trans_grp.attrs['gene_names']
    
    gx_min, gx_max = int(rx_min // 250), int(rx_max // 250)
    gy_min, gy_max = int(ry_min // 250), int(ry_max // 250)
    
    roi_transcripts = []
    for gx in range(gx_min, gx_max + 1):
        for gy in range(gy_min, gy_max + 1):
            grid_key = f"{gx},{gy}"
            if grid_key in grids_grp:
                g = grids_grp[grid_key]
                loc = g['location'][:, :2]
                genes = g['gene_identity'][:, 0]
                val = g['valid'][:, 0]
                mask = (val == 1) & (loc[:, 0] >= rx_min) & (loc[:, 0] < rx_max) & (loc[:, 1] >= ry_min) & (loc[:, 1] < ry_max)
                loc_filtered = loc[mask]
                genes_filtered = genes[mask]
                for l, g_idx in zip(loc_filtered, genes_filtered):
                    roi_transcripts.append({'x': l[0], 'y': l[1], 'gene': gene_names[g_idx]})
    trans_df = pd.DataFrame(roi_transcripts)
    
    cells_store = zarr.ZipStore(os.path.join(base_dir, "cells.zarr.zip"), mode='r')
    cells_grp = zarr.group(store=cells_store)
    p0 = cells_grp['polygon_sets/0']
    vertices = p0['vertices'][:]
    num_vertices = p0['num_vertices'][:]
    
    morph_path = os.path.join(base_dir, "morphology.ome.tif")
    with tifffile.TiffFile(morph_path) as tif:
        store = tif.series[0].levels[0].aszarr()
        z_grp = zarr.open(store, mode='r')
        z_arr = z_grp['0']
        
        pixel_size_l0 = 0.2125
        py_min = int(ry_min / pixel_size_l0)
        py_max = int(ry_max / pixel_size_l0)
        px_min = int(rx_min / pixel_size_l0)
        px_max = int(rx_max / pixel_size_l0)
        
        morph_roi = z_arr[7, py_min:py_max, px_min:px_max]
    
    p1, p99 = np.percentile(morph_roi, (1, 99.5))
    morph_roi_norm = np.clip(morph_roi, p1, p99)
    morph_roi_norm = ((morph_roi_norm - p1) / (p99 - p1) * 255).astype(np.uint8)
    
    panels = [
        {
            'title': 'Panel A: ARID1A-High Tumor Cells',
            'cell_filter': lambda r: (r['cell_type'] == 'Tumor_Epithelial') & (r['arid1a_status'] == 'ARID1A-High'),
            'genes': ['ARID1A'],
            'color': '#00E5FF',
            'outline_color': '#00E5FF'
        },
        {
            'title': 'Panel B: ARID1A-Low Tumor Cells',
            'cell_filter': lambda r: (r['cell_type'] == 'Tumor_Epithelial') & (r['arid1a_status'] == 'ARID1A-Low'),
            'genes': ['ARID1A'],
            'color': '#FF00FF',
            'outline_color': '#FF00FF'
        },
        {
            'title': 'Panel C: Tex (Exhausted T Cells)',
            'cell_filter': lambda r: r['cell_type'] == 'Tex',
            'genes': ['PDCD1', 'HAVCR2'],
            'color': '#FF3333',
            'outline_color': '#FF3333'
        },
        {
            'title': 'Panel D: M2 Macrophages',
            'cell_filter': lambda r: r['cell_type'] == 'M2_Macrophages',
            'genes': ['CD68', 'CD163', 'MRC1'],
            'color': '#FF9900',
            'outline_color': '#FF9900'
        },
        {
            'title': 'Panel E: Fibroblasts',
            'cell_filter': lambda r: r['cell_type'] == 'Fibroblasts',
            'genes': ['ACTA2', 'FAP'],
            'color': '#00FF00',
            'outline_color': '#00FF00'
        }
    ]
    
    fig, axes = plt.subplots(1, 5, figsize=(25, 5), dpi=300)
    for idx, panel in enumerate(panels):
        ax = axes[idx]
        ax.imshow(morph_roi_norm, cmap='gray', extent=[rx_min, rx_max, ry_max, ry_min])
        cells_sub = roi_cells_df[roi_cells_df.apply(panel['cell_filter'], axis=1)]
        for _, cell_row in cells_sub.iterrows():
            cell_idx = cell_row.name
            poly_pts = vertices[cell_idx].reshape(-1, 2)
            n_v = num_vertices[cell_idx]
            poly_pts = poly_pts[:n_v]
            polygon = patches.Polygon(poly_pts, closed=True, fill=False, edgecolor=panel['outline_color'], linewidth=0.8, alpha=0.8)
            ax.add_patch(polygon)
        if len(trans_df) > 0:
            trans_sub = trans_df[trans_df['gene'].isin(panel['genes'])]
            if len(trans_sub) > 0:
                ax.scatter(trans_sub['x'], trans_sub['y'], s=2.5, c=panel['color'], marker='o', alpha=0.9, label=', '.join(panel['genes']))
                ax.legend(loc='upper right', frameon=True, facecolor='black', edgecolor='white', labelcolor='white', fontsize=8)
        ax.set_xlim(rx_min, rx_max)
        ax.set_ylim(ry_max, ry_min)
        ax.set_title(panel['title'], color='white', fontsize=12, pad=10)
        ax.axis('off')
    fig.patch.set_facecolor('#0f0f0f')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fluorescent_channels_split.png"), bbox_inches='tight', facecolor=fig.get_facecolor(), dpi=300)
    plt.close()
    print("Multi-channel split plot updated!")

if __name__ == "__main__":
    main()
