import os
import zarr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tifffile

def main():
    base_dir = "/home/eto/bladder_spatial/data/extracted/LCCC_553_Xenium"
    output_dir = "/home/eto/bladder_spatial/output/individual"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load cell metadata
    print("Loading cell metadata...")
    cells_df = pd.read_csv("/home/eto/bladder_spatial/output/all_cells_meta.csv")
    
    x_min, x_max = cells_df['x'].min(), cells_df['x'].max()
    y_min, y_max = cells_df['y'].min(), cells_df['y'].max()
    
    # Selected ROI (best ROI)
    rx_min, ry_min, rx_max, ry_max = 5252.2, 6599.0, 5652.2, 6999.0
    
    # Filter cells in ROI
    roi_cells_df = cells_df[
        (cells_df['x'] >= rx_min) & (cells_df['x'] < rx_max) &
        (cells_df['y'] >= ry_min) & (cells_df['y'] < ry_max)
    ]
    
    # 2. Load cell outlines (polygons)
    print("Loading cell outlines...")
    cells_store = zarr.ZipStore(os.path.join(base_dir, "cells.zarr.zip"), mode='r')
    cells_grp = zarr.group(store=cells_store)
    p0 = cells_grp['polygon_sets/0']
    vertices = p0['vertices'][:]
    num_vertices = p0['num_vertices'][:]
    
    # 3. Load transcripts in ROI
    print("Loading transcripts in ROI...")
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
                    roi_transcripts.append({
                        'x': l[0],
                        'y': l[1],
                        'gene': gene_names[g_idx]
                    })
                    
    trans_df = pd.DataFrame(roi_transcripts)
    
    # 4. Load morphology images
    morph_path = os.path.join(base_dir, "morphology.ome.tif")
    print("Loading morphology images...")
    with tifffile.TiffFile(morph_path) as tif:
        morph_level5 = tif.series[0].levels[5].asarray()
        if len(morph_level5.shape) == 3:
            morph_level5 = morph_level5[7]
            
        store = tif.series[0].levels[0].aszarr()
        z_grp = zarr.open(store, mode='r')
        z_arr = z_grp['0']
        
        pixel_size_l0 = 0.2125
        py_min = int(ry_min / pixel_size_l0)
        py_max = int(ry_max / pixel_size_l0)
        px_min = int(rx_min / pixel_size_l0)
        px_max = int(rx_max / pixel_size_l0)
        
        morph_roi = z_arr[7, py_min:py_max, px_min:px_max]
    
    # Normalize contrast for level 5 whole-slide overview
    p1_5, p99_5 = np.percentile(morph_level5, (1, 99))
    morph_l5_norm = np.clip(morph_level5, p1_5, p99_5)
    morph_l5_norm = ((morph_l5_norm - p1_5) / (p99_5 - p1_5) * 255).astype(np.uint8)
    
    p1_2, p99_2 = np.percentile(morph_roi, (1, 99.5))
    morph_roi_norm = np.clip(morph_roi, p1_2, p99_2)
    morph_roi_norm = ((morph_roi_norm - p1_2) / (p99_2 - p1_2) * 255).astype(np.uint8)
    
    # Define Panels with glowing fluorescent colors
    panels = [
        {
            'key': 'arid1a_high',
            'title_roi': 'ARID1A-High Tumor Cells (ROI)',
            'title_whole': 'ARID1A-High Tumor Cells (Whole-Slide)',
            'cell_filter': lambda r: (r['cell_type'] == 'Tumor_Epithelial') & (r['arid1a_status'] == 'ARID1A-High'),
            'genes': ['ARID1A'],
            'color': '#00E5FF', # Glowing Cyan
        },
        {
            'key': 'arid1a_low',
            'title_roi': 'ARID1A-Low Tumor Cells (ROI)',
            'title_whole': 'ARID1A-Low Tumor Cells (Whole-Slide)',
            'cell_filter': lambda r: (r['cell_type'] == 'Tumor_Epithelial') & (r['arid1a_status'] == 'ARID1A-Low'),
            'genes': ['ARID1A'],
            'color': '#FF00FF', # Glowing Magenta
        },
        {
            'key': 'tex',
            'title_roi': 'Tex (Exhausted T Cells) (ROI)',
            'title_whole': 'Tex (Exhausted T Cells) (Whole-Slide)',
            'cell_filter': lambda r: r['cell_type'] == 'Tex',
            'genes': ['PDCD1', 'HAVCR2'],
            'color': '#FF3333', # Glowing Red
        },
        {
            'key': 'm2_macrophages',
            'title_roi': 'M2 Macrophages (ROI)',
            'title_whole': 'M2 Macrophages (Whole-Slide)',
            'cell_filter': lambda r: r['cell_type'] == 'M2_Macrophages',
            'genes': ['CD68', 'CD163', 'MRC1'],
            'color': '#FF9900', # Glowing Gold/Orange
        },
        {
            'key': 'fibroblasts',
            'title_roi': 'Fibroblasts (ROI)',
            'title_whole': 'Fibroblasts (Whole-Slide)',
            'cell_filter': lambda r: r['cell_type'] == 'Fibroblasts',
            'genes': ['ACTA2', 'FAP'],
            'color': '#00FF00', # Glowing Green
        }
    ]
    
    for panel in panels:
        # ---- 1. Plot Zoom ROI Plot ----
        print(f"Generating individual ROI plot for {panel['key']}...")
        fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
        ax.imshow(morph_roi_norm, cmap='gray', extent=[rx_min, rx_max, ry_max, ry_min])
        
        cells_sub = roi_cells_df[roi_cells_df.apply(panel['cell_filter'], axis=1)]
        
        # Outlines
        for _, cell_row in cells_sub.iterrows():
            cell_idx = cell_row.name
            poly_pts = vertices[cell_idx].reshape(-1, 2)
            n_v = num_vertices[cell_idx]
            poly_pts = poly_pts[:n_v]
            polygon = patches.Polygon(poly_pts, closed=True, fill=False, edgecolor=panel['color'], linewidth=0.8, alpha=0.8)
            ax.add_patch(polygon)
            
        # Transcripts
        if len(trans_df) > 0:
            trans_sub = trans_df[trans_df['gene'].isin(panel['genes'])]
            if len(trans_sub) > 0:
                ax.scatter(trans_sub['x'], trans_sub['y'], s=3.0, c=panel['color'], marker='o', alpha=0.9, label=', '.join(panel['genes']))
                ax.legend(loc='upper right', frameon=True, facecolor='black', edgecolor='white', labelcolor='white', fontsize=8)
                
        ax.set_xlim(rx_min, rx_max)
        ax.set_ylim(ry_max, ry_min)
        ax.set_title(panel['title_roi'], color='white', fontsize=12, pad=10)
        ax.axis('off')
        fig.patch.set_facecolor('#0f0f0f')
        plt.savefig(os.path.join(output_dir, f"{panel['key']}_roi.png"), bbox_inches='tight', facecolor=fig.get_facecolor(), dpi=300)
        plt.close()
        
        # ---- 2. Plot Whole Slide Plot ----
        print(f"Generating individual whole-slide plot for {panel['key']}...")
        fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
        ax.imshow(morph_l5_norm, cmap='gray', extent=[x_min, x_max, y_max, y_min])
        
        # Filter all cells
        cells_all_sub = cells_df[cells_df.apply(panel['cell_filter'], axis=1)]
        ax.scatter(cells_all_sub['x'], cells_all_sub['y'], s=0.1, c=panel['color'], alpha=0.7, edgecolors='none')
        
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_max, y_min)
        ax.set_title(panel['title_whole'], color='white', fontsize=12, pad=10)
        ax.axis('off')
        fig.patch.set_facecolor('#0f0f0f')
        plt.savefig(os.path.join(output_dir, f"{panel['key']}_whole.png"), bbox_inches='tight', facecolor=fig.get_facecolor(), dpi=300)
        plt.close()
        
    print("All individual images generated successfully!")

if __name__ == "__main__":
    main()
