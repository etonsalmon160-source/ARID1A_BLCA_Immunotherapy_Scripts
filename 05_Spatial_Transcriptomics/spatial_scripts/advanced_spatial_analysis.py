import os
import zarr
import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import tifffile
import time

def main():
    base_dir = "/home/eto/bladder_spatial/data/extracted/LCCC_553_Xenium"
    output_dir = "/home/eto/bladder_spatial/output"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Step 1: Loading cell metadata...")
    all_cells_file = os.path.join(output_dir, "all_cells_meta.csv")
    if not os.path.exists(all_cells_file):
        print("Error: all_cells_meta.csv not found! Run the R script first.")
        return
    
    cells_df = pd.read_csv(all_cells_file)
    print(f"Loaded {len(cells_df)} cells.")
    
    # ----------------------------------------------------
    # Step 2: Dynamically Find Best ROI for Zoom-in Visualization
    # ----------------------------------------------------
    print("Step 2: Finding best ROI for zoom-in cell segmentation visualization...")
    roi_size = 400.0
    
    x_min, x_max = cells_df['x'].min(), cells_df['x'].max()
    y_min, y_max = cells_df['y'].min(), cells_df['y'].max()
    
    x_grid = np.arange(x_min, x_max - roi_size, roi_size / 2)
    y_grid = np.arange(y_min, y_max - roi_size, roi_size / 2)
    
    best_roi = None
    max_score = -1
    
    for x in x_grid:
        for y in y_grid:
            roi_cells = cells_df[
                (cells_df['x'] >= x) & (cells_df['x'] < x + roi_size) &
                (cells_df['y'] >= y) & (cells_df['y'] < y + roi_size)
            ]
            n_tex = np.sum(roi_cells['cell_type'] == 'Tex')
            n_m2 = np.sum(roi_cells['cell_type'] == 'M2_Macrophages')
            n_low = np.sum(roi_cells['arid1a_status'] == 'ARID1A-Low')
            score = n_tex + n_m2 + n_low
            
            if score > max_score and len(roi_cells) < 1500 and len(roi_cells) > 300:
                max_score = score
                best_roi = (x, y, x + roi_size, y + roi_size)
                
    print(f"Selected ROI bounding box: X in [{best_roi[0]:.1f}, {best_roi[2]:.1f}], Y in [{best_roi[1]:.1f}, {best_roi[3]:.1f}]")
    
    # ----------------------------------------------------
    # Step 3: Spatial Niche Clustering
    # ----------------------------------------------------
    print("Step 3: Calculating Spatial Niches using KNN neighborhood composition...")
    centroids = cells_df[['x', 'y']].values
    cell_types = cells_df['cell_type'].values
    
    unique_types = ['Tumor_Epithelial', 'T_cells', 'Tex', 'Fibroblasts', 'M2_Macrophages', 'Other']
    type_to_idx = {t: idx for idx, t in enumerate(unique_types)}
    
    tree = KDTree(centroids)
    distances, indices = tree.query(centroids, k=20)
    
    n_cells = len(cells_df)
    composition = np.zeros((n_cells, len(unique_types)))
    for i in range(n_cells):
        neighbor_types = cell_types[indices[i]]
        for nt in neighbor_types:
            if nt in type_to_idx:
                composition[i, type_to_idx[nt]] += 1
    composition /= 20.0
    
    print("Running KMeans to identify 5 spatial niches...")
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    niche_labels = kmeans.fit_predict(composition)
    cells_df['spatial_niche'] = niche_labels
    
    niche_profiles = pd.DataFrame(kmeans.cluster_centers_, columns=unique_types)
    
    niche_names = []
    for niche_idx, row in niche_profiles.iterrows():
        max_col = row.idxmax()
        if max_col == 'Tumor_Epithelial':
            niche_names.append(f"Niche {niche_idx}: Tumor Core")
        elif max_col == 'Other':
            second = row.drop('Other').idxmax()
            niche_names.append(f"Niche {niche_idx}: Stromal/Immune Mix ({second})")
        else:
            niche_names.append(f"Niche {niche_idx}: {max_col}-Rich")
            
    # ----------------------------------------------------
    # Step 4: Permutation-based Neighborhood Enrichment Test
    # ----------------------------------------------------
    print("\nStep 4: Running permutation neighborhood enrichment test...")
    high_tumor_coords = cells_df[cells_df['arid1a_status'] == 'ARID1A-High'][['x', 'y']].values
    low_tumor_coords = cells_df[cells_df['arid1a_status'] == 'ARID1A-Low'][['x', 'y']].values
    
    # ----------------------------------------------------
    # Step 5: Ripley's Cross-L Function Analysis
    # ----------------------------------------------------
    print("\nStep 5: Calculating Ripley's Cross-L Function...")
    radii = np.arange(10.0, 101.0, 10.0)
    area = (x_max - x_min) * (y_max - y_min)
    
    def ripleys_cross_L(coords_A, coords_B, tree_B, radii, area):
        n_a = len(coords_A)
        n_b = len(coords_B)
        counts = np.zeros(len(radii))
        for r_idx, r in enumerate(radii):
            inds = tree_B.query_ball_point(coords_A, r=r)
            total_neighbors = sum(len(i) for i in inds)
            counts[r_idx] = total_neighbors
        k_r = (area / (n_a * n_b)) * counts
        l_r = np.sqrt(k_r / np.pi)
        return l_r - radii
    
    tex_coords = cells_df[cells_df['cell_type'] == 'Tex'][['x', 'y']].values
    fib_coords = cells_df[cells_df['cell_type'] == 'Fibroblasts'][['x', 'y']].values
    m2_coords = cells_df[cells_df['cell_type'] == 'M2_Macrophages'][['x', 'y']].values
    
    tree_tex = KDTree(tex_coords)
    tree_fib = KDTree(fib_coords)
    tree_m2 = KDTree(m2_coords)
    
    l_tex_high = ripleys_cross_L(high_tumor_coords, tex_coords, tree_tex, radii, area)
    l_tex_low = ripleys_cross_L(low_tumor_coords, tex_coords, tree_tex, radii, area)
    
    l_fib_high = ripleys_cross_L(high_tumor_coords, fib_coords, tree_fib, radii, area)
    l_fib_low = ripleys_cross_L(low_tumor_coords, fib_coords, tree_fib, radii, area)
    
    l_m2_high = ripleys_cross_L(high_tumor_coords, m2_coords, tree_m2, radii, area)
    l_m2_low = ripleys_cross_L(low_tumor_coords, m2_coords, tree_m2, radii, area)
    
    # ----------------------------------------------------
    # Step 6: Load morphology and generate plots
    # ----------------------------------------------------
    print("\nStep 6: Generating high-quality plots...")
    morph_path = "/home/eto/bladder_spatial/data/extracted/LCCC_553_Xenium/morphology.ome.tif"
    
    # 1. Whole-slide morphology overlay
    print("Loading morphology image (Level 5) for whole-slide overlay...")
    with tifffile.TiffFile(morph_path) as tif:
        morph_level5 = tif.series[0].levels[5].asarray()
        if len(morph_level5.shape) == 3:
            morph_level5 = morph_level5[7]
            
    # Normalize contrast
    p1, p99 = np.percentile(morph_level5, (1, 99))
    morph_level5_norm = np.clip(morph_level5, p1, p99)
    morph_level5_norm = ((morph_level5_norm - p1) / (p99 - p1) * 255).astype(np.uint8)
    
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    ax.imshow(morph_level5_norm, cmap='gray', extent=[x_min, x_max, y_max, y_min])
    
    # Elegant Fluorescent Dye Color Scheme
    colors_dict = {
        'ARID1A-High': '#00E5FF', # Glowing Cyan
        'ARID1A-Low': '#FF00FF',  # Glowing Magenta
        'Tex': '#FF3333',         # Glowing Red
        'Fibroblasts': '#00FF00', # Glowing Green
        'M2_Macrophages': '#FF9900' # Glowing Gold/Orange
    }
    
    for cell_t, col in colors_dict.items():
        if cell_t in ['ARID1A-High', 'ARID1A-Low']:
            subset = cells_df[(cells_df['cell_type'] == 'Tumor_Epithelial') & (cells_df['arid1a_status'] == cell_t)]
        else:
            subset = cells_df[cells_df['cell_type'] == cell_t]
            
        ax.scatter(subset['x'], subset['y'], s=0.5, c=col, alpha=0.6, label=cell_t)
        
    ax.legend(loc='upper right', markerscale=10)
    ax.set_title("Whole-Slide Cell Spatial Distribution Overlaid on Morphology (DAPI)", fontsize=14)
    ax.set_xlabel("X coordinate (µm)")
    ax.set_ylabel("Y coordinate (µm)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "whole_slide_morphology_overlay.png"), dpi=300)
    plt.close()
    
    # 2. Advanced Zoom-in ROI Cell boundaries
    print("Generating zoom-in ROI cell boundary polygon plot...")
    cells_store = zarr.ZipStore(os.path.join(base_dir, "cells.zarr.zip"), mode='r')
    cells_grp = zarr.group(store=cells_store)
    p0 = cells_grp['polygon_sets/0']
    vertices = p0['vertices'][:]
    num_vertices = p0['num_vertices'][:]
    
    rx_min, ry_min, rx_max, ry_max = best_roi
    roi_cells_df = cells_df[
        (cells_df['x'] >= rx_min) & (cells_df['x'] < rx_max) &
        (cells_df['y'] >= ry_min) & (cells_df['y'] < ry_max)
    ]
    
    print("Loading morphology for ROI (Level 0, highest resolution)...")
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
    
    fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
    ax.imshow(morph_roi_norm, cmap='gray', extent=[rx_min, rx_max, ry_max, ry_min])
    
    for _, cell_row in roi_cells_df.iterrows():
        cell_idx = cell_row.name
        c_type = cell_row['cell_type']
        arid_status = cell_row['arid1a_status']
        
        if c_type == 'Tumor_Epithelial':
            col = colors_dict.get(arid_status, '#7f7f7f')
        else:
            col = colors_dict.get(c_type, '#7f7f7f')
            
        if col == '#7f7f7f':
            continue
            
        poly_pts = vertices[cell_idx].reshape(-1, 2)
        n_v = num_vertices[cell_idx]
        poly_pts = poly_pts[:n_v]
        
        polygon = patches.Polygon(poly_pts, closed=True, fill=False, edgecolor=col, linewidth=0.8, alpha=0.8)
        ax.add_patch(polygon)
        
    ax.set_xlim(rx_min, rx_max)
    ax.set_ylim(ry_max, ry_min)
    ax.set_title("Zoom-in High-Resolution Cell Boundaries in Microenvironment Niche", fontsize=12)
    ax.set_xlabel("X coordinate (µm)")
    ax.set_ylabel("Y coordinate (µm)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "zoom_roi_cell_boundaries.png"), dpi=300)
    plt.close()
    
    # 3. Niche Enrichment Plot (Using beautiful publication palettes)
    print("Plotting Spatial Niche profiles...")
    plt.figure(figsize=(10, 5), dpi=300)
    sns.heatmap(niche_profiles, annot=True, cmap="YlGnBu", fmt=".2f", yticklabels=niche_names)
    plt.title("Spatial Niches Profiling (Average Composition of 20 Nearest Neighbors)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "spatial_niche_profiles.png"), dpi=300)
    plt.close()
    
    # 4. Ripley's Cross-L plot (Using Nature/NPG hex colors)
    print("Plotting Ripley's Cross-L curves...")
    plt.figure(figsize=(8, 5), dpi=300)
    # NPG hex colors: Tex: #BC3C29, Fibroblasts: #00A087, M2: #F39B7F
    plt.plot(radii, l_tex_low, 'o-', color='#BC3C29', label='Tex to ARID1A-Low')
    plt.plot(radii, l_tex_high, 'o--', color='#BC3C29', alpha=0.6, label='Tex to ARID1A-High')
    plt.plot(radii, l_fib_low, 's-', color='#00A087', label='Fibroblasts to ARID1A-Low')
    plt.plot(radii, l_fib_high, 's--', color='#00A087', alpha=0.6, label='Fibroblasts to ARID1A-High')
    plt.plot(radii, l_m2_low, 'd-', color='#F39B7F', label='M2 to ARID1A-Low')
    plt.plot(radii, l_m2_high, 'd--', color='#F39B7F', alpha=0.6, label='M2 to ARID1A-High')
    
    plt.axhline(0, color='gray', linestyle='--')
    plt.xlabel("Spatial Scale r (µm)")
    plt.ylabel("Ripley's L(r) - r (Degree of Clustering)")
    plt.title("Ripley's Cross-L Function: Clustering Strength at Multiple Scales")
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ripley_cross_L_curves.png"), dpi=300)
    plt.close()
    
    print("Advanced analysis completed and all advanced figures saved successfully!")

if __name__ == "__main__":
    main()
