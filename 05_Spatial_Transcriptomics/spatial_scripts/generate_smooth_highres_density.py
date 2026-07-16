import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import matplotlib.colors as mcolors

def create_custom_cmap(color_hex):
    rgb = mcolors.hex2color(color_hex)
    return mcolors.LinearSegmentedColormap.from_list(
        f"custom_{color_hex}",
        [(0, 0, 0), rgb],
        N=256
    )

def main():
    base_dir = "/home/eto/bladder_spatial/data/extracted/LCCC_553_Xenium"
    output_dir = "/home/eto/bladder_spatial/output"
    
    # 1. Load cell metadata
    print("Loading cell metadata...")
    cells_df = pd.read_csv(os.path.join(output_dir, "all_cells_meta.csv"))
    
    targets = {
        'ARID1A-High': ('Tumor_Epithelial', 'ARID1A-High', '#00E5FF'),      # Cyan
        'ARID1A-Low': ('Tumor_Epithelial', 'ARID1A-Low', '#FF00FF'),        # Magenta
        'Tex': ('Tex', None, '#FF3333'),                                    # Red
        'Fibroblasts': ('Fibroblasts', None, '#00FF00'),                    # Green
        'M2_Macrophages': ('M2_Macrophages', None, '#FF9900')               # Orange/Gold
    }
    
    x_min, x_max = cells_df['x'].min() - 100, cells_df['x'].max() + 100
    y_min, y_max = cells_df['y'].min() - 100, cells_df['y'].max() + 100
    
    # Grid size: 5 um (high spatial resolution)
    bin_size = 5.0 
    xbins = np.arange(x_min, x_max + bin_size, bin_size)
    ybins = np.arange(y_min, y_max + bin_size, bin_size)
    
    H, W = len(ybins) - 1, len(xbins) - 1
    print(f"High-Res Grid size: {H}x{W} bins")
    
    # 2. Calculate 2D histograms and apply smooth continuous Gaussian filter
    # sigma = 25.0 bins on 5um grid = 125 um physical scale
    # This represents a true local probability density estimation, merging points into a smooth field.
    sigma_pixels = 25.0 
    
    density_maps = {}
    for name, (c_type, status, color) in targets.items():
        if status is not None:
            sub_cells = cells_df[(cells_df['cell_type'] == c_type) & (cells_df['arid1a_status'] == status)]
        else:
            sub_cells = cells_df[cells_df['cell_type'] == c_type]
            
        hist, _, _ = np.histogram2d(sub_cells['y'].values, sub_cells['x'].values, bins=[ybins, xbins])
        
        # Apply Gaussian blur for continuous probability density
        smoothed = gaussian_filter(hist, sigma=sigma_pixels)
        
        # We apply a gamma correction (power law) to enhance the visibility of lower density gradients
        # This is standard in microscopy to make smooth halos stand out beautifully!
        if smoothed.max() > 0:
            norm_smoothed = smoothed / smoothed.max()
            norm_smoothed = np.power(norm_smoothed, 0.7) # Enhance low density gradients
        else:
            norm_smoothed = smoothed
            
        density_maps[name] = norm_smoothed
        print(f"Computed smooth density map for {name}")
        
    # 3. Generate high-resolution composite plot
    print("Generating color-blended composite overlay...")
    composite_rgb = np.zeros((H, W, 3))
    
    for name, (c_type, status, color_hex) in targets.items():
        rgb = np.array(mcolors.hex2color(color_hex))
        d_map = density_maps[name]
        composite_rgb += d_map[:, :, np.newaxis] * rgb[np.newaxis, np.newaxis, :]
        
    composite_rgb = np.clip(composite_rgb, 0, 1)
    
    # Save the composite plot
    fig, ax = plt.subplots(figsize=(12, 14), dpi=300, facecolor='#0B0C10')
    ax.set_facecolor('#0B0C10')
    
    # Use bilinear interpolation for smooth gradients
    ax.imshow(composite_rgb, extent=[x_min, x_max, y_max, y_min], origin='upper', interpolation='bilinear')
    
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color, edgecolor='none', label=name) for name, (_, _, color) in targets.items()]
    ax.legend(handles=legend_elements, facecolor='#121212', edgecolor='#333333', loc='upper right', fontsize=12, labelcolor='#FFFFFF')
    
    ax.set_title("Whole-Slide Continuous Spatial Density Composite", color='#FFFFFF', fontsize=18, pad=20, fontweight='bold')
    ax.set_xlabel("X Coordinate (µm)", color='#888888', fontsize=12)
    ax.set_ylabel("Y Coordinate (µm)", color='#888888', fontsize=12)
    ax.tick_params(colors='#888888', labelsize=10)
    ax.set_aspect('equal', 'box')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "whole_slide_density_composite.png"), bbox_inches='tight', dpi=300)
    plt.savefig(os.path.join(output_dir, "whole_slide_density_composite.pdf"), bbox_inches='tight') # Vector PDF for infinite zoom!
    
    # Also update the 5-panel layout
    fig, axes = plt.subplots(1, 5, figsize=(25, 6), dpi=300, facecolor='#0B0C10')
    for idx, (name, (c_type, status, color)) in enumerate(targets.items()):
        ax = axes[idx]
        ax.set_facecolor('#0B0C10')
        cmap = create_custom_cmap(color)
        ax.imshow(density_maps[name], extent=[x_min, x_max, y_max, y_min], cmap=cmap, origin='upper', interpolation='bilinear')
        ax.set_title(f"{name} Density (Smooth)", color='#FFFFFF', fontsize=16, fontweight='bold', pad=10)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "whole_slide_smooth_density_panel.png"), bbox_inches='tight', dpi=300)
    plt.close()
    
    print("Smooth density plots generated successfully!")

if __name__ == "__main__":
    main()
