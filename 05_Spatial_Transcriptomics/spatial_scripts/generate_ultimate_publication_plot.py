import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import matplotlib.colors as mcolors
import tifffile

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
    
    # 2. Setup grid for density calculation (5 um resolution)
    bin_size = 5.0 
    xbins = np.arange(x_min, x_max + bin_size, bin_size)
    ybins = np.arange(y_min, y_max + bin_size, bin_size)
    H, W = len(ybins) - 1, len(xbins) - 1
    
    # Calculate density maps with smoothing (sigma = 20.0 pixels = 100 um physical scale)
    sigma_pixels = 20.0
    density_maps = {}
    for name, (c_type, status, color) in targets.items():
        if status is not None:
            sub_cells = cells_df[(cells_df['cell_type'] == c_type) & (cells_df['arid1a_status'] == status)]
        else:
            sub_cells = cells_df[cells_df['cell_type'] == c_type]
            
        hist, _, _ = np.histogram2d(sub_cells['y'].values, sub_cells['x'].values, bins=[ybins, xbins])
        smoothed = gaussian_filter(hist, sigma=sigma_pixels)
        if smoothed.max() > 0:
            norm_smoothed = smoothed / smoothed.max()
        else:
            norm_smoothed = smoothed
        density_maps[name] = norm_smoothed
        
    # 3. Load Level 5 morphology (whole-slide) to use as a faint, elegant background tissue structural map
    print("Loading morphology Level 5 background...")
    morph_path = os.path.join(base_dir, "morphology.ome.tif")
    with tifffile.TiffFile(morph_path) as tif:
        morph_level5 = tif.series[0].levels[5].asarray()
        if len(morph_level5.shape) == 3:
            morph_level5 = morph_level5[7]
            
    # Normalize and convert morphology to a dark navy-blue background color
    p1_5, p99_5 = np.percentile(morph_level5, (1, 99))
    morph_l5_norm = np.clip(morph_level5, p1_5, p99_5)
    morph_l5_norm = (morph_l5_norm - p1_5) / (p99_5 - p1_5)
    
    # Create dark navy-blue RGB image representing the tissue structure
    dapi_color = np.array([0.05, 0.08, 0.18]) # Faint navy blue
    dapi_bg_rgb = morph_l5_norm[:, :, np.newaxis] * dapi_color[np.newaxis, np.newaxis, :]
    
    # 4. Generate Neon Glow underlay by blending densities at a low opacity (alpha = 0.22)
    # This acts as a smooth gradient background under the sharp contour lines!
    print("Generating neon glow layer...")
    glow_rgb = np.zeros((H, W, 3))
    for name, (c_type, status, color_hex) in targets.items():
        rgb = np.array(mcolors.hex2color(color_hex))
        d_map = density_maps[name]
        glow_rgb += d_map[:, :, np.newaxis] * rgb[np.newaxis, np.newaxis, :]
        
    # Normalize glow layer to prevent clipping
    max_vals = glow_rgb.max(axis=-1, keepdims=True)
    glow_rgb = np.where(max_vals > 1.0, glow_rgb / max_vals, glow_rgb)
    
    # ------------------ Plot: The Ultimate Masterpiece ------------------
    print("Plotting Ultimate Masterpiece...")
    fig, ax = plt.subplots(figsize=(12, 14), dpi=300, facecolor='#060709')
    ax.set_facecolor('#060709')
    
    # Layer 1: Plot the DAPI tissue structural background (faint blue-gray)
    # This physically anchors the cells to the real tissue morphology!
    # Extent is calculated based on Level 5 resolution (Level 5 has shape 1382x1154, let's map to physical bounds)
    # The physical size matches the tissue size: Width=7851.45, Height=9402.49
    ax.imshow(dapi_bg_rgb, extent=[0, 7851.45, 9402.49, 0], origin='upper', alpha=0.6, interpolation='bilinear', rasterized=True)
    
    # Layer 2: Plot the neon-glow spatial density field at a low opacity (alpha = 0.25)
    # This provides a smooth, colorful gradient landscape under the lines.
    ax.imshow(glow_rgb, extent=[x_min, x_max, y_max, y_min], origin='upper', alpha=0.25, interpolation='bilinear', rasterized=True)
    
    # Layer 3: Plot the target cells as tiny, high-contrast, anti-aliased scatter dots (s=0.18, alpha=0.55)
    for name, (c_type, status, color) in targets.items():
        if status is not None:
            sub_cells = cells_df[(cells_df['cell_type'] == c_type) & (cells_df['arid1a_status'] == status)]
        else:
            sub_cells = cells_df[cells_df['cell_type'] == c_type]
            
        ax.scatter(sub_cells['x'], sub_cells['y'], c=color, s=0.18, alpha=0.55, rasterized=True)
        
    # Layer 4: Overlay the sharp, glowing contour lines (linewidth = 1.0, alpha = 0.85)
    # We use meshgrid based on the 5um grid
    x_centers = (xbins[:-1] + xbins[1:]) / 2
    y_centers = (ybins[:-1] + ybins[1:]) / 2
    X, Y = np.meshgrid(x_centers, y_centers)
    
    for name, (c_type, status, color) in targets.items():
        d_map = density_maps[name]
        if d_map.max() > 0:
            # Generate 4 contour levels
            levels = np.linspace(d_map.max() * 0.20, d_map.max() * 0.85, 4)
            # Plot sharp, glowing outline lines on top
            ax.contour(X, Y, d_map, levels=levels, colors=[color], linewidths=0.9, alpha=0.85)
            
    # Setup premium legend
    from matplotlib.lines import Line2D
    legend_elements = []
    for name, (_, _, color) in targets.items():
        legend_elements.append(Line2D([0], [0], marker='o', color='none',
                                      markerfacecolor=color, markersize=7,
                                      markeredgecolor='none', label=f"{name} Cells"))
        legend_elements.append(Line2D([0], [0], color=color, linewidth=1.5,
                                      linestyle='-', label=f"{name} Density Gradient"))
        
    ax.legend(handles=legend_elements, facecolor='#0C0E14', edgecolor='#222735',
              loc='upper right', fontsize=10, labelcolor='#FFFFFF', ncol=1)
    
    ax.set_title("Whole-Slide Cellular Microenvironment & Spatial Gradients", color='#FFFFFF', fontsize=18, pad=20, fontweight='bold')
    ax.set_xlabel("X Coordinate (µm)", color='#888888', fontsize=12)
    ax.set_ylabel("Y Coordinate (µm)", color='#888888', fontsize=12)
    ax.tick_params(colors='#888888', labelsize=10)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_max, y_min)
    ax.set_aspect('equal', 'box')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "whole_slide_density_contour_scatter_overlay.png"), bbox_inches='tight', dpi=300)
    plt.savefig(os.path.join(output_dir, "whole_slide_density_contour_scatter_overlay.pdf"), bbox_inches='tight')
    plt.close()
    print("Masterpiece plots generated successfully!")

if __name__ == "__main__":
    main()
