import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import matplotlib.colors as mcolors

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
    
    # 2. Setup grid for contour calculation
    bin_size = 10.0
    xbins = np.arange(x_min, x_max + bin_size, bin_size)
    ybins = np.arange(y_min, y_max + bin_size, bin_size)
    
    # Pre-calculate smooth density maps for contour plotting
    # sigma = 15.0 bins = 150 um physical scale (standard microenvironmental niche size)
    sigma_pixels = 15.0
    density_grids = {}
    for name, (c_type, status, color) in targets.items():
        if status is not None:
            sub_cells = cells_df[(cells_df['cell_type'] == c_type) & (cells_df['arid1a_status'] == status)]
        else:
            sub_cells = cells_df[cells_df['cell_type'] == c_type]
            
        hist, xedges, yedges = np.histogram2d(sub_cells['x'].values, sub_cells['y'].values, bins=[xbins, ybins])
        smoothed = gaussian_filter(hist, sigma=sigma_pixels)
        density_grids[name] = (smoothed, xedges, yedges)
        
    # ------------------ Plot: Single-Cell Scatter with Contour Lines Overlay ------------------
    print("Generating Scatter + Contour Overlay...")
    fig, ax = plt.subplots(figsize=(12, 14), dpi=300, facecolor='#0B0C10')
    ax.set_facecolor('#0B0C10')
    
    # Step 1: Plot background tissue structure (Other cells in very dim gray)
    bg_cells = cells_df[cells_df['cell_type'].isin(['Other', 'T_cells'])]
    ax.scatter(bg_cells['x'], bg_cells['y'], c='#111115', s=0.03, alpha=0.3, rasterized=True)
    
    # Step 2: Plot target cells as tiny, sharp, solid dots
    for name, (c_type, status, color) in targets.items():
        if status is not None:
            sub_cells = cells_df[(cells_df['cell_type'] == c_type) & (cells_df['arid1a_status'] == status)]
        else:
            sub_cells = cells_df[cells_df['cell_type'] == c_type]
            
        # Draw cells as very small, sharp dots (s=0.2, alpha=0.5)
        # This provides the single-cell structural ground truth!
        ax.scatter(sub_cells['x'], sub_cells['y'], c=color, s=0.15, alpha=0.4, rasterized=True)
        
    # Step 3: Overlay the Probability Density Gradient as sharp, concentric contour lines
    for name, (c_type, status, color) in targets.items():
        smoothed, xedges, yedges = density_grids[name]
        
        # Centers of bins
        x_centers = (xedges[:-1] + xedges[1:]) / 2
        y_centers = (yedges[:-1] + yedges[1:]) / 2
        X, Y = np.meshgrid(x_centers, y_centers)
        
        if smoothed.max() > 0:
            # We select 4 distinct density thresholds (e.g. from 20% to 80% of peak density)
            levels = np.linspace(smoothed.max() * 0.20, smoothed.max() * 0.85, 4)
            
            # Plot contours (lines only) with high contrast and anti-aliasing
            # The contour lines represent the mathematically rigorous probability density gradient!
            ax.contour(X, Y, smoothed.T, levels=levels, colors=[color], linewidths=0.9, alpha=0.9)
            
    # Setup premium legend with both dots (cells) and lines (density gradients)
    from matplotlib.lines import Line2D
    legend_elements = []
    for name, (_, _, color) in targets.items():
        legend_elements.append(Line2D([0], [0], marker='o', color='none',
                                      markerfacecolor=color, markersize=6,
                                      markeredgecolor='none', label=f"{name} Cells"))
        legend_elements.append(Line2D([0], [0], color=color, linewidth=1.5,
                                      linestyle='-', label=f"{name} Density Gradient"))
        
    ax.legend(handles=legend_elements, facecolor='#121212', edgecolor='#333333',
              loc='upper right', fontsize=9, labelcolor='#FFFFFF', ncol=1)
    
    ax.set_title("Whole-Slide Cellular Topography & Density Gradients", color='#FFFFFF', fontsize=16, pad=20, fontweight='bold')
    ax.set_xlabel("X Coordinate (µm)", color='#888888', fontsize=12)
    ax.set_ylabel("Y Coordinate (µm)", color='#888888', fontsize=12)
    ax.tick_params(colors='#888888', labelsize=10)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_max, y_min) # Flip Y to match microscopy coordinates
    ax.set_aspect('equal', 'box')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "whole_slide_density_contour_scatter_overlay.png"), bbox_inches='tight', dpi=300)
    plt.savefig(os.path.join(output_dir, "whole_slide_density_contour_scatter_overlay.pdf"), bbox_inches='tight')
    plt.close()
    print("Scatter + Contour overlay plots generated successfully!")

if __name__ == "__main__":
    main()
