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
    
    x_min, x_max = cells_df['x'].min() - 50, cells_df['x'].max() + 50
    y_min, y_max = cells_df['y'].min() - 50, cells_df['y'].max() + 50
    
    # ------------------ Style 1: Hexagonal honeycomb binning (Hexbin) ------------------
    # Hexbin is a classic top-journal visualization that shows density using sharp, structured hexagons!
    print("Generating Hexagonal Binning plot (Hexbin)...")
    fig, axes = plt.subplots(1, 5, figsize=(25, 6), dpi=300, facecolor='#0B0C10')
    
    for idx, (name, (c_type, status, color)) in enumerate(targets.items()):
        ax = axes[idx]
        ax.set_facecolor('#0B0C10')
        
        if status is not None:
            sub_cells = cells_df[(cells_df['cell_type'] == c_type) & (cells_df['arid1a_status'] == status)]
        else:
            sub_cells = cells_df[cells_df['cell_type'] == c_type]
            
        # Create a colormap from black to target color
        rgb = mcolors.hex2color(color)
        cmap = mcolors.LinearSegmentedColormap.from_list("hex", [(0.05, 0.05, 0.08), rgb], N=256)
        
        # Plot hexbin with gridsize=100 (determines hexagon size)
        # mincnt=1 hides empty hexagons
        hb = ax.hexbin(sub_cells['x'], sub_cells['y'], gridsize=80, cmap=cmap, mincnt=1, edgecolors='none')
        
        ax.set_title(f"{name} Hex-Density", color='#FFFFFF', fontsize=16, fontweight='bold', pad=10)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_max, y_min) # Flip Y to match coordinate system
        ax.axis('off')
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "whole_slide_hexbin_density.png"), bbox_inches='tight', dpi=300)
    plt.savefig(os.path.join(output_dir, "whole_slide_hexbin_density.pdf"), bbox_inches='tight') # Vector PDF!
    plt.close()
    
    # ------------------ Style 2: Sharp Contour Lines (等高线风格) ------------------
    # Instead of filled blurred colors, we use sharp concentric contour lines!
    print("Generating Sharp Contour Lines plot...")
    
    bin_size = 10.0
    xbins = np.arange(x_min, x_max + bin_size, bin_size)
    ybins = np.arange(y_min, y_max + bin_size, bin_size)
    
    fig, ax = plt.subplots(figsize=(12, 14), dpi=300, facecolor='#0B0C10')
    ax.set_facecolor('#0B0C10')
    
    # Plot contour lines for each cell type
    for name, (c_type, status, color) in targets.items():
        if status is not None:
            sub_cells = cells_df[(cells_df['cell_type'] == c_type) & (cells_df['arid1a_status'] == status)]
        else:
            sub_cells = cells_df[cells_df['cell_type'] == c_type]
            
        hist, xedges, yedges = np.histogram2d(sub_cells['x'].values, sub_cells['y'].values, bins=[xbins, ybins])
        smoothed = gaussian_filter(hist, sigma=3.0)
        
        # Center of bins
        x_centers = (xedges[:-1] + xedges[1:]) / 2
        y_centers = (yedges[:-1] + yedges[1:]) / 2
        X, Y = np.meshgrid(x_centers, y_centers)
        
        # Plot 4 levels of contours (outline lines only, no fill!)
        # This creates extremely sharp vector lines
        if smoothed.max() > 0:
            levels = np.linspace(smoothed.max() * 0.15, smoothed.max() * 0.9, 4)
            ax.contour(X, Y, smoothed.T, levels=levels, colors=[color], linewidths=1.2, alpha=0.85)
            
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='none', edgecolor=color, linewidth=1.5, label=name) for name, (_, _, color) in targets.items()]
    ax.legend(handles=legend_elements, facecolor='#121212', edgecolor='#333333', loc='upper right', fontsize=12, labelcolor='#FFFFFF')
    
    ax.set_title("Whole-Slide Spatial Contour Landscape (Sharp Vector Lines)", color='#FFFFFF', fontsize=18, pad=20, fontweight='bold')
    ax.set_xlabel("X Coordinate (µm)", color='#888888', fontsize=12)
    ax.set_ylabel("Y Coordinate (µm)", color='#888888', fontsize=12)
    ax.tick_params(colors='#888888', labelsize=10)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_max, y_min)
    ax.set_aspect('equal', 'box')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "whole_slide_contour_sharp.png"), bbox_inches='tight', dpi=300)
    plt.savefig(os.path.join(output_dir, "whole_slide_contour_sharp.pdf"), bbox_inches='tight') # Vector PDF!
    plt.close()
    
    print("Sharp vector contour and hexbin plots generated successfully!")

if __name__ == "__main__":
    main()
