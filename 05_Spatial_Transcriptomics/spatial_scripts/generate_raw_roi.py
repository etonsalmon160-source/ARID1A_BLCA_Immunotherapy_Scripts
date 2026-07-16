import os
import numpy as np
import tifffile
import zarr
import matplotlib.pyplot as plt

def main():
    base_dir = "/home/eto/bladder_spatial/data/extracted/LCCC_553_Xenium"
    output_dir = "/home/eto/bladder_spatial/output"
    
    # ROI: X in [5252.2, 5652.2], Y in [6599.0, 6999.0]
    rx_min, ry_min, rx_max, ry_max = 5252.2, 6599.0, 5652.2, 6999.0
    
    pixel_size_l0 = 0.2125
    py_min = int(ry_min / pixel_size_l0)
    py_max = int(ry_max / pixel_size_l0)
    px_min = int(rx_min / pixel_size_l0)
    px_max = int(rx_max / pixel_size_l0)
    
    print("Loading raw morphology for ROI (Level 0, DAPI channel)...")
    morph_path = os.path.join(base_dir, "morphology.ome.tif")
    with tifffile.TiffFile(morph_path) as tif:
        store = tif.series[0].levels[0].aszarr()
        z_grp = zarr.open(store, mode='r')
        z_arr = z_grp['0']
        
        # Read channel 7 (DAPI / morphology focus)
        raw_roi = z_arr[7, py_min:py_max, px_min:px_max]
        
    # Normalize contrast
    p1, p99 = np.percentile(raw_roi, (1, 99.5))
    raw_roi_norm = np.clip(raw_roi, p1, p99)
    raw_roi_norm = ((raw_roi_norm - p1) / (p99 - p1) * 255).astype(np.uint8)
    
    # Save image
    fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
    ax.imshow(raw_roi_norm, cmap='gray', extent=[rx_min, rx_max, ry_max, ry_min])
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "raw_morphology_roi.png"), bbox_inches='tight', dpi=300)
    plt.close()
    print("Raw morphology ROI image saved successfully!")

if __name__ == "__main__":
    main()
