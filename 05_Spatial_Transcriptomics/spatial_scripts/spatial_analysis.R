library(Seurat)
library(ggplot2)
library(dplyr)
library(patchwork)

# Set working directory
output_dir <- "/home/eto/bladder_spatial/output"
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

# Find files
find_file <- function(base_dir, pattern) {
  files <- list.files(base_dir, pattern = pattern, recursive = TRUE, full.names = TRUE)
  if (length(files) == 0) {
    stop(paste("Error: File with pattern", pattern, "not found in", base_dir))
  }
  return(files[1])
}

cells_file <- "/home/eto/bladder_spatial/data/extracted/LCCC_553_Xenium/cells_meta.csv"
mex_dir <- "/home/eto/bladder_spatial/data/extracted/LCCC_553_Xenium/mex"

cat("Loading counts matrix from MEX...\n")
counts <- Read10X(mex_dir)

cat("Creating Seurat object...\n")
seurat_obj <- CreateSeuratObject(counts = counts, project = "Xenium_BLCA")

cat("Loading spatial coordinates...\n")
cells_meta <- read.csv(cells_file)
rownames(cells_meta) <- cells_meta$cell_id

# Align and add metadata
cells_meta <- cells_meta[colnames(seurat_obj), ]
seurat_obj <- AddMetaData(seurat_obj, metadata = cells_meta)

cat("Normalizing data...\n")
seurat_obj <- NormalizeData(seurat_obj)

cat("Defining marker signatures...\n")
signatures <- list(
  T_cells = c("CD3E", "CD8A"),
  Tex = c("PDCD1", "CXCL13", "HAVCR2"),
  Fibroblasts = c("ACTA2", "FAP", "POSTN"),
  M2_Macrophages = c("CD68", "CD163", "MRC1", "APOE"),
  Tumor_Epithelial = c("EPCAM", "KRT17")
)

cat("Calculating signature scores...\n")
for (name in names(signatures)) {
  genes <- signatures[[name]]
  genes_exist <- genes[genes %in% rownames(seurat_obj)]
  if (length(genes_exist) > 0) {
    if (length(genes_exist) == 1) {
      seurat_obj[[name]] <- seurat_obj@assays$RNA@data[genes_exist, ]
    } else {
      seurat_obj[[name]] <- colMeans(as.matrix(seurat_obj@assays$RNA@data[genes_exist, ]))
    }
  } else {
    seurat_obj[[name]] <- 0
  }
}

cat("Assigning cell types based on signatures...\n")
sig_matrix <- data.frame(
  T_cells = seurat_obj$T_cells,
  Tex = seurat_obj$Tex,
  Fibroblasts = seurat_obj$Fibroblasts,
  M2_Macrophages = seurat_obj$M2_Macrophages,
  Tumor_Epithelial = seurat_obj$Tumor_Epithelial
)

max_sig <- apply(sig_matrix, 1, max)
max_name <- colnames(sig_matrix)[apply(sig_matrix, 1, which.max)]

# Assign type
cell_type <- ifelse(max_sig > 0.05, max_name, "Other")
seurat_obj$cell_type <- cell_type

# Print cell type counts
print(table(seurat_obj$cell_type))

cat("Defining ARID1A status in tumor cells...\n")
tumor_indices <- which(seurat_obj$cell_type == "Tumor_Epithelial")
if (length(tumor_indices) > 0) {
  arid1a_expr <- seurat_obj@assays$RNA@data["ARID1A", tumor_indices]
  q_low <- quantile(arid1a_expr, 0.3)
  q_high <- quantile(arid1a_expr, 0.7)
  
  arid1a_status <- rep("Intermediate", ncol(seurat_obj))
  arid1a_status[tumor_indices[arid1a_expr <= q_low]] <- "ARID1A-Low"
  arid1a_status[tumor_indices[arid1a_expr >= q_high]] <- "ARID1A-High"
  
  # For non-tumor cells, mark as Microenvironment
  arid1a_status[-tumor_indices] <- "Microenvironment"
  seurat_obj$arid1a_status <- arid1a_status
} else {
  stop("Error: No tumor cells found!")
}

print(table(seurat_obj$arid1a_status))

cat("Preparing spatial coordinates...\n")
coords <- data.frame(
  x = seurat_obj$x_centroid,
  y = seurat_obj$y_centroid,
  cell_type = seurat_obj$cell_type,
  arid1a_status = seurat_obj$arid1a_status
)
rownames(coords) <- colnames(seurat_obj)

# Ensure nabor is installed for fast spatial distance calculation
if (!requireNamespace("nabor", quietly = TRUE)) {
  cat("Installing nabor package...\n")
  install.packages("nabor", repos = "https://mirrors.tuna.tsinghua.edu.cn/CRAN/")
}
library(nabor)

cat("Calculating distances to ARID1A-high and ARID1A-low tumor cells...\n")
tumor_low_coords <- coords[coords$arid1a_status == "ARID1A-Low", c("x", "y")]
tumor_high_coords <- coords[coords$arid1a_status == "ARID1A-High", c("x", "y")]

# Define microenvironment cells we want to analyze
me_types <- c("Tex", "Fibroblasts", "M2_Macrophages")

distance_list <- list()
for (me_type in me_types) {
  me_coords <- coords[coords$cell_type == me_type, c("x", "y")]
  if (nrow(me_coords) > 0 && nrow(tumor_low_coords) > 0 && nrow(tumor_high_coords) > 0) {
    # Nearest distance to ARID1A-Low
    nn_low <- knn(data = as.matrix(tumor_low_coords), query = as.matrix(me_coords), k = 1)
    dist_low <- nn_low$nn.dists[, 1]
    
    # Nearest distance to ARID1A-High
    nn_high <- knn(data = as.matrix(tumor_high_coords), query = as.matrix(me_coords), k = 1)
    dist_high <- nn_high$nn.dists[, 1]
    
    df <- data.frame(
      cell_type = me_type,
      dist_to_low = dist_low,
      dist_to_high = dist_high
    )
    distance_list[[me_type]] <- df
  }
}

distance_df <- do.call(rbind, distance_list)

# Save distance data as CSV
write.csv(coords, "/home/eto/bladder_spatial/output/all_cells_meta.csv", row.names = FALSE)
write.csv(distance_df, "/home/eto/bladder_spatial/output/nearest_distances.csv", row.names = FALSE)
cat("Saved nearest distances to CSV.\n")

cat("Plotting spatial distribution...\n")
plot_coords <- coords %>%
  filter(cell_type != "Other")

p_spatial <- ggplot(plot_coords, aes(x = x, y = y, color = cell_type)) +
  geom_point(size = 0.1, alpha = 0.6) +
  scale_color_manual(values = c(
    Tumor_Epithelial = "#3C5488", 
    Tex = "#BC3C29", 
    T_cells = "#8491B4", 
    Fibroblasts = "#00A087", 
    M2_Macrophages = "#F39B7F"
  )) +
  labs(title = "Spatial Distribution of Tumor and Microenvironment Cells",
       x = "X Coordinate (µm)", y = "Y Coordinate (µm)", color = "Cell Type") +
  theme_classic() +
  theme(legend.position = "right")

ggsave(file.path(output_dir, "spatial_distribution_map.png"), plot = p_spatial, width = 10, height = 8, dpi = 300)

cat("Plotting distance comparison (Violin plot)...\n")
# Reshape data for plotting
plot_dist_df <- data.frame(
  cell_type = rep(distance_df$cell_type, 2),
  distance = c(distance_df$dist_to_low, distance_df$dist_to_high),
  target = c(rep("Nearest ARID1A-Low", nrow(distance_df)), rep("Nearest ARID1A-High", nrow(distance_df)))
)

p_dist <- ggplot(plot_dist_df, aes(x = cell_type, y = distance, fill = target)) +
  geom_violin(position = position_dodge(0.8), alpha = 0.7) +
  geom_boxplot(position = position_dodge(0.8), width = 0.1, outlier.shape = NA, color = "black") +
  labs(title = "Distance to Nearest ARID1A-High vs ARID1A-Low Tumor Cells",
       x = "Microenvironment Cell Type", y = "Distance (µm)", fill = "Tumor Target") +
  scale_fill_manual(values = c("Nearest ARID1A-Low" = "#E64B35", "Nearest ARID1A-High" = "#3C5488")) +
  theme_classic() +
  ylim(0, 300)

ggsave(file.path(output_dir, "arid1a_vs_tme_distance_violin.png"), plot = p_dist, width = 8, height = 6, dpi = 300)

cat("Performing spatial co-localization analysis (Correlation of local densities)...\n")
# Divide spatial coordinates into grid bins and count cell densities
grid_size <- 50 # µm
coords$grid_x <- floor(coords$x / grid_size)
coords$grid_y <- floor(coords$y / grid_size)
coords$grid_id <- paste(coords$grid_x, coords$grid_y, sep = "_")

# Create density matrix
grid_counts <- coords %>%
  group_by(grid_id, cell_type) %>%
  tally() %>%
  tidyr::spread(cell_type, n, fill = 0)

# Calculate mean ARID1A per grid
arid1a_grid <- coords %>%
  filter(cell_type == "Tumor_Epithelial") %>%
  mutate(arid1a_val = seurat_obj@assays$RNA@data["ARID1A", rownames(.)]) %>%
  group_by(grid_id) %>%
  summarize(mean_arid1a = mean(arid1a_val, na.rm = TRUE))

grid_data <- merge(grid_counts, arid1a_grid, by = "grid_id", all.x = TRUE)
grid_data[is.na(grid_data)] <- 0

# Calculate correlation matrix
corr_features <- c("mean_arid1a", "Tex", "Fibroblasts", "M2_Macrophages", "Tumor_Epithelial")
corr_features <- corr_features[corr_features %in% colnames(grid_data)]
corr_mat <- cor(grid_data[, corr_features], method = "spearman")

# Plot correlation heatmap
if (!requireNamespace("reshape2", quietly = TRUE)) {
  install.packages("reshape2", repos = "https://mirrors.tuna.tsinghua.edu.cn/CRAN/")
}
library(reshape2)

melted_corr <- melt(corr_mat)
p_corr <- ggplot(melted_corr, aes(x = Var1, y = Var2, fill = value)) +
  geom_tile(color = "white") +
  scale_fill_gradient2(low = "#3C5488", high = "#E64B35", mid = "white", 
                       midpoint = 0, limit = c(-1,1), space = "Lab", 
                       name="Spearman\nCorr") +
  theme_minimal() + 
  theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1)) +
  coord_fixed() +
  labs(title = "Spatial Co-localization Correlation (50µm Grid)", x = "", y = "")

ggsave(file.path(output_dir, "spatial_correlation_heatmap.png"), plot = p_corr, width = 6, height = 5, dpi = 300)

cat("All analyses completed successfully! Plots saved in:", output_dir, "\n")
