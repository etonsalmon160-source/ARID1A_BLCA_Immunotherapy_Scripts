# analyze_bladder_bulk_mutation_tme_indices_high_tmb.R
library(httr)
library(jsonlite)
library(ggplot2)
library(grid)

study_id <- "blca_tcga_pan_can_atlas_2018"
target_gene <- 8289 # ARID1A

# Define gene lists with Entrez IDs
t_cell_ex_ids <- c(5133, 29126, 1493, 84868, 3902, 201633, 30048) # PDCD1, CD274, CTLA4, HAVCR2, LAG3, TIGIT, TOX
m2_mac_ids <- c(968, 9332, 4360, 1436, 3586)                      # CD68, CD163, MRC1, CSF1R, IL10
fib_caf_ids <- c(59, 2191, 1277, 1278, 5159, 7040)                 # ACTA2, FAP, COL1A1, COL1A2, PDGFRB, TGFB1

all_entrez_ids <- unique(c(target_gene, t_cell_ex_ids, m2_mac_ids, fib_caf_ids))

message("1. Fetching TCGA BLCA clinical data...")
url_samples <- paste0("https://www.cbioportal.org/api/studies/", study_id, "/clinical-data?projection=DETAILED&clinicalDataType=SAMPLE")
response_samples <- GET(url_samples, accept_json())
if (status_code(response_samples) != 200) stop("Failed to fetch clinical data.")
sample_data <- fromJSON(content(response_samples, as = "text", encoding = "UTF-8"))

# Reshape clinical data
message("Reshaping clinical data...")
sample_ids <- unique(sample_data$sampleId)
df_samples <- data.frame(SAMPLE_ID = sample_ids, stringsAsFactors = FALSE)

attr_data <- sample_data[sample_data$clinicalAttributeId == "MUTATION_COUNT", ]
merged <- data.frame(sampleId = attr_data$sampleId, MUTATION_COUNT = as.numeric(attr_data$value), stringsAsFactors = FALSE)
df_samples <- merge(df_samples, merged, by.x = "SAMPLE_ID", by.y = "sampleId", all.x = TRUE)

df_samples <- df_samples[!is.na(df_samples$MUTATION_COUNT), ]

# Determine High TMB threshold (top 33%)
tmb_threshold <- quantile(df_samples$MUTATION_COUNT, 0.67, na.rm = TRUE)
df_high_tmb <- df_samples[df_samples$MUTATION_COUNT >= tmb_threshold, ]
high_tmb_sample_ids <- df_high_tmb$SAMPLE_ID
message(sprintf("High TMB cohort size (>= %.1f mutations): %d samples", tmb_threshold, length(high_tmb_sample_ids)))

# Fetch ARID1A mutation status for High TMB cohort
message("2. Fetching somatic mutations for High TMB cohort...")
url_mut <- paste0("https://www.cbioportal.org/api/molecular-profiles/", study_id, "_mutations/mutations/fetch")
fetch_body_mut <- list(
  entrezGeneIds = I(8289),
  sampleIds = high_tmb_sample_ids
)
response_mut <- POST(
  url_mut,
  body = toJSON(fetch_body_mut, auto_unbox = TRUE),
  content_type_json(),
  accept_json()
)
if (status_code(response_mut) != 200) stop("Failed to fetch mutations.")
mut_data <- fromJSON(content(response_mut, as = "text", encoding = "UTF-8"))

mutated_samples <- unique(mut_data$sampleId)
df_high_tmb$ARID1A_Status <- factor(
  ifelse(df_high_tmb$SAMPLE_ID %in% mutated_samples, "Mutant", "Wild-Type"),
  levels = c("Wild-Type", "Mutant")
)

message("3. Fetching RNA Z-scores for High TMB cohort...")
url_rna <- paste0("https://www.cbioportal.org/api/molecular-profiles/", study_id, "_rna_seq_v2_mrna_median_Zscores/molecular-data/fetch")
fetch_body_rna <- list(
  entrezGeneIds = all_entrez_ids,
  sampleIds = high_tmb_sample_ids
)
response_rna <- POST(
  url_rna,
  body = toJSON(fetch_body_rna, auto_unbox = TRUE),
  content_type_json(),
  accept_json()
)
if (status_code(response_rna) != 200) stop("Failed to fetch RNA expression data.")
rna_data <- fromJSON(content(response_rna, as = "text", encoding = "UTF-8"))

# Build RNA expression matrix
rna_wide <- reshape(
  rna_data[, c("sampleId", "entrezGeneId", "value")],
  idvar = "sampleId",
  timevar = "entrezGeneId",
  direction = "wide"
)
colnames(rna_wide) <- gsub("value\\.", "Gene_", colnames(rna_wide))
rownames(rna_wide) <- rna_wide$sampleId

# Function to calculate index mean
calc_index <- function(gene_ids) {
  cols <- paste0("Gene_", gene_ids)
  present_cols <- intersect(cols, colnames(rna_wide))
  if (length(present_cols) == 0) return(rep(NA, nrow(rna_wide)))
  if (length(present_cols) == 1) return(rna_wide[[present_cols]])
  rowMeans(rna_wide[, present_cols, drop = FALSE], na.rm = TRUE)
}

rna_wide$T_Cell_Exhaustion <- calc_index(t_cell_ex_ids)
rna_wide$M2_Macrophage <- calc_index(m2_mac_ids)
rna_wide$Fibroblast_CAF <- calc_index(fib_caf_ids)

# Merge mutation status and RNA indices
df_merged <- merge(rna_wide, df_high_tmb[, c("SAMPLE_ID", "ARID1A_Status")], by.x = "sampleId", by.y = "SAMPLE_ID")

# Run Wilcoxon statistics
message("\n=== Comparisons in R (High TMB Cohort - ARID1A Somatic Mutation) ===")
sig_names <- c("T_Cell_Exhaustion", "M2_Macrophage", "Fibroblast_CAF")
titles <- c("T-Cell Exhaustion Index\n(PD-1, CTLA-4, TOX, etc.)", 
            "Immunosuppressive M2 TAM Index\n(CD163, MRC1, CSF1R, etc.)", 
            "Cancer-Associated Fibroblast Index\n(a-SMA, FAP, COL1A1, etc.)")

plots <- list()

for (i in 1:3) {
  sig <- sig_names[i]
  title_text <- titles[i]
  
  sub_df <- na.omit(df_merged[, c("ARID1A_Status", sig)])
  wt_vals <- sub_df[sub_df$ARID1A_Status == "Wild-Type", sig]
  mut_vals <- sub_df[sub_df$ARID1A_Status == "Mutant", sig]
  
  pval <- wilcox.test(wt_vals, mut_vals, alternative = "two.sided")$p.value
  
  cat(sprintf("\nMetric: %s\n", sig))
  cat(sprintf("Wild-Type (N=%d): Mean=%.4f, Median=%.4f\n", length(wt_vals), mean(wt_vals), median(wt_vals)))
  cat(sprintf("Mutant (N=%d): Mean=%.4f, Median=%.4f\n", length(mut_vals), mean(mut_vals), median(mut_vals)))
  cat(sprintf("Wilcoxon rank-sum test p-value: %.6e\n", pval))
  
  sig_label <- "ns"
  if (pval < 0.05) sig_label <- "*"
  if (pval < 0.01) sig_label <- "**"
  if (pval < 0.001) sig_label <- "***"
  if (pval < 0.0001) sig_label <- "****"
  
  y_max <- max(sub_df[[sig]])
  y_min <- min(sub_df[[sig]])
  h <- (y_max - y_min) * 0.05
  
  # Format custom x-tick labels with sample size
  x_labs <- c(sprintf("Wild-Type\n(N=%d)", length(wt_vals)), sprintf("Mutant\n(N=%d)", length(mut_vals)))
  
  p <- ggplot(sub_df, aes(x = ARID1A_Status, y = .data[[sig]], fill = ARID1A_Status)) +
    geom_boxplot(width = 0.45, outlier.shape = NA, linewidth = 0.8, show.legend = FALSE) +
    geom_jitter(width = 0.2, alpha = 0.35, size = 1.2, color = "black", show.legend = FALSE) +
    scale_fill_manual(values = c("Wild-Type" = "#4DBBD5", "Mutant" = "#E64B35")) +
    scale_x_discrete(labels = x_labs) +
    theme_classic() +
    labs(title = title_text, x = "", y = "Signature Expression (Z-score Mean)") +
    theme(
      plot.title = element_text(size = 11, face = "bold", hjust = 0.5),
      axis.text.x = element_text(size = 10, face = "bold"),
      axis.title.y = element_text(size = 9)
    ) +
    coord_cartesian(ylim = c(y_min - 2*h, y_max + 6*h)) +
    # Draw statistical bar
    annotate("path", x = c(1, 1, 2, 2), y = c(y_max + h, y_max + 1.5*h, y_max + 1.5*h, y_max + h), color = "black", linewidth = 0.5) +
    annotate("text", x = 1.5, y = y_max + 2.5*h, label = sprintf("Wilcoxon P = %.2e (%s)", pval, sig_label), size = 3.2, fontface = "bold", color = "#1A252C")
  
  plots[[i]] <- p
}

# Save figure using grid layout
png("arid1a_bulk_mutation_indices_r_verification_high_tmb.png", width = 3600, height = 1400, res = 300)
grid.newpage()
pushViewport(viewport(layout = grid.layout(1, 3)))
print(plots[[1]], vp = viewport(layout.pos.row = 1, layout.pos.col = 1))
print(plots[[2]], vp = viewport(layout.pos.row = 1, layout.pos.col = 2))
print(plots[[3]], vp = viewport(layout.pos.row = 1, layout.pos.col = 3))
dev.off()

message("R TME indices High TMB Mutation verification plot saved to arid1a_bulk_mutation_indices_r_verification_high_tmb.png")
