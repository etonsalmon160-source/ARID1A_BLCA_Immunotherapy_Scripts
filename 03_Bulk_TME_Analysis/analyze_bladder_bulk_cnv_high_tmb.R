# analyze_bladder_bulk_cnv_high_tmb.R
# Load required libraries
library(httr)
library(jsonlite)
library(ggplot2)

study_id <- "blca_tcga_pan_can_atlas_2018"

message("1. Fetching TCGA BLCA clinical data from cBioPortal...")
url_samples <- paste0("https://www.cbioportal.org/api/studies/", study_id, "/clinical-data?projection=DETAILED&clinicalDataType=SAMPLE")

response_samples <- GET(url_samples, accept_json())
if (status_code(response_samples) != 200) {
  stop("Failed to fetch clinical data from cBioPortal.")
}
sample_data <- fromJSON(content(response_samples, as = "text", encoding = "UTF-8"))

# Reshape long clinical format to wide sample format
message("Reshaping clinical data...")
sample_ids <- unique(sample_data$sampleId)
samples_df <- data.frame(SAMPLE_ID = sample_ids, stringsAsFactors = FALSE)

# Extract relevant attributes
attributes <- c("FRACTION_GENOME_ALTERED", "ANEUPLOIDY_SCORE", "MUTATION_COUNT")
for (attr in attributes) {
  attr_data <- sample_data[sample_data$clinicalAttributeId == attr, ]
  # Merge by sampleId
  merged <- data.frame(sampleId = attr_data$sampleId, val = as.numeric(attr_data$value), stringsAsFactors = FALSE)
  colnames(merged)[2] <- attr
  samples_df <- merge(samples_df, merged, by.x = "SAMPLE_ID", by.y = "sampleId", all.x = TRUE)
}

# Drop rows where FRACTION_GENOME_ALTERED or MUTATION_COUNT is NA
samples_df <- samples_df[!is.na(samples_df$FRACTION_GENOME_ALTERED) & !is.na(samples_df$MUTATION_COUNT), ]

# Determine High TMB threshold (top 33%, percentile 0.67)
tmb_threshold <- quantile(samples_df$MUTATION_COUNT, 0.67, na.rm = TRUE)
message(sprintf("TMB High threshold (>= 67th percentile): %.1f mutations", tmb_threshold))

# Filter for High TMB cohort
samples_df_high <- samples_df[samples_df$MUTATION_COUNT >= tmb_threshold, ]
message("Total High TMB samples with FGA data: ", nrow(samples_df_high))

# 2. Fetch ARID1A mutation status
message("2. Fetching ARID1A mutations...")
url_mutations <- paste0("https://www.cbioportal.org/api/molecular-profiles/", study_id, "_mutations/mutations/fetch")
fetch_body <- list(
  entrezGeneIds = I(8289), # Protect Entrez Gene ID from unboxing to [8289]
  sampleIds = samples_df_high$SAMPLE_ID
)

response_mutations <- POST(
  url_mutations,
  body = toJSON(fetch_body, auto_unbox = TRUE),
  content_type_json(),
  accept_json()
)
if (status_code(response_mutations) != 200) {
  stop("Failed to fetch mutation data from cBioPortal.")
}
mut_data <- fromJSON(content(response_mutations, as = "text", encoding = "UTF-8"))

mutated_samples <- unique(mut_data$sampleId)

# Assign ARID1A status
samples_df_high$ARID1A_Status <- ifelse(samples_df_high$SAMPLE_ID %in% mutated_samples, "Mutant", "Wild-Type")

# 3. Perform statistical analysis
message("\n=== Comparisons in R (High TMB Cohort) ===")

wt_fga <- na.omit(samples_df_high[samples_df_high$ARID1A_Status == "Wild-Type", "FRACTION_GENOME_ALTERED"])
mut_fga <- na.omit(samples_df_high[samples_df_high$ARID1A_Status == "Mutant", "FRACTION_GENOME_ALTERED"])
pval_fga <- wilcox.test(wt_fga, mut_fga, alternative = "two.sided")$p.value

cat("\nMetric: FRACTION_GENOME_ALTERED\n")
cat(sprintf("Wild-Type (N=%d): Mean=%.4f, Median=%.4f\n", length(wt_fga), mean(wt_fga), median(wt_fga)))
cat(sprintf("Mutant (N=%d): Mean=%.4f, Median=%.4f\n", length(mut_fga), mean(mut_fga), median(mut_fga)))
cat(sprintf("Wilcoxon rank-sum test p-value: %.6f\n", pval_fga))

wt_aneu <- na.omit(samples_df_high[samples_df_high$ARID1A_Status == "Wild-Type", "ANEUPLOIDY_SCORE"])
mut_aneu <- na.omit(samples_df_high[samples_df_high$ARID1A_Status == "Mutant", "ANEUPLOIDY_SCORE"])
pval_aneu <- wilcox.test(wt_aneu, mut_aneu, alternative = "two.sided")$p.value

cat("\nMetric: ANEUPLOIDY_SCORE\n")
cat(sprintf("Wild-Type (N=%d): Mean=%.4f, Median=%.4f\n", length(wt_aneu), mean(wt_aneu), median(wt_aneu)))
cat(sprintf("Mutant (N=%d): Mean=%.4f, Median=%.4f\n", length(mut_aneu), mean(mut_aneu), median(mut_aneu)))
cat(sprintf("Wilcoxon rank-sum test p-value: %.6f\n", pval_aneu))

# 4. Generate validation plot in R
message("\nGenerating verification plot in R (High TMB)...")
p1 <- ggplot(samples_df_high, aes(x = ARID1A_Status, y = FRACTION_GENOME_ALTERED, fill = ARID1A_Status)) +
  geom_violin(alpha = 0.6, trim = FALSE, show.legend = FALSE) +
  geom_boxplot(width = 0.1, fill = "white", outlier.shape = NA, show.legend = FALSE) +
  geom_jitter(width = 0.2, alpha = 0.3, size = 1.2, show.legend = FALSE) +
  scale_fill_manual(values = c("Wild-Type" = "#4DBBD5", "Mutant" = "#E64B35")) +
  theme_classic() +
  labs(title = "Fraction Genome Altered\n(High TMB Cohort)", x = "ARID1A Status", y = "FGA") +
  coord_cartesian(ylim = c(-0.05, 1.15)) +
  annotate("text", x = 1.5, y = 1.08, label = sprintf("p = %.4f", pval_fga), size = 4.5, fontface = "bold", color = "#2C3E50")

p2 <- ggplot(samples_df_high, aes(x = ARID1A_Status, y = ANEUPLOIDY_SCORE, fill = ARID1A_Status)) +
  geom_violin(alpha = 0.6, trim = FALSE, show.legend = FALSE) +
  geom_boxplot(width = 0.1, fill = "white", outlier.shape = NA, show.legend = FALSE) +
  geom_jitter(width = 0.2, alpha = 0.3, size = 1.2, show.legend = FALSE) +
  scale_fill_manual(values = c("Wild-Type" = "#4DBBD5", "Mutant" = "#E64B35")) +
  theme_classic() +
  labs(title = "Aneuploidy Score\n(High TMB Cohort)", x = "ARID1A Status", y = "Score") +
  coord_cartesian(ylim = c(-5, 45)) +
  annotate("text", x = 1.5, y = 41, label = sprintf("p = %.4f", pval_aneu), size = 4.5, fontface = "bold", color = "#2C3E50")

# Save as PNG using R's built-in grid package
png("bladder_tcga_bulk_cnv_r_verification_high_tmb.png", width = 2400, height = 1200, res = 300)
library(grid)
grid.newpage()
pushViewport(viewport(layout = grid.layout(1, 2)))
print(p1, vp = viewport(layout.pos.row = 1, layout.pos.col = 1))
print(p2, vp = viewport(layout.pos.row = 1, layout.pos.col = 2))
dev.off()
message("R High TMB verification plot saved to bladder_tcga_bulk_cnv_r_verification_high_tmb.png")
