# ===================================================================
# PART 1: GENERATE ALIGNED FEATURE VECTORS
# ===================================================================

# --- 1. Setup ---
library(geomorph)
library(ggplot2)

n_landmarks <- 34
n_subjects_total <- 18
times_to_use <- c(1, 3, 5, 7)
expressions <- list(hap = "Happy", sad = "Sadness", sur = "Surprise", dis = "Disgusted")
raw_data_list <- list(hap = hap, sad = sad, sur = sur, dis = dis)

drop_subj <- c(6, 13)
keep_subj_indices <- setdiff(1:n_subjects_total, drop_subj)
n_subjects_to_keep <- length(keep_subj_indices)

# --- 2. Create Array for RAW Shapes (to be aligned) ---
# Using a unique name to avoid conflicts: all_shapes_to_align_array
all_shapes_to_align_array <- array(NA, dim = c(
  n_landmarks,
  2,
  n_subjects_to_keep * length(expressions) * length(times_to_use)
))

array_counter <- 1
for (expr_data in raw_data_list) {
  for (subj_idx in keep_subj_indices) {
    for (time_val in times_to_use) {
      landmark_indices <- 1:n_landmarks + (subj_idx - 1) * n_landmarks
      shape_data <- expr_data[landmark_indices, , time_val]
      shape_data[, 2] <- -shape_data[, 2]
      all_shapes_to_align_array[, , array_counter] <- shape_data
      array_counter <- array_counter + 1
    }
  }
}

# --- 3. Perform Procrustes Analysis ---
# Using a unique name: aligned_shapes_array
gpa_results <- gpagen(all_shapes_to_align_array, print.progress = FALSE)
aligned_shapes_array <- gpa_results$coords
cat("Procrustes alignment complete.\n")


# --- 4. Reconstruct ALIGNED Feature Vectors ---
# Using unique names: X_all_aligned, y_all_aligned
X_all_aligned <- matrix(NA,
                        nrow = n_subjects_to_keep * length(expressions),
                        ncol = n_landmarks * length(times_to_use) * 2)
y_all_aligned <- character(0)

rebuild_counter <- 1
for (expr_name in names(expressions)) {
  for (subj_num in 1:n_subjects_to_keep) {
    t1_aligned <- aligned_shapes_array[, , rebuild_counter]
    t3_aligned <- aligned_shapes_array[, , rebuild_counter + 1]
    t5_aligned <- aligned_shapes_array[, , rebuild_counter + 2]
    t7_aligned <- aligned_shapes_array[, , rebuild_counter + 3]
    
    feature_vector <- c(as.vector(t1_aligned), as.vector(t3_aligned), as.vector(t5_aligned), as.vector(t7_aligned))
    
    final_matrix_row <- (which(names(expressions) == expr_name) - 1) * n_subjects_to_keep + subj_num
    X_all_aligned[final_matrix_row, ] <- feature_vector
    
    rebuild_counter <- rebuild_counter + 4
  }
  y_all_aligned <- c(y_all_aligned, rep(expressions[[expr_name]], n_subjects_to_keep))
}
y_all_aligned <- factor(y_all_aligned)

cat("Aligned feature vectors created successfully.\n")

# ===================================================================
# PART 2: CREATE PCA PLOTS FROM THE ALIGNED DATA
# ===================================================================

# --- 5. Prepare Data for Plotting ---
# Perform PCA on the individual ALIGNED shapes array
pca_on_all_aligned_shapes <- prcomp(t(array(aligned_shapes_array, dim = c(34 * 2, 256))), center = TRUE, scale = TRUE)
scores_all_aligned_shapes <- as.data.frame(pca_on_all_aligned_shapes$x)

# Create the metadata to describe each of the 256 aligned shapes
subject_id_meta_aligned <- rep(1:n_subjects_to_keep, each = length(times_to_use), times = length(expressions))
expression_meta_aligned <- rep(levels(y_all_aligned), each = n_subjects_to_keep * length(times_to_use))
timepoint_meta_aligned <- rep(times_to_use, times = n_subjects_to_keep * length(expressions))

plot_df_trajectory_aligned <- data.frame(
  Subject = subject_id_meta_aligned,
  Expression = expression_meta_aligned,
  Time = timepoint_meta_aligned,
  PC1 = scores_all_aligned_shapes$PC1,
  PC2 = scores_all_aligned_shapes$PC2
)

# --- 6. Generate the Plot ---
cat("\nGenerating PCA trajectory plot from ALIGNED data...\n")

plot_aligned <- ggplot(plot_df_trajectory_aligned, aes(x = PC1, y = PC2, color = Expression, group = Subject)) +
  geom_point(aes(shape = as.factor(Time)), size = 2) +
  geom_path(alpha = 0.5) +
  facet_wrap(~ Expression, scales = "free") +
  labs(
    title = "Expression Trajectories from Procrustes Aligned Landmark Data",
    subtitle = "Note the consistent directionality of trajectories within each class",
    x = "Principal Component 1",
    y = "Principal Component 2",
    shape = "Time Point"
  ) +
  theme_minimal() +
  scale_color_brewer(palette = "Set1")

print(plot_aligned)
