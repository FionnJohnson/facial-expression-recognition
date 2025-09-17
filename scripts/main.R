
          # ===================================================== #
              # --- Section 1. Data Loading and Cleaning --- #
          # ===================================================== #

# --- Loading Data --- #

dis = 100 * scan("disg.txt")
dis = array(dis, c(612, 2, 7))

sur=100*scan("sur.txt")
sur=array(sur,c(612,2,7))
ind =1:34+11*34
cc=c(1,2,3,4,5,11,12,13,14,15,16,17,18,6,7,8,9,10,19:34)
sur[ind,,5]=sur[ind[cc],,5]
sur[ind[c(17,18)],,1]=sur[ind[c(18,17)],,1]

sad=100*scan("sad.txt")
sad=array(sad,c(612,2,7))

hap=100*scan("hap.txt")
hap=array(hap,c(612,2,7))


          # ===================================================== #
                # --- Section 2. Feature engineering --- #
          # ===================================================== #


library(geomorph)


# Setup 
n_landmarks <- 34
n_subjects_total <- 18
expressions <- list(hap = "Happy", sad = "Sadness", sur = "Surprise", dis = "Disgusted")
raw_data_list <- list(hap = hap, sad = sad, sur = sur, dis = dis)

# --- FOR EXPERIMENT 3: EXPRESSION PAIRS --- #
# expressions <- list(sur = "Surprised", dis = "Disgusted")
# raw_data_list <- list(sur = sur, dis = dis)
# ------------------------------------------ #


drop_subj <- c(6, 13)
keep_subj_indices  <- setdiff(1:n_subjects_total, drop_subj)
new_subjects    <- length(keep_subj_indices)

# -- Choose time points for experiment -- #

# times_to_use <- 1:7
times_to_use <- c(1, 3, 5, 7)
# times_to_use <- 7

# ------------------------------------------------------------------------ #

all_shapes_array <- array(NA, dim = c(n_landmarks, 2, new_subjects * length(expressions) * length(times_to_use)))


array_counter <- 1
for (expr_data in raw_data_list) {
  for (subj_idx in keep_subj_indices) {
    for (time_val in times_to_use) {
      
      landmark_indices <- 1:n_landmarks + (subj_idx - 1) * n_landmarks
      shape_data <- expr_data[landmark_indices, , time_val]
      
      shape_data[, 2] <- -shape_data[, 2]
      
      all_shapes_array[, , array_counter] <- shape_data
      array_counter <- array_counter + 1
    }
  }
}

# --- Perform GPA --- #

# comment these two lines to perform experiment 2 (RAW vs ALIGNED)
gpa_results <- gpagen(all_shapes_array, print.progress = FALSE)
aligned_shapes <- gpa_results$coords

# aligned_shapes <- all_shapes_array  # uncomment for RAW analysis

# --- Reconstructing feature vectors into final X and Y --- #

X_all <- matrix(NA, 
                nrow = new_subjects * length(expressions), 
                ncol = n_landmarks * length(times_to_use) * 2)
y_all <- character(0)
rebuild_counter <- 1

for (expr_name in names(expressions)) {
  for (subj_num in 1:new_subjects) {
    
    # --- Option 1: For 4 Time Points (1, 3, 5, 7) --- #
    t1 <- aligned_shapes[, , rebuild_counter]
    t3 <- aligned_shapes[, , rebuild_counter + 1]
    t5 <- aligned_shapes[, , rebuild_counter + 2]
    t7 <- aligned_shapes[, , rebuild_counter + 3]
    feature_vector <- c(as.vector(t1), as.vector(t3), as.vector(t5), as.vector(t7))
    
    # --- Option 2: For All 7 Time Points --- #
    # t1 <- aligned_shapes[, , rebuild_counter]
    # t2 <- aligned_shapes[, , rebuild_counter + 1]
    # t3 <- aligned_shapes[, , rebuild_counter + 2]
    # t4 <- aligned_shapes[, , rebuild_counter + 3]
    # t5 <- aligned_shapes[, , rebuild_counter + 4]
    # t6 <- aligned_shapes[, , rebuild_counter + 5]
    # t7 <- aligned_shapes[, , rebuild_counter + 6]
    # feature_vector <- c(as.vector(t1), as.vector(t2), as.vector(t3), as.vector(t4),
    #                     as.vector(t5), as.vector(t6), as.vector(t7))
    
    # # --- Option 3: For 7th Time Point Only --- #
    # t7 <- aligned_shapes[, , rebuild_counter]
    # feature_vector <- as.vector(t7)
    
    
    
    final_matrix_row <- (which(names(expressions) == expr_name) - 1) * new_subjects + subj_num
    X_all[final_matrix_row, ] <- feature_vector
    
    rebuild_counter <- rebuild_counter + length(times_to_use)
  }
  y_all <- c(y_all, rep(expressions[[expr_name]], new_subjects))
}
y_all <- factor(y_all)



            # ===================================================== #
             # --- Section 3. Main Analysis Pipeline (LOSO-CV) --- #
            # ===================================================== #


subject_ids <- rep(1:new_subjects, times = length(expressions))

all_predictions <- factor(character(0), levels = levels(y_all))
all_true_labels <- factor(character(0), levels = levels(y_all))


for (i in 1:new_subjects) {
  
  # For this specific fold, find the corresponding / appropriate test and train indices
  test_indices <- which(subject_ids == i)
  train_indices <- which(subject_ids != i)
  
  # Index out from the X_all and y_all arrays.
  X_train <- X_all[train_indices, ]
  y_train <- y_all[train_indices]
  X_test <- X_all[test_indices, ]
  y_test <- y_all[test_indices]
  
  # Choose number of principal components
  num_PCs <- 10
  pca_model <- prcomp(X_train, center = TRUE, scale. = TRUE)
  X_train_pca <- pca_model$x[, 1:num_PCs]
  X_test_pca <- predict(pca_model, newdata = X_test)[, 1:num_PCs]
  
  
  # --- CHOOSE CLASSIFIER (Uncomment one) --- #
  
  # --- Logistic Regression --- #
  library(nnet)
  # Data frame is needed for Logistic regression
  train_data_df <- data.frame(X_train_pca, expression = y_train)
  test_data_df <- data.frame(X_test_pca)
  # Train and predict
  lr_model <- multinom(expression ~ ., data = train_data_df, trace = FALSE)
  predictions <- predict(lr_model, newdata = test_data_df)
  
  
  # --- XG Boost --- #
  # library(xgboost)
  # y_train_numeric <- as.numeric(y_train) - 1
  # y_test_numeric <- as.numeric(y_test) - 1
  # # Convert data into XGBoost's special matrix format
  # dtrain <- xgb.DMatrix(data = X_train_pca, label = y_train_numeric)
  # dtest <- xgb.DMatrix(data = X_test_pca, label = y_test_numeric)
  # 
  # # Set model parameters (Change num_class = 2 for experiment 3, otherwise keep at 4)
  # params <- list(objective = "multi:softmax", num_class = 4, max_depth = 3, eta = 0.1)
  # xgb_model <- xgb.train(params = params, data = dtrain, nrounds = 50)
  # numeric_pred <- predict(xgb_model, dtest)
  # # Convert the numeric predictions back to your original factor labels
  # predictions <- factor(levels(y_all)[numeric_pred + 1], levels = levels(y_all))
  
  
  # --- K-Nearest Neighbours --- #
  # library(class)
  # predictions <- knn(train = X_train_pca, 
  #                    test = X_test_pca, 
  #                    cl = as.factor(y_train), 
  #                    k = 3)
  
  
  # --- Support Vector Machine --- #
  # library(e1071)
  # svm_model <- svm(X_train_pca, y_train, type = "C-classification", kernel = "linear")
  # predictions <- predict(svm_model, X_test_pca)
  
  
  # --- Decision Tree model --- #
  # library(rpart)
  # library(rpart.plot)
  # train_data_df <- data.frame(X_train_pca, expression = y_train)
  # test_data_df <- data.frame(X_test_pca)
  # tree_model <- rpart(expression ~ ., data = train_data_df, method = "class")
  # rpart.plot(tree_model)
  # predictions <- predict(tree_model, newdata = test_data_df, type = "class")
  
  
  
  # --- Random Forests model --- #
  # library(randomForest)
  # set.seed(152) # to reproduce the same results
  # randF_model <- randomForest(x = X_train_pca, y = y_train, ntree = 500)
  # predictions <- predict(randF_model, newdata = X_test_pca)
  
  
  
  
  # --- Store predictions and true test labels --- #
  all_predictions <- c(all_predictions, predictions)
  all_true_labels <- c(all_true_labels, y_test)
  
}


# ---  Calculate Final Accuracy and Confusion matrix --- #
conf_matrix <- table(Predicted = all_predictions, True = all_true_labels)
print(conf_matrix)

accuracy <- sum(diag(conf_matrix)) / sum(conf_matrix)
cat("\nFinal Subject-Independent Accuracy:", round(accuracy * 100, 2), "%\n")



          # ===================================================== #
                  # --- Section 4. Data Visualisation --- #
          # ===================================================== #

# To be run after the feature space has been constructed (Section 2)
# Best with the four time frame feature space

library(ggplot2)

# --- 1. Prepare Data for Plotting --- #

# Calculate total number of individual shapes
n_total_shapes <- new_subjects * length(expressions) * length(times_to_use)

pca_on_all_shapes <- prcomp(t(array(aligned_shapes, dim = c(34 * 2, n_total_shapes))), center = TRUE, scale = TRUE)
scores_all_shapes <- as.data.frame(pca_on_all_shapes$x)

# Create the metadata to describe each of the 256 shapes
subject_id_meta <- rep(1:new_subjects, each = length(times_to_use), times = length(expressions))
expression_meta <- rep(levels(y_all), each = new_subjects * length(times_to_use))
timepoint_meta <- rep(times_to_use, times = new_subjects * length(expressions))

plot_df_trajectory <- data.frame(
  Subject = subject_id_meta,
  Expression = expression_meta,
  Time = timepoint_meta,
  PC1 = scores_all_shapes$PC1,
  PC2 = scores_all_shapes$PC2
)

# --- 2. Generate the Trajectory Plot --- #

trajectory_plot <- ggplot(plot_df_trajectory, aes(x = PC1, y = PC2, color = Expression, group = Subject)) +
  geom_point(aes(shape = as.factor(Time)), size = 2) +
  geom_path(alpha = 0.5) +
  facet_wrap(~ Expression, scales = "free") +
  labs(
    title = "Expression Trajectories in Principal Component Space",
    x = "Principal Component 1",
    y = "Principal Component 2",
    shape = "Time Point"
  ) +
  theme_minimal() +
  scale_color_brewer(palette = "Set1")

print(trajectory_plot)

