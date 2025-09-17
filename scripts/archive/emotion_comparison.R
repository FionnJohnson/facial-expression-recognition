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



                          # --- Surprised vs Disgusted (single example of expression pair) --- #

library(geomorph)

n_landmarks <- 34
n_subjects_total <- 18

times_to_use <- c(1, 3, 5, 7)

expressions <- list(
  sur = "Surprised",
  dis = "Disgusted"
)

raw_data_list <- list(
  sur = sur,
  dis = dis
)

drop_subj <- c(6, 13)

keep_subj_indices  <- setdiff(1:n_subjects_total, drop_subj)
new_subjects    <- length(keep_subj_indices)

all_shapes_array <- array(NA, dim = c(n_landmarks, 2, new_subjects * length(expressions) * length(times_to_use)))

# --------------------------------------------------------------------------- #

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


# --- Perform Procrustes Analysis on all the shapes --- #

gpa_results <- gpagen(all_shapes_array, print.progress = FALSE)
aligned_shapes <- gpa_results$coords

aligned_shapes
dim(aligned_shapes)


# --- Reconstructing feature vectors into final X and Y --- #

X_all <- matrix(NA, 
                nrow = new_subjects * length(expressions), 
                ncol = n_landmarks * length(times_to_use) * 2)

y_all <- character(0)

rebuild_counter <- 1

for (expr_name in names(expressions)) {
  for (subj_num in 1:new_subjects) {
    
    t1_aligned <- aligned_shapes[, , rebuild_counter]
    t3_aligned <- aligned_shapes[, , rebuild_counter + 1]
    t5_aligned <- aligned_shapes[, , rebuild_counter + 2]
    t7_aligned <- aligned_shapes[, , rebuild_counter + 3]
    
    
    feature_vector <- c(as.vector(t1_aligned),
                        as.vector(t3_aligned),
                        as.vector(t5_aligned),
                        as.vector(t7_aligned))
    
    # Find the row in X_all we should put this shape in. 
    final_matrix_row <- (which(names(expressions) == expr_name) - 1) * new_subjects + subj_num
    X_all[final_matrix_row, ] <- feature_vector
    
    rebuild_counter <- rebuild_counter + 4
  }
  y_all <- c(y_all, rep(expressions[[expr_name]], new_subjects))
}

y_all <- factor(y_all)



                      # --- eXtreme Gradient Boosting (single example of one of the classifiers I used) --- #

library(xgboost)

subject_ids <- rep(1:new_subjects, times = 2)

# --- Leave-One-Subject-Out Cross-Validation --- #

all_predictions <- factor(character(0), levels = levels(y_all)) # one for each fold 
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
  
  
  # Apply PCA to the training set only
  pca_model <- prcomp(X_train, center = TRUE, scale. = TRUE)
  
  # Use the PCA model from the training data to transform BOTH sets
  X_train_pca <- pca_model$x[, 1:5]
  
  # Use predict() to apply the SAME transformation to the test data
  X_test_pca <- predict(pca_model, newdata = X_test)[, 1:5]
  
  
  # --- XG Boost --- #
  
  # Numeric Labels: "Happy" -> 0, "Sadness" -> 1, "Surprised" -> 2, "Disgusted" -> 3
  y_train_numeric <- as.numeric(as.factor(y_train)) - 1
  y_test_numeric <- as.numeric(as.factor(y_test)) - 1
  
  # Convert data into XG-Boost's special matrix format
  dtrain <- xgb.DMatrix(data = X_train_pca, label = y_train_numeric)
  dtest <- xgb.DMatrix(data = X_test_pca, label = y_test_numeric)
  
  # Set model parameters
  params <- list(
    objective = "multi:softmax",
    num_class = 2,
    max_depth = 3,               # Max depth of a tree
    eta = 0.1                    # Learning rate
    
  )
  
  # Train the model
  xgb_model <- xgb.train(params = params, data = dtrain, nrounds = 50) # nrounds is boosting rounds
  
  numeric_pred <- predict(xgb_model, dtest)
  
  # Convert the numeric predictions back to your original factor labels
  predictions <- factor(levels(y_all)[numeric_pred + 1], levels = levels(y_all))
  
  
  # Store predictions and true test labels
  all_predictions <- c(all_predictions, predictions)
  all_true_labels <- c(all_true_labels, y_test)
  
}


# --- 3. Calculate Final Accuracy ---

# Create a confusion matrix to see the results
conf_matrix <- table(Predicted = all_predictions, True = all_true_labels)
print("Confusion Matrix:")
print(conf_matrix)

# Calculate the overall accuracy
accuracy <- sum(diag(conf_matrix)) / sum(conf_matrix)
cat("\nFinal Subject-Independent Accuracy:", round(accuracy * 100, 2), "%\n")





