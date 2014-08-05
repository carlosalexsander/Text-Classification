# In the first part of this function we subset the document term matrix
# into a training set, a cross-validation set, and a test set.
# We also split the document term matrix into the numeric part and the
# labels.
# Then we call the k nearest neighbor algorithm.
# We generate a confusion matrix, comparing our predicions with the actual
# values. Then, calculate the f_score using precision and recall.
RunKNN <- function(complete_dtm, k_value){
      random_ordering <- sample(nrow(complete_dtm), nrow(complete_dtm), replace = FALSE)
      cutoff1 <- (ceiling(nrow(complete_dtm) * 0.6))
      cutoff2 <- (ceiling(nrow(complete_dtm) * 0.8))
      train_index <- random_ordering[1:cutoff1]
      cross_val_index <- random_ordering[(cutoff1 + 1):cutoff2]
      test_index <- random_ordering[(cutoff2 + 1):nrow(complete_dtm)]                             
      
      dtm_sent <- complete_dtm[,"ReviewType"]
      dtm_words <- complete_dtm[, !colnames(complete_dtm) %in% "ReviewType"]
      
      knn_pred <- knn(train = dtm_words[train_index, ], test = dtm_words[cross_val_index, ], cl = dtm_sent[train_index], k = k_value)
      confusion_matrix <- table("Predictions" = knn_pred, "Actual" = dtm_sent[cross_val_index])
      
      precision <- confusion_matrix[1,1] / (confusion_matrix[1,1] + confusion_matrix[1,2])
      recall <- confusion_matrix[1,1] / (confusion_matrix[1,1] + confusion_matrix[2,1])
      
      f_score <- 2 * (precision * recall) / (precision + recall)
      return(f_score)
}