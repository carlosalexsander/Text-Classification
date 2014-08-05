# This is a function that tests each parameter individually

threshold_sequence <- seq(from = 0.3, to = 0.8, by = 0.05)
f_scores <- rep(0,10)
vector <- rep(0,length(threshold_sequence))

testForThreshold <- function(threshold_sequence, k_value) {
      for (i in 1:length(threshold_sequence)) {
            threshold = threshold_sequence[i]
            dtm = lapply(sentiments, BuildDTM, directory = wd, threshold = threshold)
            review_dtm = lapply(dtm, AddSentiment)                      
            complete_dtm = do.call(rbind.fill, review_dtm)
            complete_dtm[is.na(complete_dtm)] = 0
            test_dtm = complete_dtm
            set.seed(1000)
            for (j in 1:10) {
                  f_scores[j] <- RunKNN(test_dtm, k_value)
            }
            vector[i] <- mean(f_scores)
      }
      return(vector)
}

max_threshold <- testForThreshold(threshold_sequence)
threshold_se2 <- seq(.4,.6,.02)
max_threshold2 <- testForThreshold(threshold_se2)
#Set threshold equal to .48

testForK <- function(ksequence) {
      dtm = lapply(sentiments, BuildDTM, directory = wd, threshold = 0.48)
      review_dtm = lapply(dtm, AddSentiment)                      
      complete_dtm = do.call(rbind.fill, review_dtm)
      complete_dtm[is.na(complete_dtm)] = 0
      for (i in 1:length(ksequence)) {
            set.seed(1000)
            for (j in 1:20) {
                  f_scores2[j] = RunKNN(complete_dtm = complete_dtm, k_value = ksequence[i])
            }
            vector2[i] = mean(f_scores2)
      }
      return(vector2)
}

ksequence <- seq(1,20,1)
f_scores2 <- rep(0,20)
vector2 <- rep(0,length(ksequence))

max_k <- testForK(ksequence)