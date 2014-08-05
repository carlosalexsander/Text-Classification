# This script finds the optimal values for the removeSparseterms threshold
# and the k-nearest neighbors parameter.

# This function creates a matrix to evaluate both parameters.
# We use the mean f score over ten function calls to evaluate the parameters
tseq <- seq(.47,.49,.005)
kseq <- seq(1,7,2)
f_scores3 <- rep(0,10)
param_matrix <- matrix(0,length(tseq), length(kseq))
testParameters <- function(tseq, kseq){
      for (i in 1:length(tseq)) {
            for (j in 1:length(kseq)) {
                  dtm = lapply(sentiments, BuildDTM, directory = wd, threshold = tseq[i])
                  review_dtm = lapply(dtm, AddSentiment)                      
                  complete_dtm = do.call(rbind.fill, review_dtm)
                  complete_dtm[is.na(complete_dtm)] = 0
                  for (k in 1:10) {
                        f_scores3[k] = RunKNN(complete_dtm = complete_dtm, k_value = kseq[j])
                  }
                  param_matrix[i,j] = mean(f_scores3)
            }
      }
      return(param_matrix)
}
ptest <- testParameters(tseq,kseq)

