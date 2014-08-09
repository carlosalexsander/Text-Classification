# These function calls test the importance of each component of the
# CleanCorpus Function

# This is a test set that allows us to fine tune the algorithm.
test_script <- function(vector, k_value){
      for (i in 1:10) {
            vector[i] <- RunKNN(complete_dtm, k_value)
      }
      return(summary(vector))
}
# The following are all function calls that evaluate the entire script
# ten times and return a summary vector of the f_scores. Each time, a
# single component of the script was changed, so we can determine which
# parts of the script are most important.

# Here nothing was changed. Our mean f_score is .7568
base <- test_script(numeric())

# The corpus was not cleaned before creating the document term matrix
# the mean f_score dropped significantly to .6525
noclean <- test_script(numeric())

# We added stemming. The mean f_score remained the same as the base.
stem <- test_script(numeric())

# The following three function calls had no affect compared to the base
# case. However, I suspect the reason for this is that the remove sparse
# terms level is relatively high. If we lower this threshold, then these
# functions will probably demonstrate value.
punc <- test_script(numeric())
white <- test_script(numeric())
lower <- test_script(numeric())

# This accounted for the entire drop in performance of the cleaning function.
# Removing stop words is very important because it reduces the dimensionality
# of the DTM, which is very important for the k nearest neighbor algorithm.
stopwords <- test_script(numeric())