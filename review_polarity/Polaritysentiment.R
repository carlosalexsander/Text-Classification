# Michael Frasco
# 08/02/2014

# We need several libraries in addition to the base package in R
library(NLP)
library(tm)
library(stringr)
library(plyr)
library(SnowballC)
library(class)

# Set as the default option
options(stringsAsFactors = FALSE)

# There are two types of sentiments in the movie reviews: negative and
# positive. We create a vector of these sentiments, which correspond to
# the subdirectories where the reviews are located.
sentiments <- c("neg", "pos")
wd <- "~/Documents/Non-School/Data Science/Text Classification/review_polarity/txt_sentoken"


# This is a function that takes a corpus of documents and applies some
# cleaning functions to that corpus. We are dealing with two corpuses:
# the first is a corpus of the negative reviews and the second has positive
# reviews. The most important line in this function is the removeWords
# function. This function removes stop words (i.e. parts of speech like
# articles, which convey very little meaning about the review.) We choose
# not to stem the words in the corpus.
CleanCorpus <- function(a) {
      b <- tm_map(a, removePunctuation)
      b <- tm_map(b, stripWhitespace, mc.cores=1)
      b <- tm_map(b, content_transformer(tolower))
      b <- tm_map(b, removeWords, stopwords("english"), mc.cores=1)
      #b <- tm_map(b, stemDocument, language = "english")
      return(b)
}


# In the following function, we build our document-term matrix. This is
# a matrix that has reviews as rows and words as columns. A value in this
# matrix represents the number of times a particular review contains a 
# particular word. We are also removing sparse terms, so as to improve
# the performance of the k-nearest neighbor algorithm. We return a list
# with the type of sentiment and the document term matrix.
BuildDTM <- function(sentiment, directory) {
      file_directory <- paste(wd, "/", sentiment, sep = "")
      a <- Corpus(DirSource(directory = file_directory))
      b <- CleanCorpus(a)
      dtm <- DocumentTermMatrix(b)
      dtm <- removeSparseTerms(dtm, 0.7)
      result <- list(review = sentiment, dtm = dtm)
      return(result)
}

# This creates a document term matrix for each sentiment.
dtm <- lapply(sentiments, BuildDTM, directory = wd)

# We attach the sentiment type as a column at the end of our document
# term matrix.
AddSentiment <- function(dtm) {
      c <- data.matrix(dtm[["dtm"]])
      d <- as.data.frame(c, stringsAsFactors = FALSE)
      d <- cbind(d, rep(dtm[["review"]], nrow(d)))
      colnames(d)[ncol(d)] <- "ReviewType"
      return(d)
}

# The next chunk of code merges the two document-term matrices into a
# single matrix with both negative and positive reviews.
review_dtm <- lapply(dtm, AddSentiment)                      
complete_dtm <- do.call(rbind.fill, review_dtm)
complete_dtm[is.na(complete_dtm)] <- 0

# Set the seed for reproducability.
set.seed(1000)

# In the first part of this function we subset the document term matrix
# into a training set, a cross-validation set, and a test set.
# We also split the document term matrix into the numeric part and the
# labels.
# Then we call the k nearest neighbor algorithm.
# We generate a confusion matrix, comparing our predicions with the actual
# values. Then, calculate the f_score using precision and recall.
RunKNN <- function(complete_dtm){
      random_ordering <- sample(nrow(complete_dtm), nrow(complete_dtm), replace = FALSE)
      cutoff1 <- (ceiling(nrow(complete_dtm) * 0.6))
      cutoff2 <- (ceiling(nrow(complete_dtm) * 0.8))
      train_index <- random_ordering[1:cutoff1]
      cross_val_index <- random_ordering[(cutoff1 + 1):cutoff2]
      test_index <- random_ordering[(cutoff2 + 1):nrow(complete_dtm)]                             
      
      dtm_sent <- complete_dtm[,"ReviewType"]
      dtm_words <- complete_dtm[, !colnames(complete_dtm) %in% "ReviewType"]
      
      knn_pred <- knn(dtm_words[train_index, ], dtm_words[cross_val_index, ], dtm_sent[train_index])
      confusion_matrix <- table("Predictions" = knn_pred, "Actual" = dtm_sent[cross_val_index])
      
      precision <- confusion_matrix[1,1] / (confusion_matrix[1,1] + confusion_matrix[1,2])
      recall <- confusion_matrix[1,1] / (confusion_matrix[1,1] + confusion_matrix[2,1])
      
      f_score <- 2 * (precision * recall) / (precision + recall)
      return(f_score)
}

# This is a test set that allows us to fine tune the algorithm.
test_script <- function(vector){
      for (i in 1:10) {
            vector[i] <- RunKNN(complete_dtm)
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

