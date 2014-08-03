# Michael Frasco
# 08/02/2014

library(NLP)
library(tm)
library(stringr)
library(plyr)
library(SnowballC)
library(class)

options(stringsAsFactors = FALSE)

sentiments <- c("neg", "pos")
wd <- "~/Documents/Non-School/Data Science/Text Classification/review_polarity/txt_sentoken"

CleanCorpus <- function(a) {
      b <- tm_map(a, removePunctuation)
      b <- tm_map(b, stripWhitespace, mc.cores=1)
      b <- tm_map(b, content_transformer(tolower))
      b <- tm_map(b, removeWords, stopwords("english"), mc.cores=1)
      return(b)
}

BuildDTM <- function(sentiment, directory) {
      file_directory <- paste(wd, "/", sentiment, sep = "")
      a <- Corpus(DirSource(directory = file_directory))
      b <- CleanCorpus(a)
      dtm <- DocumentTermMatrix(b)
      dtm <- removeSparseTerms(dtm, 0.7)
      result <- list(review = sentiment, dtm = dtm)
      return(result)
}

dtm <- lapply(sentiments, BuildDTM, directory = wd)
                             
AddSentiment <- function(dtm) {
      c <- data.matrix(dtm[["dtm"]])
      d <- as.data.frame(c, stringsAsFactors = FALSE)
      d <- cbind(d, rep(dtm[["review"]], nrow(d)))
      colnames(d)[ncol(d)] <- "ReviewType"
      return(d)
}

review_dtm <- lapply(dtm, AddSentiment)
                             
complete_dtm <- do.call(rbind.fill, review_dtm)
complete_dtm[is.na(complete_dtm)] <- 0
                             
set.seed(1000)
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
