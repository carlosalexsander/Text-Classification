# This function transitions us from the document term matrix
# to the format we need for k-nearest neighbors

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