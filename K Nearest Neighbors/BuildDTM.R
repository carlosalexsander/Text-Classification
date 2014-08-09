# This function builds our document term matrix

# In the following function, we build our document-term matrix. This is
# a matrix that has reviews as rows and words as columns. A value in this
# matrix represents the number of times a particular review contains a 
# particular word. We are also removing sparse terms, so as to improve
# the performance of the k-nearest neighbor algorithm. We return a list
# with the type of sentiment and the document term matrix.
BuildDTM <- function(sentiment, directory, threshold) {
      file_directory <- paste(wd, "/", sentiment, sep = "")
      a <- Corpus(DirSource(directory = file_directory))
      b <- CleanCorpus(a)
      dtm <- DocumentTermMatrix(b)
      dtm <- removeSparseTerms(dtm, threshold)
      result <- list(review = sentiment, dtm = dtm)
      return(result)
}

# This creates a document term matrix for each sentiment.
dtm <- lapply(sentiments, BuildDTM, directory = wd, threshold = .48)