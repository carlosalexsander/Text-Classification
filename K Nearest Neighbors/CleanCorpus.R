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