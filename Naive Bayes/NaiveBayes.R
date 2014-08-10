# Michael Frasco
# 08/09/2014
# Naive Bayes algorithm for text classification

# Initial set up
library(NLP)
library(tm)
library(stringr)
library(stringi)
set.seed(666)

wd <- "/Users/mfrasco/Documents/Non-School/Data Science/Text Classification/Review_polarity_data/txt_sentoken"
sentiments <- c("neg", "pos")


# Tokenization (perform for each document)
      # Need to remove duplicate words
neg_docs <- list.files(paste(wd, sentiments[1], sep = "/"))
pos_docs <- list.files(paste(wd, sentiments[2], sep = "/"))
neg_train_index <- sample(1:length(neg_docs),ceiling(length(neg_docs) * .6))
pos_train_index <- sample(1:length(pos_docs),ceiling(length(pos_docs) * .6))
neg_training <- neg_docs[neg_train_index]
pos_training <- pos_docs[pos_train_index]
neg_testing <- neg_docs[-neg_train_index]
pos_testing <- pos_docs[-pos_train_index]
neg_words <- c()
pos_words <- c()

for (i in 1:length(neg_training)){
      doc_name = neg_training[i]
      file_location = paste(wd, sentiments[1], doc_name, sep = "/")
      doc = readChar(file_location, file.info(file_location)$size)
      doc = gsub("[^[:alnum:][:space:]]", "", doc)
      doc = gsub("\n", "", doc)
      doc = gsub(" +", " ", doc)
      tokenized_doc = MC_tokenizer(doc)
      duplicates = duplicated(tokenized_doc)
      unique_tokens = tokenized_doc[!duplicates]
      neg_words = append(neg_words, unique_tokens)
}

for (i in 1:length(pos_training)){
      doc_name = pos_training[i]
      file_location = paste(wd, sentiments[2], doc_name, sep = "/")
      doc = readChar(file_location, file.info(file_location)$size)
      doc = gsub("[^[:alnum:][:space:]]", "", doc)
      doc = gsub("\n", "", doc)
      doc = gsub(" +", " ", doc)
      tokenized_doc = MC_tokenizer(doc)
      duplicates = duplicated(tokenized_doc)
      unique_tokens = tokenized_doc[!duplicates]
      pos_words = append(pos_words, unique_tokens)
}

# Feature Extraction (the words)
      # How to handle negation
            # Add NOT_ to every word betweeen negation and the following
            # punctuation.

# Classification using different classifiers (Naive Bayes)
      # Naive Bayes
            # The most likely class according to Naive Bayes is the class
            # that maximizes the product of the probability of the class 
            # (the prior) and the product over all positions in the document
            # of the likelihood of the word in that document given the class

            # In practice, we use simply laplace or add-one smoothing. So
            # the probability of the word given the class equals the count
            # of the word in the class + 1 divided by the count of words
            # in the class + size of te vocabulary.

            # Binarized (Boolean feature) multinomial naive bayes
                  # For sentiment word occurrence may matter more than word
                  # frequency. Boolean multinomial naive bayes clips all
                  # the word counts in each document at 1
# Summary
      # From training corpus, extract Vocabulary
      # Calculate probability of each class
            # for each class divide the number of documents with that class
            # by the total number of documents
      # Calculate the probability of a word given the class
            # Remove duplicates in each document. That is, for each word type
            # in a document, retain only a single instance of the word
            # Then for each word in the vocabulary, take the number of
            # of documents the word occurs in and add 1, then divide by
            # the total number of words

neg_prior = length(neg_docs) / sum(length(neg_docs) + length(pos_docs))
pos_prior = length(pos_docs) / sum(length(neg_docs) + length(pos_docs))
neg_string = paste(neg_words, collapse = " ")
pos_string = paste(pos_words, collapse = " ")
neg_denom = length(neg_words) + length(unique(neg_words))
pos_denom = length(pos_words) + length(unique(pos_words))
neg_sentiment_predictions = rep(0, length(neg_testing))
pos_sentiment_predictions = rep(1, length(pos_testing))

for (i in length(neg_testing)) {
      doc_name = neg_testing[i]
      file_location = paste(wd, sentiments[1], doc_name, sep = "/")
      doc = readChar(file_location, file.info(file_location)$size)
      doc = gsub("[^[:alnum:][:space:]]", "", doc)
      doc = gsub("\n", "", doc)
      doc = gsub(" +", " ", doc)
      tokenized_doc = MC_tokenizer(doc)
      duplicates = duplicated(tokenized_doc)
      unique_tokens = tokenized_doc[!duplicates]
      for (i in length(unique_tokens)) {
            word = unique_tokens[i]
            neg_word_count = str_count(neg_string, word) + 1
            pos_word_count = str_count(pos_string, word) + 1
            neg_posterior = c(neg_posterior, neg_word_count / neg_denom)
            pos_posterior = c(pos_posterior, pos_word_count / pos_denom)
      }
      neg_prob = neg_prior * prod(neg_posterior)
      pos_prob = pos_prior * prod(pos_posterior)
      if (neg_prob < pos_prob) {
            neg_sentiment_predictions[i] = 1
      }
      neg_posterior = numeric()
      pos_posterior = numeric()
}



# For test document, remove all duplicate words. the compute Naive Bayes
# using the same equation.

# Cross validation with 10 folds. Inside each fold, there should be the 
# same numner of positive and negative.