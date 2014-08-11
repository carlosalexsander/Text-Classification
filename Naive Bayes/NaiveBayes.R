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


neg_docs <- list.files(paste(wd, sentiments[1], sep = "/"))
pos_docs <- list.files(paste(wd, sentiments[2], sep = "/"))
neg_train_index <- sample(1:length(neg_docs),ceiling(length(neg_docs) * .6))
pos_train_index <- sample(1:length(pos_docs),ceiling(length(pos_docs) * .6))
neg_training <- neg_docs[neg_train_index]
pos_training <- pos_docs[pos_train_index]
neg_testing <- neg_docs[-neg_train_index]
pos_testing <- pos_docs[-pos_train_index]
setwd(paste(wd, "/neg", sep = ""))
file.copy(from = neg_training, to = "neg_train")
neg_words <- c()
pos_words <- c()

# Use documents in each file to create a corpus. This might work better than
# manipulating the files individually. Although there is the duplicates
# problem.
neg_train_directory <- paste(wd, "neg_train", sep = "/")
neg_corpus <- Corpus(DirSource(directory = neg_train_directory))
neg_corpus <- tm_map(neg_corpus, removePunctuation)
neg_corpus <- tm_map(neg_corpus, stripWhitespace, mc.cores=1)
neg_corpus <- tm_map(neg_corpus, content_transformer(tolower))
neg_corpus <- tm_map(neg_corpus, removeWords, stopwords("english"), mc.cores=1)
#neg_corpus <- tm_map(neg_corpus, stemDocument, language = "english")

pos_train_directory <- paste(wd, "pos_test", sep = "/")
pos_corpus <- Corpus(DirSource(directory = pos_train_directory))
pos_corpus <- tm_map(pos_corpus, removePunctuation)
pos_corpus <- tm_map(pos_corpus, stripWhitespace, mc.cores=1)
pos_corpus <- tm_map(pos_corpus, content_transformer(tolower))
pos_corpus <- tm_map(pos_corpus, removeWords, stopwords("english"), mc.cores=1)
#pos_corpus <- tm_map(pos_corpus, stemDocument, language = "english")

neg_word_list = c()
for (i in 1:length(neg_corpus)) {
      review = neg_corpus[[i]]$content
      review = paste(review, collapse = "")
      tokenized_review = MC_tokenizer(review)
      duplicates = duplicated(tokenized_review)
      unique_tokenized = tokenized_review[!duplicates]
      neg_word_list = append(neg_word_list, unique_tokenized)
}
pos_word_list = c()
for (i in 1:length(pos_corpus)) {
      review = pos_corpus[[i]]$content
      review = paste(review, collapse = "")
      tokenized_review = MC_tokenizer(review)
      duplicates = duplicated(tokenized_review)
      unique_tokenized = tokenized_review[!duplicates]
      pos_word_list = append(pos_word_list, unique_tokenized)
}

# Create a frequency table for each word in the large word lists
neg_table <- table(neg_word_list)
pos_table <- table(pos_word_list)

# Here are the priors:
neg_prior = length(neg_training) / sum(length(neg_training) + length(pos_training))
pos_prior = length(pos_training) / sum(length(neg_training) + length(pos_training))
# This is the number of total words
neg_denominator <- length(neg_word_list) + length(neg_table)
pos_denominator <- length(pos_word_list) + length(pos_table)

neg_test_directory <- paste(wd, "neg_test", sep = "/")
neg_test_corpus <- Corpus(DirSource(directory = neg_test_directory))
neg_test_corpus <- tm_map(neg_test_corpus, removePunctuation)
neg_test_corpus <- tm_map(neg_test_corpus, stripWhitespace, mc.cores=1)
neg_test_corpus <- tm_map(neg_test_corpus, content_transformer(tolower))
neg_test_corpus <- tm_map(neg_test_corpus, removeWords, stopwords("english"), mc.cores=1)
#neg_test_corpus <- tm_map(neg_test_corpus, stemDocument, language = "english")

pos_test_directory <- paste(wd, "pos_test", sep = "/")
pos_test_corpus <- Corpus(DirSource(directory = pos_test_directory))
pos_test_corpus <- tm_map(pos_test_corpus, removePunctuation)
pos_test_corpus <- tm_map(pos_test_corpus, stripWhitespace, mc.cores=1)
pos_test_corpus <- tm_map(pos_test_corpus, content_transformer(tolower))
pos_test_corpus <- tm_map(pos_test_corpus, removeWords, stopwords("english"), mc.cores=1)
#pos_testcorpus <- tm_map(pos_test_corpus, stemDocument, language = "english")

neg_predictions = numeric()
for (i in 1:length(neg_corpus)) {
      review = neg_corpus[[i]]$content
      review = paste(review, collapse = "")
      tokenized_review = MC_tokenizer(review)
      duplicates = duplicated(tokenized_review)
      unique_tokenized = tokenized_review[!duplicates]
      for (j in length(unique_tokenized)) {
            word = unique_tokenized[j]
            neg_word_count = neg_table[word]
            if (is.na(neg_word_count)){
                  neg_word_count = 0
            }
            neg_word_count = neg_word_count + 1
            neg_posterior = c(neg_posterior, neg_word_count / neg_denominator)
            pos_word_count = pos_table[word]
            if (is.na(pos_word_count)){
                  pos_word_count = 0
            }
            pos_word_count = pos_word_count + 1
            pos_posterior = c(pos_posterior, pos_word_count / pos_denominator)
      }
      if (neg_prior * sum(log(neg_posterior)) > pos_prior * sum(log(pos_posterior))){
            neg_predictions[i] = 0
      }
      else{
            neg_predictions[i] = 1
      }
      neg_posterior = c()
      pos_posterior = c()
}


for (i in 1:length(pos_test_corpus)) {
      review = pos_test_corpus[[i]]$content
      review = paste(review, collapse = "")
      tokenized_review = MC_tokenizer(review)
      duplicates = duplicated(tokenized_review)
      unique_tokenized = tokenized_review[!duplicates]
      pos_word_list = append(pos_word_list, unique_tokenized)
}

# Process each document individually.
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


neg_string = paste(neg_words, collapse = " ")
pos_string = paste(pos_words, collapse = " ")
neg_denom = length(neg_words) + length(unique(neg_words))
pos_denom = length(pos_words) + length(unique(pos_words))
neg_sentiment_predictions = rep(0, length(neg_testing))
pos_sentiment_predictions = rep(1, length(pos_testing))

for (i in 1:length(neg_testing)) {
      doc_name = neg_testing[i]
      file_location = paste(wd, sentiments[1], doc_name, sep = "/")
      doc = readChar(file_location, file.info(file_location)$size)
      doc = gsub("[^[:alnum:][:space:]]", "", doc)
      doc = gsub("\n", "", doc)
      doc = gsub(" +", " ", doc)
      tokenized_doc = MC_tokenizer(doc)
      duplicates = duplicated(tokenized_doc)
      unique_tokens = tokenized_doc[!duplicates]
      for (j in 1:length(unique_tokens)) {
            word = unique_tokens[j]
            neg_word_count = str_count(neg_string, word) + 1
            pos_word_count = str_count(pos_string, word) + 1
            neg_posterior = c(neg_posterior, neg_word_count / neg_denom)
            pos_posterior = c(pos_posterior, pos_word_count / pos_denom)
      }
      if (neg_prior * sum(log(neg_posterior)) < pos_prior * sum(log(pos_posterior))) {
            neg_sentiment_predictions[i] = 1
      }
      neg_posterior = numeric()
      pos_posterior = numeric()
}



# For test document, remove all duplicate words. the compute Naive Bayes
# using the same equation.

# Cross validation with 10 folds. Inside each fold, there should be the 
# same numner of positive and negative.