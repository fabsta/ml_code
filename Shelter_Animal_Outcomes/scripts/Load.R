# clear up environment
rm(list = ls())

# loading data
train <- read.csv("../data/train.csv", stringsAsFactors = F)
test <- read.csv("../data/test.csv", stringsAsFactors = F)
sample <- read.csv("../data/sample_submission.csv")

# summary(train)
#head(train[, -c(1, 5)])
#head(train)
# exclude AnimalID and OutcomeSubtype from train
train <- train[, -c(1, 5)]

# exclude ID from test
test <- test[, -1]