# DrivenData Blood Donations Warm-up Comp
#
# AIM: predict whether someone made a blood donation in March 2007 based on past
# donation history
#
# N.B. run 0-blooddonations (to line 88) before this script
#
#
#
# load packages ===============================================================
library(ggplot2)
library(e1071)
library(car)
library(caret)
library(pROC)
library(randomGLM)
library(splines)

# Logistic regression ==========================================================
# caret train control settings 10 x 5 fold cv
ctrl <- trainControl(method = "repeatedcv",
  number = 5,
  repeats = 10,
  ## Estimate class probabilities
  classProbs = TRUE,
  summaryFunction = logloss_caret)

# "Basic" logistic regression model (no interactions, polynomial terms) -------
set.seed(1056)  
glm_model <- train(March.2007 ~ .,
  data = training,
  method = "glm", 
  family = "binomial",
  trControl = ctrl)

glm_model # model performance
summary(glm_model)

# residual analysis plots
plot(glm_model$finalModel)  
plot(training$Last.Donation, glm_model$finalModel$residuals)
plot(training$First.Donation, glm_model$finalModel$residuals)
plot(training$Num.Donations, glm_model$finalModel$residuals)

#plot(training$avg_times, glm_model$finalModel$residuals)
plot(training$overdue, glm_model$finalModel$residuals)

# Logistic regression model w/ interaction terms ------------------------------
set.seed(1056)  
glm_model2 <- train(March.2007 ~ . +
    I(Last.Donation*Num.Donations) +
    I(Last.Donation*First.Donation) +
    I(First.Donation*Num.Donations) +
    I(Last.Donation*Num.Donations*First.Donation) +
    overdue,
  data = training[-outliers, ],
  method = "glm", 
  family = "binomial",
  trControl = ctrl)

glm_model2 # model performance
summary(glm_model2)



# Outliers and influential obs plots
plot(glm_model2$finalModel, which = 4, cook.levels = cutoff)
influencePlot(glm_model2$finalModel)
crPlots(glm_model2$finalModel)

# ad-hoc deletion of outliers and highly influential points using graphs
outliers <- c(1, 5, 9, 79, 93, 191, 207, 264, 334, 369, 387, 389, 398, 433, 489, 
  516, 526, 537, 567)

# make predictions +++++
glm2_preds <- predict(glm_model2, type = "prob", newdata = test)
glm2_preds <- data.frame(id, glm2_preds$Yes)
names(glm2_preds) <- c(" ", "Made Donation in March 2007")
write.csv(glm2_preds, file = "../submissions/submission.csv", row.names = FALSE) # output file


