# Log-loss function
#
# logarithmic-loss penalises overly confident incorrect predictions
#
#

logloss <- function(obs, y_hat) { # function for two vectors y and y_hat
  
  # error messages
  if (!is.numeric(y_hat))
    stop("y_hat must be numeric")
  if (length(y) != length(y_hat))
    stop("arguments not of equal length")
  if (is.factor(y) && length(levels(y)) > 2)
    stop("function defined for binary response")
  
  if (is.factor(y))
    y <- (as.numeric(y)-1)
  n <- length(y)
  log_loss <- (-1/n)*(sum(y*log(y_hat)+(1-y)*log(1-y_hat)))
  log_loss
  
}

# e.g. application
predictions <- runif(576, min = 0, max = 1)  # random predictions
logloss(y_hat = predictions, y = training$March.2007)


# Log-loss function for use in caret:::train =================================
# specific to current competition, includes kappa and accuracy

logloss_caret <- function (data, lev = NULL, model = NULL) {
  
  if (is.factor(data$obs))
    y <- (as.numeric(data$obs)-1)
  y_hat <- data$Yes
  
  # error messages
  if (!is.numeric(y_hat))
    stop("y_hat must be numeric")
  if (length(y) != length(y_hat))
    stop("arguments not of equal length")
  if (is.factor(y) && length(levels(y)) > 2)
    stop("function defined for binary response")
  
  n <- length(y)
  log_loss <- (-1/n)*(sum(y*log(y_hat)+(1-y)*log(1-y_hat)))
  
  accuracy <- sum(data$pred == data$obs)/n
  
  #expected <- 
  #kappa <- accuracy - 
  
  out <- c(log_loss, accuracy)
    
  names(out) <- c("LogLoss", "Accuracy")
  out
  
}

