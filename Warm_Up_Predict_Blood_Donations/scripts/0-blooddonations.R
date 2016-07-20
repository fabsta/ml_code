# DrivenData Blood Donations Warm-up Comp
#
# AIM: predict whether someone made a blood donation in March 2007 based on past
# donation history
#
#
#
# Load data and packages ======================================================
training <- read.csv(file="../data/train.csv", 
                     header = TRUE, sep = ",")
head(training)
str(training)

test <- read.csv(file="../data/test.csv", 
                 header = TRUE, sep = ",")
head(test)
str(test)

# packages
library(ggplot2)
library(caret)

# Reformat data set names =====================================================
# make response variable into a factor
training$Made.Donation.in.March.2007 <- factor(training$Made.Donation.in.March.2007,
  labels = c("No", "Yes"))

# remove X (id)
training <- training[ ,-1]
id <- test[ ,1]
test <- test[ ,-1]

# Shorten variable names
names(training) <- c("Last.Donation", "Num.Donations", "Volume.Donated", 
  "First.Donation", "March.2007")
names(test) <- c("Last.Donation", "Num.Donations", "Volume.Donated", 
  "First.Donation")

# Log-loss function ===========================================================
logloss <- function(y, y_hat) { # function for two vectors y and y_hat
  
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


# missingness & near zero variance ============================================
summary(sapply(training, is.na)) # any missing values? nope
nearZeroVar(training)


# Visual exploration ==========================================================
# Number of Donations and Volume Donated --------------------------------------
# BOXPLOTs of Number.of.Donations by Made.Donation.in.March.2007
qplot(x = March.2007, y = Num.Donations,
  data = training, geom = c("boxplot", "jitter"))

# BOXPLOTs of Total.Volume.Donated by Made.Donation.in.March.2007
qplot(x = March.2007, y =Volume.Donated,
  data = training, geom = c("boxplot", "jitter")) 
# both are similar!!

# ...because all donations are 250cc
training$Volume.Donated/training$Num.Donations
# remove Total.Volume.Donated
training <- training[ ,-3]
test <- test[ ,-3]

# ok lets make more features
training <- transform(training, avg_times = First.Donation / Num.Donations)
training <- transform(training, overdue = Last.Donation -  avg_times)

test <- transform(training, avg_times = First.Donation / Num.Donations)
test <- transform(training, overdue = Last.Donation -  avg_times)

# HISTOGRAM of Number.of.Donations facet wrapped (by response) ----------------
# 1. Create a ggplot2 object 
gg_donations <- ggplot(data = training, aes(x = Num.Donations, 
  fill = March.2007))
# 2. Histogram
gg_donations + geom_histogram(col = "black") + 
  facet_wrap(~ March.2007, nrow =2)
# 3. Density plot
gg_donations + geom_density(col = "black") + 
  facet_wrap(~ March.2007, nrow =2)

BoxCoxTrans(training$Num.Donations) # transform?
qplot(x = log(Num.Donations), data = training, geom = "histogram")

# The density plot reveals a slight tendency for those high in Num.Donations
# to have donated in March 2007

# Months since first donation -------------------------------------------------
# HISTOGRAM of First.Donation facet wrapped (by response)
# 1. Create a ggplot2 object
gg_first <- ggplot(data = training, aes(x = First.Donation,
  fill = March.2007))
# 2. Histogram
gg_first + geom_histogram(binwidth = 6, col = "black") + 
  facet_wrap(~ March.2007, nrow =2)
# 3. Density
gg_first + geom_density(col = "black") + 
  facet_wrap(~ March.2007, nrow =2)

# Overlap between the groups

# Months since last donation -------------------------------------------------
# HISTOGRAM of Months.since.Last.Donation facet wrapped (by response)
# 1. Create ggplot2 object
gg_last <- ggplot(data = training, aes(x = Last.Donation,
  fill = March.2007))
# 2. Histogram
gg_last + geom_histogram(col = "black") + 
  facet_wrap(~ March.2007, nrow =2)
# 3. Density
gg_last + geom_density(col = "black") + 
  facet_wrap(~ March.2007, nrow =2)

# Clear tendency of those who donated in March 2007 to have recently donated, 
# much more variability in those who didn't

# Bivariate Plots -------------------------------------------------------------
# Explore relationships between predictor variables:
# Months.since.First.Donation, Months.since.Last.Donation +++++++++++++++++++++
# 1. Create ggplot object
gg_bivar1 <- ggplot(data = training, aes(x = First.Donation, 
  y = Last.Donation, colour = March.2007))
# 2a. SCATTER plot w/ smoother (loess)
gg_bivar1 + geom_point() + stat_smooth() # cleaner code than above
# 2b. SCATTER plot w/ smoother (lm)
gg_bivar1 + geom_point() + stat_smooth(method = lm)

# No obvious relationship between continuous variables. Those who donated in 
# March 2007 tend to have donated more recently as evident in lower Last.Donation
# values

# Months.since.First.Donation, Number.of.Donations ++++++++++++++++++++++++++++
# 1. Create ggplot object
gg_bivar2 <- ggplot(data = training, aes(x = First.Donation,
  y = log(Num.Donations), colour = March.2007))
# 2a. SCATTER plot w/ smoother (loess)
gg_bivar2 + geom_point() + stat_smooth() # cleaner code than above
# 2b. SCATTER plot w/ smoother (lm)
gg_bivar2 + geom_point() + stat_smooth(method = lm)
# Correlation
cor.test(training$Num.Donations, training$First.Donation)

# Positive correlation between the variables (r = 0.62). Tendency for those 
# who donated in March 2007 to be slightly higher in both variables.
# Relationship is slightly non-linear, though linear fit is not bad. 

# Months.since.Last.Donation, Number.of.Donations +++++++++++++++++++++++++++++
# 1. Create ggplot object
gg_bivar3 <- ggplot(data = training, aes(x = log(Num.Donations),
  y = Last.Donation, colour = March.2007))
# 2a. SCATTER plot w/ smoother (loess)
gg_bivar3 + geom_point() + stat_smooth() # cleaner code than above
# 2b. SCATTER plot w/ smoother (lm)
gg_bivar3 + geom_point() + stat_smooth(method = lm)

# No strong relationship between the variables, similar patterns as before with
# crowding in the bottom right of those who donated in March 2007, i.e. high in
# Num.Donations and low in months since last donation.


# Response frequencies ===========================================
response_freq <- table(training$March.2007)
prop.table(response_freq)

