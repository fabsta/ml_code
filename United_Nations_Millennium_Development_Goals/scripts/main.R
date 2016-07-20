rm(list = ls())

train <- read.csv("../data/TrainingSet.csv", stringsAsFactors = F)
out_file <- "../submissions/next_submissionR.csv"
sample <- read.csv("../data/SubmissionRows.csv")


for(i in 1:length(sample$id)){
#for(i in 1:2){
#curr_id <- sample$id[i]
  curr_id <- sample$id[i]
  #curr_id <- 25419
  cat("checking ",curr_id,"\n")
  example_row <- subset(train, ID == curr_id)
  no_of_columns <- dim(example_row)[1]
  
  example_data <- as.data.frame(t(example_row[,2:37]))
  last_x_years <- tail(example_data)
  non_na_values <- sum( !is.na( tail(example_data) ) ) 
  
  #example_data <- example_row[,2:37]
  if(non_na_values < 6) {
    prev_values  <- tail(example_data,n=2)
    val_2007 <- example_data["X2007",]
    val_2006 <- example_data["X2006",]
    slope <- val_2007
    if(!is.na(val_2006)){
      slope = val_2007 - val_2006
    }
    val_2008 = slope + val_2007 
    val_2012 = slope * 2.5 +  val_2007 
  }
  else{
    example_data <- as.data.frame(t(example_row[,2:37]))
    non_na_values <- sum( !is.na( example_data ) ) 
    #head(example_data)
    T_data = ts(example_data, frequency=1, start=c(1972), end=c(2007))
      #plot(T_data)
      # train = window(T_data,end=2004)
      # test = window(T_data,start=2005)
      # fit.holt=holt(train, h=5, initial="optimal")
      # summary(fit.holt)
      # plot(forecast(fit.holt))
      # lines(test, type="o")
      # 
      # on all data
    fit.holt=holt(T_data, h=5, initial="optimal")
    #summary(fit.holt)
    forecasts <- forecast(fit.holt)
    val_2008 = as.numeric(forecasts$mean)[1]
    val_2012 = as.numeric(forecasts$mean)[5]
  }
  cat("2008: ",val_2008," 2012: ",val_2012," \n")
  sample[i,2] = val_2008
  sample[i,3] = val_2012
  #plot(forecast(fit.holt))
  #lines(test, type="o")
  # put in time series
}

#write.table(sample, file = out_file,col.names=FALSE, row.names=FALSE,sep=",")
write.table(sample, file = out_file,row.names=FALSE, na="",sep=",",col.names=c("id","2008 [YR2008]","2012 [YR2012]"))
# get those whw


apply(train, 1,function(x) sum(is.na(x)))
