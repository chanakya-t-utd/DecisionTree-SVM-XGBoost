#call the libraries
library(xgboost)
library(e1071)
library(caret)

#Load the data
setwd("/000_UTD/Sem_3/3_MachineLearning/Project_2/Contraceptive")
df<-read.csv("cmcdata.csv",header = TRUE)

#Change the Class variable to numeric and tag them from 0 to 6
fact<-as.factor(c("1","2","3"))
vals<-c(0,1,2)

#our vlookup function:
vlookup<-function(fact,vals,x) {
  #probably should do an error checking to make sure fact 
  #   and vals are the same length
  
  out<-rep(vals[1],length(x)) 
  for (i in 1:length(x)) {
    out[i]<-vals[levels(fact)==x[i]]
  }
  return(out)
}

df$CLASS<-vlookup(fact,vals,df$conmeth)

#Split the data into train and test (70-30)
set.seed(2222)
trainid = sample(1:nrow(df),nrow(df)/1.428)
testid = -trainid
train = df[trainid,]
test = df[testid,]
trainMatrix <- as.matrix(train) #convert into matrix
testMatrix <- as.matrix(test)
mode(trainMatrix) = "numeric"  #change the mode of matrix into numeric
y<-trainMatrix[,11] #is the target variable, extracted out as it needs to be passed as parameter into the xgboost function.
y <- as.matrix(as.integer(y)) #change the label variable into  matrix
param <- list("objective" = "multi:softprob",   
              "num_class" = 3 ,    
              "eval_metric" = "merror",   
              "nthread" = 8,  
              "max_depth" =5,   
              "eta" = 0.3,    
              "gamma" = 0,    
              "subsample" = 1,   
              "colsample_bytree" = 1  
)

set.seed(42)  # set random seed to make model reproducible.
bst.cv <- xgb.cv(param=param, data=trainMatrix, label=y, nfold=4, nrounds=50, prediction=TRUE, verbose=TRUE)
#Confusion Matrix for the model:
pred.cv = matrix(bst.cv$pred, nrow=length(bst.cv$pred)/3, ncol=3) #make a 3*3 matrix of probabilities for each  cluster.
pred.cv = max.col(pred.cv, "last")       #pick the cluster with the maximum probability.
confusionMatrix(factor(y+1), factor(pred.cv))   #build confusion matrix for the predicted cluster
table(pred.cv,y)









