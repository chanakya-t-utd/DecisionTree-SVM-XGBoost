#Call required packages
library(caret)
library(e1071)
#Laod data
setwd("/000_UTD/Sem_3/3_MachineLearning/Project_2/Contraceptive")
df<-read.csv("cmcdata.csv",header = TRUE)
df$wifeedu <- as.factor(df$wifeedu)
df$husedu <- as.factor(df$husedu)
df$wifereg <- as.factor(df$wifereg)
df$wifenotworking <- as.factor(df$wifenotworking)
df$husocc <- as.factor(df$husocc)
df$solindex <- as.factor(df$solindex)
df$medexp <- as.factor(df$medexp)
df$conmeth <- as.factor(df$conmeth)

#Split the data into train and test (70-30)
set.seed(2222)
trainid = sample(1:nrow(df),nrow(df)/1.428)
testid = -trainid
train = df[trainid,]
test = df[testid,]
str(df)

#Train the SVM model using Sigmoid Kernel
model_sigmoid <- svm(conmeth~ 
                       wifeage + wifeedu + husedu + noofchild + wifereg + wifenotworking + husocc + solindex + medexp, 
                     data=train, method="C-classification", kernel="sigmoid", 
                     probability=T, gamma=0.0002, cost=100000)
prediction <- predict(model_sigmoid, test)
dtt<-table(test$conmeth, prediction)
sum(diag(dtt))/sum(dtt)  ## 50.45% Accuracy
confusionMatrix(test$conmet, prediction)

#Train the SVM model using Radial Kernel
model_radial <- svm(conmeth~ 
                       wifeage + wifeedu + husedu + noofchild + wifereg + wifenotworking + husocc + solindex + medexp, 
                     data=train, method="C-classification", kernel="radial", 
                     probability=T, gamma=0.0002, cost=100000)
prediction <- predict(model_radial, test)
dtt<-table(test$conmeth, prediction)
sum(diag(dtt))/sum(dtt)  ## 57.46% Accuracy
confusionMatrix(test$conmet, prediction)

#Train the SVM model using Linear Kernel
model_linear <- svm(conmeth~ 
                      wifeage + wifeedu + husedu + noofchild + wifereg + wifenotworking + husocc + solindex + medexp, 
                    data=train, method="C-classification", kernel="linear", 
                    probability=T, gamma=0.0002, cost=100000)
prediction <- predict(model_linear, test)
dtt<-table(test$conmeth, prediction)
sum(diag(dtt))/sum(dtt) #41.62%
confusionMatrix(test$conmet, prediction)

#Train the SVM model using Polynomial Kernel
model_polynomial <- svm(conmeth~ 
                      wifeage + wifeedu + husedu + noofchild + wifereg + wifenotworking + husocc + solindex + medexp, 
                    data=train, method="C-classification", kernel="polynomial", 
                    probability=T, gamma=0.0002, cost=100000)
prediction <- predict(model_polynomial, test)
dtt<-table(test$conmeth, prediction)
sum(diag(dtt))/sum(dtt) #44.34% Accuracy
confusionMatrix(test$conmet, prediction)

#Therefore radial kernel has the highest accuracy for this dataset.

