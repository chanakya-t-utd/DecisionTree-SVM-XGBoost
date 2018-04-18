#Call required packages
library(e1071)
library(caret)
#Laod data
setwd("/000_UTD/Sem_3/3_MachineLearning/Project_2/ImageSegmentation")
df<-read.csv("segmentationtestdata.csv",header = TRUE)

#Split the data into test and train (30-70)
set.seed(2222)
trainid = sample(1:nrow(df),nrow(df)/1.428)
testid = -trainid
train = df[trainid,]
test = df[testid,]

#Train the SVM model using Sigmoid Kernel
model_sigmoid <- svm(CLASS~ 
               REGION.CENTROID.COL + REGION.CENTROID.ROW 
             + SHORT.LINE.DENSITY.5 + SHORT.LINE.DENSITY.2 + VEDGE.MEAN + VEDGE.SD 
             + HEDGE.MEAN + HEDGE.SD + INTENSITY.MEAN + RAWBLUE.MEAN + RAWBLUE.MEAN 
             + RAWGREEN.MEAN + EXRED.MEAN + EXBLUE.MEAN + EXGREEN.MEAN + VALUE.MEAN 
             + SATURATION.MEAN + HUE.MEAN, 
             data=train, method="C-classification", kernel="sigmoid", 
             probability=T, gamma=0.0002, cost=100000)
prediction <- predict(model_sigmoid, test)
dtt<-table(test$CLASS, prediction)
sum(diag(dtt))/sum(dtt)  ## 94.66% Accuracy
confusionMatrix(test$CLASS, prediction)

#Train the SVM model using Radial Kernel
model_radial <- svm(CLASS~ 
               REGION.CENTROID.COL + REGION.CENTROID.ROW 
             + SHORT.LINE.DENSITY.5 + SHORT.LINE.DENSITY.2 + VEDGE.MEAN + VEDGE.SD 
             + HEDGE.MEAN + HEDGE.SD + INTENSITY.MEAN + RAWBLUE.MEAN + RAWBLUE.MEAN 
             + RAWGREEN.MEAN + EXRED.MEAN + EXBLUE.MEAN + EXGREEN.MEAN + VALUE.MEAN 
             + SATURATION.MEAN + HUE.MEAN, 
             data=train, method="C-classification", kernel="radial", 
             probability=T, gamma=0.0002, cost=100000)
prediction <- predict(model_radial, test)
dtt<-table(test$CLASS, prediction)
sum(diag(dtt))/sum(dtt)  ## 95.67% Accuracy
confusionMatrix(test$CLASS, prediction)

#Train the SVM model using Linear Kernel
model_linear <- svm(CLASS~ 
                      REGION.CENTROID.COL + REGION.CENTROID.ROW 
                    + SHORT.LINE.DENSITY.5 + SHORT.LINE.DENSITY.2 + VEDGE.MEAN + VEDGE.SD 
                    + HEDGE.MEAN + HEDGE.SD + INTENSITY.MEAN + RAWBLUE.MEAN + RAWBLUE.MEAN 
                    + RAWGREEN.MEAN + EXRED.MEAN + EXBLUE.MEAN + EXGREEN.MEAN + VALUE.MEAN 
                    + SATURATION.MEAN + HUE.MEAN, 
                    data=train, method="C-classification", kernel="linear", 
                    probability=T, gamma=0.0002, cost=100000)
prediction <- predict(model_linear, test)
dtt<-table(test$CLASS, prediction)
sum(diag(dtt))/sum(dtt) #89.89%
confusionMatrix(test$CLASS, prediction)

#Train the SVM model using Polynomial Kernel
model_polynomial <- svm(CLASS~ 
                      REGION.CENTROID.COL + REGION.CENTROID.ROW 
                    + SHORT.LINE.DENSITY.5 + SHORT.LINE.DENSITY.2 + VEDGE.MEAN + VEDGE.SD 
                    + HEDGE.MEAN + HEDGE.SD + INTENSITY.MEAN + RAWBLUE.MEAN + RAWBLUE.MEAN 
                    + RAWGREEN.MEAN + EXRED.MEAN + EXBLUE.MEAN + EXGREEN.MEAN + VALUE.MEAN 
                    + SATURATION.MEAN + HUE.MEAN, 
                    data=train, method="C-classification", kernel="polynomial", 
                    probability=T, gamma=0.0002, cost=100000)
prediction <- predict(model_polynomial, test)
dtt<-table(test$CLASS, prediction)
sum(diag(dtt))/sum(dtt) #38.09% Accuracy
confusionMatrix(test$CLASS, prediction)

#Therefore radial kernel has the highest accuracy for this dataset.

