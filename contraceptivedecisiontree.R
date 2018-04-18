
#call the libraries
library(rpart)
library(tree)
library(rpart.plot)
library(caret)


#Load the data
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
classs = df$conmeth
#Split the data into train and test (70-30)
set.seed(2222)
trainid = sample(1:nrow(df),nrow(df)/1.428)
testid = -trainid
train = df[trainid,]
test = df[testid,]

#This object is required to calculate the misclassification
test_Class = classs[testid]

#Training the Decision Tree classifier with criterion as information gain
dt<-rpart(conmeth~.,data = train,method ='class', parms = list(split = "information"), control = rpart.control(minsplit = 35, cp = 0.0015))
plot(dt)
text(dt, pretty = 0)
#check how the model is doing using the test data
tree_pred = predict(dt, test, type = 'class')
mean(tree_pred != test_Class) # the misclassification is 42.76%
#Confusion Matrix
confusionMatrix(tree_pred, test_Class)

##Pruning to the tree for the best cp
plotcp(dt)
printcp(dt)

pruned_tree = prune(dt,cp = 0.011)
tree_pred_pruned = predict(pruned_tree, test, type = 'class')
mean(tree_pred_pruned != test_Class) # The new misclassification is 38.91%
#Confusion Matrix
confusionMatrix(tree_pred_pruned, test_Class)


#Training the Decision Tree classifier with criterion as Gini Index
dt_gini<-rpart(conmeth~.,data = train,method ='class', parms = list(split = "gini"), control = rpart.control(minsplit = 35, cp = 0.0015))
plot(dt_gini)
text(dt, pretty = 0)
#check how the model is doing using the test data
tree_pred = predict(dt_gini, test, type = 'class')
mean(tree_pred != test_Class) # the misclassification is 45.02%
#Confusion Matrix
confusionMatrix(tree_pred, test_Class)

##Pruning to the tree for the best cp
plotcp(dt_gini)
printcp(dt_gini)

pruned_tree = prune(dt_gini,cp = 0.016)
tree_pred_pruned = predict(pruned_tree, test, type = 'class')
mean(tree_pred_pruned != test_Class) #39.1% - 0.016
#Confusion Matrix
confusionMatrix(tree_pred_pruned, test_Class)

##No need to prune the tree further.
### For this data set Gini Index infomration gain split has given the best results.

