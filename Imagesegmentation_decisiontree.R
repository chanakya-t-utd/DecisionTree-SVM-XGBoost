

#call the libraries
library(rpart)
library(tree)
library(rpart.plot)
library(caret)


#Load the data
setwd("/000_UTD/Sem_3/3_MachineLearning/Project_2/ImageSegmentation")
df<-read.csv("segmentationtestdata.csv",header = TRUE)
classs = CLASS
#Split the data into train and test (70-30)
set.seed(2222)
trainid = sample(1:nrow(df),nrow(df)/1.428)
testid = -trainid
train = df[trainid,]
test = df[testid,]

#This object is required to calculate the misclassification
test_Class = classs[testid]

#Training the Decision Tree classifier with criterion as information gain
dt<-rpart(CLASS~.,data = train,method ='class', parms = list(split = "information"), control = rpart.control(minsplit = 35, cp = 0.0015))
plot(dt)
text(dt, pretty = 0)
#check how the model is doing using the test data
tree_pred = predict(dt, test, type = 'class')
mean(tree_pred != test_Class) # the misclassification is 6.6%
#Confusion Matrix
table(tree_pred, test_Class)
confusionMatrix(tree_pred,test_Class)



##Pruning to the tree for the best cp
plotcp(dt)
printcp(dt)

pruned_tree = prune(dt,cp = 0.00015)
tree_pred_pruned = predict(pruned_tree, test, type = 'class')
mean(tree_pred_pruned != test_Class) #6.6%
#Confusion Matrix
confusionMatrix(tree_pred_pruned, test_Class)


###There is no need fr pruning as the model is not overfitting.


#Training the Decision Tree classifier with criterion as Gini Index
dt_gini<-rpart(CLASS~.,data = train,method ='class', parms = list(split = "gini"), control = rpart.control(minsplit = 35, cp = 0.0015))
plot(dt_gini)
text(dt, pretty = 0)
#check how the model is doing using the test data
tree_pred = predict(dt_gini, test, type = 'class')
mean(tree_pred != test_Class) # the misclassification is 7.93%
#Confusion Matrix
confusionMatrix(tree_pred, test_Class)

##Pruning to the tree for the best cp
plotcp(dt_gini)
printcp(dt_gini)

pruned_tree = prune(dt_gini,cp = 0.00015)
tree_pred_pruned = predict(pruned_tree, test, type = 'class')
mean(tree_pred_pruned != test_Class) #7.93%
#Confusion Matrix
table(tree_pred_pruned, test_Class)

##No need to prune the tree further.
## conlcusion :  information gain is a better splitting method.
##This is because the no of elements under each node needed weightage while allocating the entropy for the node. Thereby choosing the best split to train the model this way has provided better accuracy over the gini index method.

 
  