#########Loan_pre_using_caret####################
rm(list=ls())
ls()
data=read.csv("ltrain.csv",header=T,stringsAsFactors = T)
#edit(data)
sum(is.na(data)) 
###removing NA values
str(data)
library(caret)
preprocvalues= preProcess(data,method=c("knnImpute","center","scale"))
library(RANN)
data_processed=predict(preprocvalues,data)
sum(is.na(data_processed))
###########################################################################
data_processed$Loan_Status<-ifelse(data_processed$Loan_Status=='N',1,2)
id=data_processed$Loan_ID
data_processed$Loan_ID<-NULL
str(data_processed)
dv=dummyVars("~.",data_processed,fullRank =T)
data_trans=data.frame(predict(dv,data_processed))
str(data_trans)
data_trans$Loan_Status=as.factor(data_trans$Loan_Status)

################################################################################
# Splitting Data
index=createDataPartition(data_trans$Loan_Status,p=0.75,list=F)
train=data_trans[index,]
test=data_trans[-index,]
str(train)
str(test)
########################################################################
#############Feature Selection
control <- rfeControl(functions = rfFuncs,
                      method = "repeatedcv",
                      repeats = 3,
                      verbose = FALSE)
outcomename<-'Loan_Status'
predictors<-names(train)[!names(train) %in% outcomename]
Loan_Pred_Profile <- rfe(train[,predictors], train[,outcomename], rfeControl = control)
Loan_Pred_Profile
predictors=c('Credit_History', 'ApplicantIncome', 'Property_Area.Semiurban', 
             'CoapplicantIncome', 'Loan_Amount_Term')
###############################################################################
###Training models using Caret
model_rpart=train(train[,predictors],train[,outcomename],method='rpart')
model_rf=train(train[,predictors],train[,outcomename],method='rf')
model_gbm=train(train[,predictors],train[,outcomename],method='gbm')
model_glm=train(train[,predictors],train[,outcomename],method='glm')

################Parameter tuning using Caret#####################################
fitcontrol=trainControl(method='repeatedcv',number=5,repeats=5)
modelLookup(model='rpart')
model_rpart<-train(train[,predictors],train[,outcomename],method='rpart',trControl=fitcontrol,tuneLength=10)
plot(model_rpart)

#################################################################################
##Variable importance estimation using caret
varImp(object=model_rpart)
plot(varImp(object=model_rpart),main='decision_tree')
varImp(object=model_rf)
plot(varImp(object=model_rf),main='random_forest')
###########################################################################
######################## prediction ########################################
prediction=predict.train(object=model_rpart,test[,predictors],type='raw')
table(prediction)
confusionMatrix(prediction,test[,outcomename])
#############  Decision Tree:  Accuracy=81.05%  ##############################
prediction=predict.train(object=model_rf,test[,predictors],type='raw')
table(prediction)
confusionMatrix(prediction,test[,outcomename])
#############  Random Forest:   Accuracy=79.74%  ##############################
prediction=predict.train(object=model_gbm,test[,predictors],type='raw')
table(prediction)
confusionMatrix(prediction,test[,outcomename])
###################################################################################
prediction=predict.train(object=model_glm,test[,predictors],type='raw')
table(prediction)
confusionMatrix(prediction,test[,outcomename])
#####################################################################################
