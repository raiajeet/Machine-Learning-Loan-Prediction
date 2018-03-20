#########Loan_pre_using_caret####################
rm(list=ls())
ls()
data=read.csv("loan_pre_data.csv",header=T,stringsAsFactors = T)
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

################################ GLM ##############################
trctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3)
set.seed(14)
glm<- train(Loan_Status ~., data = train, method = "glm",
            trControl=trctrl,
            tuneLength = 10)          
glm
pre=predict(glm,test,type="prob")
pre
confusionMatrix(table(predict(glm,test),test$Loan_Status))   ###   Accuracy : 0.817  
library(pROC)
x=auc(test$Loan_Status,pre[,2])
x                                #Area under the curve: 0.7581
plot(roc(test$Loan_Status,pre[,2]))
     
###################################################################################

##########################################Decision Tree###########################################
trctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3)
set.seed(12)
library(rpart)
rpart<- train(Loan_Status ~., data = train, method = "rpart",
                    trControl=trctrl,
                    preProcess = c("center", "scale"),
                    tuneLength = 10)
rpart
pre=predict(rpart,test,type="prob")
pre
confusionMatrix(table(predict(rpart,test),test$Loan_Status))  #Accuracy : 0.8301  
library(pROC)
x=auc(test$Loan_Status,pre[,2])
x                               #Area under the curve: 0.7405
plot(roc(test$Loan_Status,pre[,2]))
####################################################################################
