             ## Loan Prediction ##
train=read.csv("ltrain.csv",header=TRUE)
test=read.csv("ltest.csv",header=TRUE)
#edit(train)
test$Loan_Status<-NA
library(rpart)
m=rpart(Loan_Status~.
        ,data=train,method="class",control=rpart.control(minsplit=20,
         minbucket=3,maxdepth=10,xval=10,usesurrogate = 2))
library(rattle)
library(rpart.plot)
library(RColorBrewer)
fancyRpartPlot(m)

t=table(train$Loan_Status,predict(pruned,type="class"))
prop.table(table(train$Loan_Status,predict(m,test,type="class")))
rownames(t)=paste("Actual",rownames(t),sep=":")
colnames(t)=paste("predicted",colnames(t),sep=":")
t
prop.table(t)
accuracy=sum(diag(t))/sum(t)###on traning data
accuracy

pre=predict(m,test,type="class")
submit <- data.frame(Loan_ID = test$Loan_ID, Loan_Status = pre)
write.csv(submit, file = "sample_submission.csv", row.names = FALSE)

