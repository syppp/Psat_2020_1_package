library(data.table)
library(plyr)
library(magrittr)
library(tidyverse)
library(MLmetrics)
library(rpart)
library(tree)
library(randomForest)

#ch0
#문제1
data<- fread("C:/Users/user1/Desktop/대학교/3학년 2학기/피셋/패키지/2주차 패키지/데이터1.csv",data.table=FALSE)
data$date <- ifelse(mday(data$event_datetime)==10, "test","train") 

train <- filter(data, date =="train") 
test <- filter(data, date=="test")

#문제2
train %<>% select(-date,-event_datetime)
test %<>% select(-date,-event_datetime)
#문제3
train[sapply(train, is.character)] %<>% lapply(as.factor)
test[sapply(test, is.character)] %<>% lapply(as.factor)

#ch1 
#문제1-1
data1 <- train  %>% select(-device_model,-predicted_house_price)
model_glm_1_1 <- glm(click~.,family=binomial(link="logit"),data=data1)
#문제1-2
data2 <- train  %>% select(-device_model)
model_glm_1_2 <- glm(click~.,family=binomial(link="logit"),data=data2)
#문제1-3
data3 <- train  %>% select(-predicted_house_price)
model_glm_1_3 <- glm(click~.,family=binomial(link="logit"),data=data3)
#문제1-4 고치기 
prediction <- predict(model_glm_1_1, newdata=test, type = "response")
LogLoss(prediction, test$click)

prediction <- predict(model_glm_1_2, newdata=test, type = "response")
LogLoss(prediction, test$click)

prediction <- predict(model_glm_1_3, newdata=test, type = "response")
LogLoss(prediction, test$click)

#문제2
train %<>% select(-device_model, -predicted_house_price)
test %<>% select(-device_model, -predicted_house_price)

#문제3-1
model_tree_1 <- rpart(click~., data=train)

#문제3-2
model_tree_2 <- tree(click~., data=train)
#문제4-1

train$click<-as.factor(train$click)
test$click<-as.factor(test$click)

#문제4-2

for (i in 1:length(train)){
  x<- levels(train[,i])
  a <- data.frame(x)
  x<- levels(test[,i])
  b<- data.frame(x)
  c <- unique(rbind(a,b))
  levels(test[,i]) <- c$x
  levels(train[,i]) <- c$x
}
#문제4-3
set.seed(1)
model_rf_1 <-randomForest(click ~., data=train, type="prob" )
#문제5
prediction <- predict(model_tree_1, test)
LogLoss(prediction, (as.numeric(test$click)-1))

prediction <- predict(model_tree_2, newdata=test)
LogLoss(prediction, (as.numeric(test$click)-1))

prediction <- predict(model_rf_1, newdata=test, type="prob")
LogLoss(prediction[,2], (as.numeric(test$click)-1))
#ch2
#문제1
library(caret)
set.seed(1)
n_split <-5
cv <- createFolds(train$click,k=n_split)
tune_parameter <- expand.grid(k=c(3,4,5))
tune_parameter$logloss <- NA

for(k in 1:NROW(tune_parameter)){
  logloss_result <- c()
  for( i in 1:n_split){
    idx <- cv[[i]]
    
    train_x<- train[-idx,]
    val_x <- train[idx,]
 
    set.seed(1) 
    model_rf_2 <-randomForest(click ~., data=train_x,mtry=tune_parameter[k,'k'])
    prediction <- predict(model_rf_2, newdata=val_x,type="prob") 
    
    logloss_temp <- LogLoss(prediction[,2],as.numeric(val_x$click)-1)
    logloss_result <- c(logloss_temp, logloss_result)
    result <- mean(logloss_result)
    
  }
  tune_parameter[k,'logloss'] <- result
}
tune_parameter
#문제2
prediction <- predict(model_rf_2, newdata=test, mtry=3, type= "prob")
LogLoss(prediction[,2],(as.numeric(test$click)-1))