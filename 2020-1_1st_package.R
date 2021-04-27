
library(data.table)
library(plyr)
library(magrittr)
library(dplyr)
library(tidyverse)
library(stringr)
library(lubridate)
library(ggplot2)

data <- fread("C:/Users/user1/Desktop/???б?/3?г? 2?б?/?Ǽ?/??Ű??/1???? ??Ű??/??????0.csv", stringsAsFactors=FALSE,data.table=FALSE)


data %<>% select(-Petal.Length) %>% filter(data$Petal.Width<=1.800)

data %<>%  mutate(Sepal_mean= mean(data$Sepal.Length)) %>%  mutate(Sepal_sd= sqrt(var(data$Sepal.Length))) 

data %<>% group_by(Species) %>% mutate(Sepal.Width_mean_each=mean(Sepal.Width))
data %<>% group_by(Species) %>% mutate(Sepal.num=n())
data %<>% ungroup() 

data <- arrange(data,desc(Sepal.Length)) 
data$Species %<>% revalue(c("virginica"="VI","versicolor" = "VE","setosa" = "SE"))
data %<>% rename("name"="Species")


#Ch.1 

data2<- fread("C:/Users/user1/Desktop/???б?/3?г? 2?б?/?Ǽ?/??Ű??/1???? ??Ű??/??????1.csv", stringsAsFactors=FALSE,data.table=FALSE)
str(data2) 
summarise(data2)

col <- length(data2)
for(i in 0:col){
  print(length(unique(data2[,i])) )}
  

data2<- select(data2,click,placement_type,event_datetime,age,gender,install_pack,marry,predicted_house_price)

data2$age<-as.factor(data2$age)

data2$weekend <- ifelse(wday(data2$event_datetime)==1|wday(data2$event_datetime)==6 , "yes","no") 
data2$day <- factor(wday(data2$event_datetime)) 
data2$time <- factor(hour(data2$event_datetime)) 
data2$date <- factor(mday(data2$event_datetime))

data2 %<>% group_by(date,time)  %>% mutate(click_mean=mean(click)) 

data2$number_install<- str_count(data2$install_pack,",")


data2[sapply(data2, is.character)] %<>% lapply(as.factor)

#Ch.2 

ggplot(data2,aes(time,click_mean,group=date))+geom_line(aes(color=date))+theme_bw()

ggplot(data2,aes(time,click_mean,group=date))+geom_line(aes(color=date))+facet_wrap(~ date,ncol=4)+theme_bw()

ggplot(data=data2) + geom_bar(mapping = aes(x=age, fill=placement_type),position="fill")
data2 %<>% group_by(age,placement_type) %>% mutate(num = n())
ggplot(data2,aes(x=age,y=placement_type,size=num))+geom_point(shape=21, colour="black",fill="skyblue")+scale_size_area(max_size=15)+theme(panel.background = element_blank())


data1<- data2 %>% gather(key,value,number_install,predicted_house_price,click_mean)

ggplot(data1, aes(sample=value,shape=weekend, colour=weekend))+stat_qq()+facet_wrap(.~key, scales="free_y")  

ggplot(data1, aes(x=weekend, y=value)) + geom_violin(aes(fill=weekend)) + geom_boxplot(width=0.3) + stat_summary(fun.y = "mean", geom = "point", shape = 5) + facet_wrap(.~key, scales = "free_y")


