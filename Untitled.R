# 
#Question 1
#

df_model <- read.csv("housing2023.csv")

#1.1
data_fit <- lm(medv ~ ., df_model)


