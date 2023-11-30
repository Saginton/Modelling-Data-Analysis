# 
#Question 1
#
getwd()
setwd("/Users/king_saginton/Monash/FIT2086/Assignment3 ")
df_model <- read.csv("housing.2023.csv")

#1.1
data_fit <- lm(medv ~ ., df_model)
df_summary <- summary(data_fit)
print(df_summary)

#1.2
p_values <- coef(df_summary)[, "Pr(>|t|)"]
n <- length(p_values) - 1 
adjusted_alpha <- 0.05 / n
s_predictors <- names(p_values[p_values < adjusted_alpha])
print(s_predictors)

#1.4
full_model <- lm(medv ~ ., data = df_model)
stepwise_model <- step(full_model, direction="both", k = log(nrow(df_model)), trace=0)
summary(stepwise_model)

#1.7
data_fit_interaction <- lm(medv ~ chas + nox + rm*dis + ptratio + lstat, data = df_model)
summary(data_fit_interaction)

#Question 2
library(tree)
library(rpart)
df_train  <- read.csv("heart.train.2023.csv")
df_test <- read.csv("heart.test.2023.csv")
source("wrappers.R")
#2.1
df_tree <- rpart(HD ~ ., df_train)
df_cv <- learn.tree.cv(HD ~ ., data = df_train, nfolds = 10, m = 5000)
print(df_cv)

#2.2
plot(df_cv$best.tree, uniform = TRUE, margin = 0.1)
text(df_cv$best.tree, pretty = 12)

#2.3
yprob_values <- df_cv$best.tree$frame$yprob
heart_disease_probs <- yprob_values[df_cv$best.tree$frame$var == "<leaf>", 2]
terminal_nodes <- which(df_cv$best.tree$frame$var == "<leaf>")
heart_disease_probs <- as.numeric(yprob_values[df_cv$best.tree$frame$var == "<leaf>", 2])
plot(df_cv$best.tree, uniform=TRUE, margin=0.1)
text(df_cv$best.tree, pretty=12)

for (i in 1:length(terminal_nodes)) {
  node_id <- terminal_nodes[i]
  prob <- heart_disease_probs[i]
  nodelabel <- paste0("P(Y) = ", round(prob, 2))
  text(df_cv$best.tree, node=node_id, label=nodelabel, cex=0.7)
}

df_train$HD <- ifelse(df_train$HD == "N", 0, 1)
logistic_model <- glm(HD ~ ., data = df_train, family = binomial)
stepwise_model <- step(logistic_model, direction = "both", k = log(nrow(df_train)))
summary(stepwise_model)

#2.6

tree.probs <- predict(df_cv$best.tree, newdata = df_test, type = "prob")
logistic.probs <- predict(stepwise_model, newdata = df_test, type = "response")
df_test$HD <- factor(df_test$HD)

source("my.prediction.stats.R")  # Load the function if not already sourced.
tree_preds <- ifelse(tree.probs[, 2] > 0.5, 1, 0)
logistic_preds <- ifelse(logistic.probs > 0.5, 1, 0)


cat("Tree Model Statistics:\n")
my.pred.stats(tree_preds, df_test$HD)

cat("Stepwise Logistic Regression Model Statistics:\n")
my.pred.stats(logistic_preds, df_test$HD)

#2.8
P_tree <- tree.probs[69, 2] 
odds_tree <- P_tree / (1 - P_tree)
P_logistic <- logistic.probs[69]
odds_logistic <- P_logistic / (1 - P_logistic)

print(odds_tree)
print(odds_logistic)

#2.9
predict_69th <- function(data, indices) {
  sample_data <- data[indices, ]
  model <- glm(HD ~ I(CP == 1) + I(CP == 2) + I(CP == 3) + THALACH + OLDPEAK + CA + I(THAL == 1) + I(THAL == 2), 
               data = sample_data, family = binomial)
  return(stats::predict(model, newdata = df_test[69, ], type = "response")[1])
}
df_train$HD <- ifelse(df_train$HD == "Y", 1, 0) 
results <- boot(data = df_train, statistic = predict_69th, R = 5000)

ci <- boot.ci(results, type = "bca")
print(ci)

tree_prob_69th <- predict(df_cv$best.tree, newdata = df_test[69, ], type = "vector")[1]
logistic_prob_69th <- predict(logistic_model, newdata = df_test[69, ], type = "response")[1]

cat("Predicted probability from tree model for 69th patient:", tree_prob_69th, "\n")
cat("Predicted probability from logistic regression for 69th patient:", logistic_prob_69th, "\n")


#############
#Question 3
#############

#3.1
rm(list = ls())
library("boot")
df_measured <- read.csv("ms.measured.2023.csv")
df_truth <- read.csv("ms.truth.2023.csv")
library("kknn")
library(ggplot2)
mse_values <- numeric(25)
for (k in 1:25) {
  k_nn_result <- kknn(intensity ~ MZ, 
                      train = df_measured, 
                      test = df_truth, 
                      k = k, 
                      kernel = "optimal")
  
  predicted_intensities <- fitted(k_nn_result)
  mse <- mean((predicted_intensities - df_truth$intensity)^2)
  mse_values[k] <- mse
}
ggplot(data.frame(k = 1:25, MSE = mse_values), aes(x = k, y = MSE)) +
  geom_line() +
  geom_point() +
  labs(title = "Mean-Squared Error vs. k", x = "k (number of neighbors)", y = "Mean-Squared Error") +
  theme_minimal()

#3.2

plot_knn_estimate <- function(k) {
  k_nn_result <- kknn(intensity ~ MZ, 
                      train = df_measured, 
                      test = df_truth, 
                      k = k, 
                      kernel = "optimal")
  
  predicted_intensities <- fitted(k_nn_result)
  p <- ggplot() +
    geom_point(data = df_measured, aes(x = MZ, y = intensity), color = "blue", alpha = 0.5, size = 1) +
    geom_line(data = df_truth, aes(x = MZ, y = intensity), color = "green") +
    geom_line(data = data.frame(MZ = df_truth$MZ, Intensity_Predicted = predicted_intensities), 
              aes(x = MZ, y = Intensity_Predicted), color = "red") +
    labs(title = paste("k-NN Estimation for k =", k),
         x = "MZ",
         y = "Intensity") +
    theme_minimal() +
    scale_color_identity(name = "Legend",
                         breaks = c("blue", "green", "red"),
                         labels = c("Training Data", "True Spectrum", "Estimated Spectrum"),
                         guide = "legend")
  return(p)
}
k_values <- c(2, 5, 10, 25)
plots <- lapply(k_values, plot_knn_estimate)
library(gridExtra)
grid.arrange(grobs = plots, ncol = 2)

#3.5
cv_results <- train.kknn(intensity ~ ., data = df_measured, kmax = 50, kernel = "optimal") # replace "optimal_kernel_name" with your optimal kernel name
best_k <- cv_results$best.parameters$k
print(best_k)

#3.6
residuals <- df_measured$intensity - best_k
noise_std_dev <- sd(residuals)
print(noise_std_dev)

#3.7
max_index <- which.max(best_k)
max_MZ <- df_measured$MZ[max_index]
print(max_MZ)

#3.8

library(boot)
library(kknn)
bootstrap_knn <- function(data, indices, mz_value, k) {
  boot_data <- data[indices, ]
  test_data <- data.frame(MZ = mz_value, intensity = NA)
  model <- kknn(intensity ~ MZ, train = boot_data, test = test_data, k = k, kernel = "optimal")
  return(predict(model)[1])
}

get_CI <- function(data, mz_value, k, R = 5000) {
  boot_results <- boot(data = data, statistic = bootstrap_knn, R = R, mz_value = mz_value, k = k)
  return(boot.ci(boot_results, type = "perc")$percent[4:5])
}

MZ_MAX <- 7500

k_values <- c(6, 3, 20)
confidence_intervals <- matrix(0, 2, length(k_values))

for(i in 1:length(k_values)){
  confidence_intervals[,i] <- get_CI(df_measured, MZ_MAX, k_values[i])
}

print(confidence_intervals)


                   
                   