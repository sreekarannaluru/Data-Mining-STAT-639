######################### Part-1 : Classification ##########################################
library(ISLR)
# plotting predictors
install.packages("psych")
library(psych)
pairs.panels(x[414:419])
#All the predictors are normally distributed. LDA/QDA might be a better bet!
dim(x)

############################################################################################
## Note: PCA might be missing to capture some of the original data. Hence, I wanna try Lasso.
library(glmnet)
df <- cbind(x,y)
#Converting into matrix format
x=model.matrix(y ~ . x , df , na.action = "na.pass")[,-1]
y=y
#Scaling the parameters
x=scale(x)

par(mfrow = c(1,1))
fit.lasso=glmnet(x,y)
plot(fit.lasso,xvar="lambda", ylab = "Lasso Coefficients")
cv.lasso=cv.glmnet(x,y)
plot(cv.lasso, ylab = "Test MSE")
lasso.best.lambda=cv.lasso$lambda.min # find the best lambda value corresponding to min cv.error
log(lasso.best.lambda) # best lambda is 0.00234 = 10^(-2.793465)
min(cv.lasso$cvm) # min cv.error
non_zero_coef <- predict(fit.lasso, type = "coefficients", s = 0.00234)

# count number of non-zero coefficients
n_nonzero <- sum(non_zero_coef[-1] != 0)
# print result
cat("Number of non-zero coefficients:", n_nonzero)
# 350 non zero coefficients at minimum lambda after cross validation!
#Let's create a subset of x with the variables having non-zero at this lambda
# get indices of non-zero coefficients
idx_nonzero <- which(non_zero_coef[-1] != 0)
# subset data matrix to include only variables with non-zero coefficients
x_subset <- x[, idx_nonzero]
dim(x_subset) # (400, 350)

########################################################################
## Now I can perform PCA to reduce noise and capture maximum variability. 
pr.out <- prcomp(x_subset, scale = TRUE)
pr.var <- pr.out$sdev^2
pve <- pr.var/sum(pr.var)
pve.cumsum <- cumsum(pve)
plot(pve)
plot(pve.cumsum)
#The plot shows the variability saturates 
library(plotrix)
draw.circle(200,0.92,8)
# 92% of variability explained by 200 variables. 

# Select the first two principal components
PC_scores <- pr.out$x[, 1:200]
# Save the loadings and mean
loadings <- pr.out$rotation[, 1:200]
mean <- pr.out$center[1:200]
# Perform the inverse transformation
class_data <- PC_scores %*% t(loadings) + mean
dim(class_data)
# I believe class_data is a more refined with reduced dimension and I hope it gives good results!

############################################
## Checking variables with high collinearity
cor_matrix <- cor(class_data)
cor_matrix <- cor_matrix - diag(nrow(cor_matrix))
# Set a threshold for high collinearity
threshold <- 0.4
# Find variable pairs with correlation coefficient above the threshold
high_corr_pairs <- which(abs(cor_matrix) > threshold & !is.na(cor_matrix), arr.ind = TRUE)
# Print the pairs
high_corr_pairs
# (V168, V234), (V282, V234), (V341, V234), and (V341, V330) are highly collinear.
# 162 and 241 are the indices for columns V234 and V341 respectively
class_data <- class_data[, -c(162, 241)]
#cross check
cor_matrix <- cor(class_data)
cor_matrix <- cor_matrix - diag(nrow(cor_matrix))
# Set a threshold for high collinearity
threshold <- 0.4
# Find variable pairs with correlation coefficient above the threshold
high_corr_pairs <- which(abs(cor_matrix) > threshold & !is.na(cor_matrix), arr.ind = TRUE)
# Print the pairs
high_corr_pairs
dim(class_data)
#No variables are highly collinear (greater than 0.4)!!!

###############################
## Linear Discriminant Analysis
library(MASS)
#test and train split
set.seed(1)
df <- data.frame(class_data,y)
train<- sample(1:nrow(df),300)
lda.fit=lda(y~.-y,data=df[train,], subset = train)
df_test = df[-train,]
lda.pred=predict(lda.fit,df_test)
table(lda.pred$class,df_test$y)
mean(lda.pred$class==df_test$y) 
#training mean accuracy is 98.5%
# Testing accuracy for 75% split is 54%

#############################
#Let's try KNN classification 
train<- sample(1:nrow(df),300)
library(class)
M <- list()
for (i in 1:100){
  knn.pred=knn(class_data[train,],class_data[-train,],y[train],k=i)
  M[i] <- mean(knn.pred==df$y[-train])
}
which.max(M)
knn.pred=knn(class_data[train,],class_data[-train,],y[train],k= which.max(M))
table(knn.pred,y[-train])
mean(knn.pred==y[-train])
# KNN is giving pretty good test accuracy of 61%! 
# It is not suffering from "curse of dimensionality" which is interesting!

#########################################################
#NN : Reload the data before performing NN classification.
# Load the necessary libraries
install.packages("neuralnet")
library(neuralnet)
library(caret)
y<- as.factor(y)
df <- cbind(x,y)
dim(df)
set.seed(1)
train<- sample(1:nrow(df),300)
df_train <- df[train,]
df_test <- df[-train,]
NN <- neuralnet(y~ .-y, df_train, hidden = c(4,2))
NN.pred <- neuralnet::compute(NN, df_test[, -501])
pred.list <- data.frame()
for (i in 1:100){
  pred.list <- rbind(pred.list, which.max(NN.pred$net.result[i,]))
}
pred.list$X2L <- gsub(1, 0, pred.list$X2L)
pred.list$X2L <- gsub(2, 1, pred.list$X2L)
predictions <- as.factor(pred.list$X2L)
reference <- df_test$y
confusionMatrix(predictions, reference)

###########################################################################
#Note: Reload the data before performing RF! Copy this section into new tab.
# Let's build a random forest!
install.packages("randomForest")
library(randomForest)
set.seed(1)
y_f <- as.factor(y)
df_rf <- cbind(x,y_f)
train<- sample(1:nrow(df_rf),300)
df_train_rf <- df_rf[train,]
df_test_rf = df_rf[-train,]
fit_rf <- randomForest(y_f ~ ., data = df_train_rf, ntree = 100)
predictions <- predict(fit_rf, newdata = df_test_rf)
fit_rf
table(predictions,y_f[-train])
mean(predictions==y_f[-train])
# OOB Error rate is 40%. Test Accuracy is 75%


######################### Part-2 : Clustering ##########################################
# Large matrix with 784000 elements!

#######################
# calculate column mean
col_mean <- apply(y, 2, mean) 
# calculate column variance
col_var <- apply(y, 2, var)
plot(col_mean)
which.max(col_mean)
plot(col_var)
which.max(col_var)

###############################
# Principal Components Analysis
pca.out=prcomp(y, scale=TRUE)
# let's calculate the proportion of variance explained
pr.var=pca.out$sdev^2
pve=pr.var/sum(pr.var)
plot(pve, xlab="Principal Component", ylab="Proportion of Variance Explained", ylim=c(0,1),type='b')
plot(cumsum(pve), xlab="Principal Component", ylab="Cumulative Proportion of Variance Explained", ylim=c(0,1),type='b')
# cumsum is a function to calculate cumulative sum of each element in a vector
library(plotrix)
draw.circle(200,0.91,10)
# 200 principal components explain 91% of the variability!
# Converting the first 200 principal components into a data set

# Select the first two principal components
PC_scores <- pca.out$x[, 1:200]

# Save the loadings and mean
loadings <- pca.out$rotation[, 1:200]
mean <- pca.out$center[1:200]

# Perform the inverse transformation
cluster_data <- PC_scores %*% t(loadings) + mean
dim(cluster_data)

##################################
# Let's perform k means clustering
#let's create a for loop and check the minimum total within sum of squares
tot_ss <- list()
for (i in 1:25){
  km.out=kmeans(cluster_data,centers=i,nstart=25)
  tot_ss[i] <- km.out$tot.withinss
}
plot(1:25, tot_ss)
as.numeric(tot_ss)
plot(-diff(as.numeric(tot_ss)))

# This plot suggests that there are mainly 4 clusters.

#########################################
# Statistical interpretation of variables. 
threshold <- 50
y_subset <- y[sapply(y, function(x) all(x > threshold))]
median(y_subset)
#Only 21% of data points with value above 50. In that 60% (of 21%) are above 210 (probably centered around 2 means 220 and 250!)
q1<- quantile(y_subset, 0.1)
q2<- quantile(y_subset, 0.2)
q3<- quantile(y_subset, 0.3)
q4<- quantile(y_subset, 0.4)
q5<- quantile(y_subset, 0.5)
q6<- quantile(y_subset, 0.6)
q7<- quantile(y_subset, 0.7)
q8<- quantile(y_subset, 0.8)
q9<- quantile(y_subset, 0.9)
mean(subset(y_subset, y_subset >= 0 & y_subset <= q1))
# As per my analysis there are 4 or 5 classes of normal distributions with means 0, 125, 150, 220, 250 (ignore 150 if 4)!
# There can be another distribution between 0 and 50 (Not validated by the plot!)

#########################
# Hierarchical Clustering
# We will use these same data and use hierarchical clustering
hc.complete=hclust(dist(cluster_data),method="complete")
hc.single=hclust(dist(cluster_data),method="single")
hc.average=hclust(dist(cluster_data),method="average")
# Lets compare this with the actual clusters in the data. 
# We will use the function `cutree` to cut the tree at level 4 and 5.
# We can use `table` to see how well they match
hc.cut=cutree(hc.complete,4)
table(hc.cut)
hc.cut=cutree(hc.complete,5)
table(hc.cut)
# cut = 4 and cut = 5 are very similar. Which implies there are wither 4 or 5 classes.
# we can also compare hierarchical clustering with kmeans

############################
#Gaussian mixture model (GMM)
library(mclust)
gmm = Mclust(cluster_data, G = 4)
plot(cluster_data,col=gmm$classification,cex=2,pch=1,lwd=2)
#soft assignment
sum(gmm$z[,1]) # 150
sum(gmm$z[,2]) # 100
sum(gmm$z[,3]) # 60
sum(gmm$z[,4]) # 690
gmm = Mclust(cluster_data, G = 5)
plot(cluster_data,col=gmm$classification,cex=2,pch=1,lwd=2)
#soft assignment
sum(gmm$z[,1]) # 145
sum(gmm$z[,2]) # 120
sum(gmm$z[,3]) # 80
sum(gmm$z[,4]) # 55
sum(gmm$z[,5]) # 600
# Pretty close to k-means and hierarchical results!
# I propose 4 classes but there is a good chance for 5 classes!




















































