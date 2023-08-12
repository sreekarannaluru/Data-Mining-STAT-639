################################
# Proposed Neural Network Model
library(neuralnet)
library(caret)
ynew <- NULL
ynew<- as.factor(ynew)
df <- cbind(xnew,ynew)
NN.pred <- neuralnet::compute(NN, df[, -501]) ## NN is the trained NN model. 
pred.list <- data.frame()
for (i in 1:100){
  pred.list <- rbind(pred.list, which.max(NN.pred$net.result[i,]))
}
pred.list$X2L <- gsub(1, 0, pred.list$X2L)
pred.list$X2L <- gsub(2, 1, pred.list$X2L)
predictions <- as.factor(pred.list$X2L)
reference <- df$ynew
confusionMatrix(predictions, reference)
test_error <- 1 - mean(predictions==ynew)




########################
# Saving the RData file.
save(ynew, test_error, file = "37.RData")
