library(caret)
library(randomForest)
library(doMC)

#Configure basic environment
#Seed for consistent results
set.seed(147852)
#Running on a multicore
doMC::registerDoMC(cores=4)

#Load training data
trainingData = read.csv("pml-training.csv");
#Remove all near zero variables
trainingData = trainingData[,-nearZeroVar(trainingData[,-150])]
#Remove all elements dat are not informative (i.e. averages and totals)
trainingData = trainingData[,-grep("^new_|^avg_|^user_name|^raw|^num|^total|^max|^min|^amplitude|^var|^stddev|^cvtd|^kurtosis|^skewness", colnames(trainingData))]
#Remove column index number
trainingData = trainingData[,-c(1)]

# Split into training and test data
inTrain = createDataPartition(y=trainingData$classe,p=0.75, list=FALSE)
training = trainingData[inTrain,]
testing = trainingData[-inTrain,]

rbind("original dataset" = dim(trainingData),"training set" = dim(training))

require(parallel)
require(doParallel)
cl <- makeCluster(detectCores() - 1)
registerDoParallel(cl)

ctrl <- trainControl(classProbs=TRUE,
                     savePredictions=TRUE,
                     allowParallel=TRUE)

system.time(rf <- train(classe ~ ., data=training, method="rf", prox=TRUE, ntree=300))
stopCluster(cl)

#rf<-randomForest(classe ~ ., data=training, method="rf", prox=TRUE, ntree=300)
print(rf)
print(rf$finalModel)
varImpPlot(rf)

predictions <- predict(rf, testing, type = "class")
confusionMatrix(predictions, testing$classe)


testDataUrl <- 'http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv';
testData = read.csv(url(testDataUrl));

testDataPredictions = predict(rf, testData, type = "Class")

pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_solutions/problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

pml_write_files(testDataPredictions)





