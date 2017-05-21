library(ggplot2)
library(gridExtra)
library(class)
library(tree)

#glass identification dataset
data.glass <- read.table("D:/Dropbox/UCSB/2015SPRING/PSTAT131/131 Final Project/glass/glass.txt",
                         sep="," , header = FALSE) #import the data
colnames(data.glass) <- c("ID","RI","Na","Mg","Al","Si","K","Ca","Ba","Fe","Type") #give column names
dim(data.glass) #acquire dimension information for the datase
data.glass <- data.glass[3:11] #exclude ID and RI from the dataset, we do not need it to analyze it

#create a variable according to the Type of glass
type <- as.factor(ifelse(data.glass$Type == 1,"one",
                         (ifelse(data.glass$Type == 2,"two",
                                 (ifelse(data.glass$Type == 3,"three",
                                         (ifelse(data.glass$Type == 4,"four",
                                                 (ifelse(data.glass$Type == 5,"five",
                                                         (ifelse(data.glass$Type == 6,"six","seven"))))))))))))
data.glass <- data.glass[1:8]
data.glass1 <- data.frame(cbind(data.glass,type)) #add the type attribute into the data set

# #acquire plots of different attributes in relation to Type
# g2 <- qplot(type, Na,  data = data.glass1, colour = Na, geom = c("point","jitter"))
# g3 <- qplot(type, Mg,  data = data.glass1, colour = Mg, geom = c("point","jitter"))
# g4 <- qplot(type, Al,  data = data.glass1, colour = Al, geom = c("point","jitter"))
# g5 <- qplot(type, Si,  data = data.glass1, colour = Si, geom = c("point","jitter"))
# g6 <- qplot(type, K,  data = data.glass1, colour = K, geom = c("point","jitter"))
# g7 <- qplot(type, Ca,  data = data.glass1, colour = Ca, geom = c("point","jitter"))
# g8 <- qplot(type, Ba,  data = data.glass1, colour = Ba, geom = c("point","jitter")) 
# g9 <- qplot(type, Fe,  data = data.glass1, colour = Fe, geom = c("point","jitter")) 
# #plot the scatterplot
# grid.arrange(g2,g3,g4,g5,g6,g7,g8,g9,ncol = 2, nrow = 4, main = "Scatterplot") 

set.seed(2)#set the seed so the answer can be reproducible

N <- nrow(data.glass1) #set the number of observations of the dataset
index.train <- sample(1:N,size = floor(N*0.75), replace = FALSE) #75% for observation
index.test <- setdiff(1:N,index.train) #the rest 25% for validation

train.glass <- data.glass1[index.train,] #apply the indexing to produce the train set
test.glass <- data.glass1[index.test,] #apply the indexing to priduce the test set

#check the dimensionality of the sets
dim(data.glass1)
dim(train.glass)
dim(test.glass)

#obtain corresponding values of the response in train and test dataset
class.train <- data.glass1[index.train, "type"]
class.test <-data.glass1[index.test, "type"]

tree.glass <- tree(type~Na+Mg+Al+Si+K+Ca+Ba+Fe,data=train.glass) #fit tree with type as response
summary(tree.glass) #acquire summary statistics of the tree
plot(tree.glass) #plot the tree
text(tree.glass,pretty=0,col="red",cex=0.5) #add text to the tree

cv.glass <- cv.tree(tree.glass, FUN = prune.misclass)
plot(cv.glass$size, cv.glass$dev, type="b", main = "Cross-Validated Error", 
     xlab = "No. of nodes", ylab = "Misclassification", col = " blue")
abline(h = min(cv.glass$dev),lty=2) # horizontal line at the minimum misclassification error

prune.glass <- prune.misclass(tree.glass,best=10) #prune the tree with optimal terminal nodes
plot(prune.glass) #plot the pruned tree
text(prune.glass,pretty=0,col="red",cex=1) #add text to the pruned tree

prune.pred <- predict(prune.glass,train.glass,type="class") # Predict on the training data using pruned tree
train.conf.mat <- table(prune.pred,train.glass$type) # Construct confusion matrix
1-sum(diag(train.conf.mat))/sum(train.conf.mat) #calculate the misclassification error on training data

prune.test.pred <- predict(prune.glass,test.glass,type="class")
test.conf.mat <- table(prune.test.pred,test.glass$type)
1-sum(diag(test.conf.mat))/sum(test.conf.mat)
# 
# tree.test.pred <- predict(tree.glass,test.glass,type="class")
# tree.test.conf.mat <- table(tree.test.pred,test.glass$type)
# 1-sum(diag(tree.test.conf.mat))/sum(tree.test.conf.mat)

#k-NN
vars <-c("Na","Mg","Al","Si","K", "Ca", "Ba", "Fe")
m <- 30 # Maximum number of k's to be considered
knn.error <- vector() # Intialize vector for keeping track of knn error
for (j in 1:m){ # m: Maximum number of values of k
  model.knn <- knn(train = train.glass[,vars], # Training data with selected variables
                   test = test.glass[,vars], # Test data with selected variables
                   cl = class.train, # labels for training data
                   k = j, # Number of nearest-neighbors
                   prob = T) # Return posterior probabilities of the class given the obs. data
  error <- table(model.knn,class.test)
  knn.error[j] <- 1-sum(diag(error))/sum(error) # Compute Misclassification Error
}

plot(1:m,knn.error, type="l", col = "red",
     xlab = "No. of Nearest Neighbors (k)",
     ylab = "Misclassification Error")
points(knn.error,col = "red", cex=.8) # add points to previous plot
abline(v = which.min(knn.error),lty=2) # Identify the minimum error and add vertical line
abline(h = min(knn.error),lty=2) # horizontal line at the minimum misclassification error

# k-Fold Cross-validation

GetFoldIndex <- function(tt,n.folds){
  #This function creates a vector of indexes that corresponds to n.folds
  #equal size sub-samples taken from tt
  #Arguments:
  #tt: data set with explanatory variables and class variable
  #n.folds: Number of folds (number of sub-samples)
  n <- dim(tt)[1] # Number of observations
  #vector of folds lables
  folds <- rep(1:n.folds,each=floor(n/n.folds))
  remainder <- n-length(folds)
  #number of folds might not be a multiple of total number of obs. so
  #assign remaining obs to a fold systematically: i.e. 1st goes to fold 1, etc
  if(length(remainder)>0){
    folds <- c(folds,1:remainder)
  }
  #permute indexes
  folds <- sample(folds)
  return(folds)
}

MiscErrorKNN <- function(X,responseY,m,n){
  # Args:
  # X: dataset with explanatory variables
  # responseY : lables
  # m: max value for nearest neighbor
  # n: Number of folds
  error.cv <- list()
  # Add index of sub-samples for n-fold validation
  data.set <- data.frame(X,responseY,Fold=GetFoldIndex(X,n.folds = n))
  # 100% observations plus vector of subsamples
  for(i in 1:n){
    # Training data
    train.set <- subset(data.set, Fold != i)[,colnames(X)]
    # Test data
    test.set <- subset(data.set, Fold == i)[,colnames(X)]
    # Vector of classes
    class.train <- subset(data.set,Fold != i)[,"responseY"]
    class.test <- subset(data.set,Fold == i)[,"responseY"]
    # For these given samples fit k-NN model for several values of k
    knn.error <- vector() # initialize vector
    for (j in 1:m){ # m: Maximum number of values of k
      model.knn <- knn(train = train.set,
                       test = test.set,
                       cl = class.train,
                       k=j,
                       prob=T) # Fit model
      error <- table(model.knn, class.test) #Compute Error
      knn.error[j] <- 1-sum(diag(error))/sum(error) #return(knn.error)
    }
    error.cv[[i]] <- knn.error
  }
  return(error.cv)
}

CrossValid <- MiscErrorKNN(X = data.glass1[,vars], #Explanatory Variables
                           responseY = data.glass1[,"type"], # labels
                           m=30, #Maximum value of k, for k-NN
                           n=10) #Number of folds (subsamples)
class(CrossValid)
names(CrossValid) <- paste("Sample",1:10) # Assign names

matplot(data.frame(CrossValid), type = "l", lty=1,
        ylab = "Error Rate",
        xlab = "k",
        main = "10-Fold Misclassification Test Error")

mean.error <- apply(data.frame(CrossValid),1,mean) #get the mean for each k
boxplot(t(data.frame(CrossValid))) #Box plot of errors
lines(1:30,mean.error, type = "l",lwd=2, col = "red") # Add mean
title(paste("10-Fold Avg. Cross Validation Error: Local minimum:k=",
            which.min(mean.error)),
      xlab = "k (No. of Nearest Neighbors)",
      ylab = "Misclassification Error")
#Place lines to indicate minimum average error and value of k
abline(h=mean.error[which.min(mean.error)], v = which.min(mean.error), col = "gray",lty = 2)
points(which.min(mean.error),mean.error[which.min(mean.error)], pch = 19,col = "blue",cex=1)
