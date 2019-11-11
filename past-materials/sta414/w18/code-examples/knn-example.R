#' ---
#' title: "K-Nearest Neighbours: Code Example"
#' author: "Alex Stringer"
#' date: "January 2018"
#' output: 
#'   html_notebook:
#'     toc: true
#' ---
#' 

# Load required packages
# 
suppressMessages({
  suppressWarnings({
    library(dplyr)
    library(ggplot2)
    library(GGally)
  })
})

#' # Dataset
#' 
#' ## Description
#' 
#' The dataset we will use is the Iris dataset, which has 4 continuous features
#' a 3-class categorical target
#' 
data(iris)
irist <- tbl_df(iris) %>%
  rename(sepal_length=Sepal.Length,
         sepal_width=Sepal.Width,
         petal_length=Petal.Length,
         petal_width=Petal.Width,
         species=Species
  )

glimpse(irist)

#' ## Plot the Data
#' 
#' The dataset is 4-D, so we can do pairs plots of each variable, and colour the
#' points by class. We can do this using the ggpairs function in the GGally package 
#' 
irist %>%
  ggpairs(aes(colour=species,alpha=0.3))

#' # KNN
#' 
#' Let's implement K Nearest Neighbours to classify **species**
#' 

set.seed(87654) # Just so it's reproducible

knn_iris <- function(K,debug=FALSE) {
  # This function works only on the iris dataset, for purposes of illustration
  
  n <- nrow(irist) # 150
  
  class_names <- irist$species %>% unique()
  train_class <- list()
  
  # Distance function
  # We need a function to measure the distance between two points; we'll use Euclidean
  distance_function <- function(x1,x2) sqrt(sum((x1 - x2)^2))
  
  # Algorithm: loop over the training set, and for each point, calculate the distance to all other points.
  # Pick the K points that are closest, and average their classes
  for (i in 1:nrow(irist)) {
    if (debug) {
      cat("KNN Algorithm, iteration: ",i,"\n")
    }
    x1 <- as.numeric(irist[i,-5])
    # Distance to all other points
    # Again, not going for efficiency here; going for illustration
    distances <- data_frame(idx=numeric(nrow(irist)-1),distance=idx)
    for (j in 1:nrow(irist)) {
      if (j != i) {
        distances[j, ] <- c(j,distance_function(x1,as.numeric(irist[j,-5])))
      }
    }
    # Sort and pick the K smallest
    K_closest_idx <- distances %>%
      filter(idx != 0) %>%
      arrange(distance) %>%
      slice(1:K) %>%
      pull(idx)
    K_closest_obs <- irist[K_closest_idx,"species"]
    
    # Get the class of the K closest points
    class_vec <- numeric(length(class_names))
    names(class_vec) <- class_names
    for (k in 1:length(class_names)) {
      class_vec[k] <- mean(K_closest_obs$species == class_names[k])
    }
    train_class[[i]] <- c("idx" = i,class_vec)
  }
  
  train_class <- purrr::reduce(train_class,bind_rows)
  
  # MAP class estimates
  train_class_map <- character(nrow(train_class))
  for (i in 1:length(train_class_map)) {
    train_class_map[i] <- as.character(class_names)[which.max(train_class[i,-1])]
  }
  
  # Add to training data
  out <- irist
  out$pred_class <- train_class_map
  
  return(out)
}

#' ## K = 3

knn_iris_3 <- knn_iris(3)
# Evaluate the class predictions
# 
knn_accuracy_3 <- mean(knn_iris_3$species == knn_iris_3$pred_class)
cat("Accuracy of 3-NN Model on Iris Dataset: ",format(knn_accuracy_3,digits=3),"\n")

#' Plot the results 
#' 
knn_iris_3 %>%
  ggpairs(aes(colour=pred_class,alpha=0.3))

#' ## K = 1
#' 
#' What should the results of a 1-NN algorithm look like?
#' 
knn_iris_1 <- knn_iris(1)
# Evaluate the class predictions
# 
knn_accuracy_1 <- mean(knn_iris_1$species == knn_iris_1$pred_class)
cat("Accuracy of 1-NN Model on Iris Dataset: ",format(knn_accuracy_1,digits=3),"\n")

#' Plot the results 
#' 
knn_iris_1 %>%
  ggpairs(aes(colour=pred_class,alpha=0.3))


#' Could you cross-validate for K?
#' 
