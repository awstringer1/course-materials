### Fit a Boosted Tree classifier
### 

# User rpart to fit the trees
# install.packages("rpart")
# 

library(dplyr)
library(rpart)
library(purrr)

# Simulated data: recreate the simulation from ESL, page 339
# 
set.seed(9876)

# Generate 10 independent N(0,1) features, and assign a target
# 
sample_data <- 1:10 %>%
  map(~rnorm(12000)) %>%
  map(~as_data_frame(.x)) %>%
  reduce(cbind) %>%
  (function(x) {
    names(x) <- stringr::str_c("X",1:length(x))
    x
  }) %>%
  mutate(y = factor(if_else(rowSums(.^2) > qchisq(0.5,10),1,-1))) %>%
  tbl_df()
  

# Split into 2,000 training and 10,000 test observations
N <- 2000
idx <- sample(1:12000,N,replace=FALSE)
traindat <- sample_data[idx, ]
testdat <- sample_data[-idx, ]


# Boosting algorithm, Adaboost
# 
# Set M 
M <- 100
# 1) Initialize weights
#
w <- rep(1/N,N)

# Set the control parameters for rpart
# Want to grow shallow tree
control <- rpart.control(
  cp=0.000001,
  minsplit=2000
)

# Save the intermediate results
alpha_vec <- numeric(M)
eps_vec <- numeric(M)
w_vec <- list()
valerror_vec <- numeric(M)

# 2) Loop over M
# 
for (m in 1:M) {
  # Fit the tree on the weighted training data
  weighted_train <- traindat %>%
    mutate_at(vars(X1:X10),funs(. * w))
  fm <- rpart(y~.,data=weighted_train,control=control)
  
  preds <- predict(fm,type="class")
  
  eps <- sum(w * (preds != traindat$y)) / sum(w)
  raw_missclass <- mean(preds != traindat$y)
  alpha <- log((1-eps)/eps)
  
  w <- w * exp(alpha * (ifelse(preds != traindat$y,1,-1)))
  
  alpha_vec[m] <- alpha
  eps_vec[m] <- eps
  w_vec[[m]] <- w
  
  valerror <- mean(predict(fm,newdat=testdat,type="class") != testdat$y)
  
  valerror_vec[m] <- valerror
  
  if (m==1 | m %% 10 == 0) {
    print(stringr::str_c("==============================Iteration ",m," of boosting=============================="))
    print("Raw Missclassification === Weighted Misclassification === Alpha === Validation Error")
    print(stringr::str_c("         ",round(raw_missclass,3),"                      ",round(eps,3),"                    ",round(alpha,3),"          ",round(valerror,3)))
  }
  
}
