#' ---
#' title: "Linear Basis Function Models: Code Example"
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
  })
})

#' # Linear Regression Example
#' 
#' Linear regression on the mtcars dataset, regressing mpg ~ disp
#' 
#' ## Look at the Data
#' 
mtcars %>%
  as_data_frame() %>% # Removes rownames and pretty-prints
  select(disp,mpg)

#' ## Plot the Data
#' 
mt_plot <- mtcars %>%
  ggplot(aes(x=disp,y=mpg)) +
  theme_light() +
  geom_point(colour="black",fill="red",pch=21,size=2) +
  labs(title = "Plot of Fuel Economy vs Engine Displacement, mtcars Data",
       x="Displacement",
       y="Miles per Gallon")

mt_plot

#' ## Linear Regression
#' 
#' The relationship between disp and mpg looks curvy. Let's first fit a linear regression, as a baseline
#' 
mod1 <- lm(mpg~disp,data=mtcars)
summary(mod1)

# Predicted values
mt_plot + 
  geom_abline(intercept = coef(mod1)[1],slope=coef(mod1)[2]) +
  labs(subtitle="Linear Regression Model")
  
# Get the root-squared prediction error
sqrt(sum((mtcars$mpg - predict(mod1))^{2}))

#' # Linear Basis Function Models
#' 
#' Let's look at some possible LBMFs
#' 
#' ## Polynomial Basis Functions
#' 

poly_model <- function(m) {
  # The "poly" function in R generates polynomial basis expansions
  lm(mpg~poly(disp,m,raw=TRUE),data=mtcars)
}

# Example: poly(disp,2)
mtcars$disp^2
poly(mtcars$disp,2,raw=TRUE)
all(mtcars$disp^2 == poly(mtcars$disp,2,raw=TRUE)[,2])

#' ### M = 1
#' 

polymod1 <- poly_model(1)
summary(polymod1)

# Predicted values
mt_plot + 
  geom_line(data = data_frame(y=predict(polymod1),x=mtcars$disp),
            aes(x=x,y=y)) +
  labs(subtitle="Polynomial LBFM, M = 1")

# Get the root-squared prediction error
sqrt(sum((mtcars$mpg - predict(polymod1))^{2}))

#' ### M = 2
#' 

polymod2 <- poly_model(2)
summary(polymod2)

# Predicted values
mt_plot + 
  geom_line(data = data_frame(y=predict(polymod2),x=mtcars$disp),
            aes(x=x,y=y)) +
  labs(subtitle="Polynomial LBFM, M = 2")

# Get the root-squared prediction error
sqrt(sum((mtcars$mpg - predict(polymod2))^{2}))

#' ### M = 9
#' 

polymod9 <- poly_model(9)
summary(polymod9)

# Predicted values
mt_plot + 
  geom_line(data = data_frame(y=predict(polymod9),x=mtcars$disp),
            aes(x=x,y=y)) +
  labs(subtitle="Polynomial LBFM, M = 9")

# Get the root-squared prediction error
sqrt(sum((mtcars$mpg - predict(polymod9))^{2}))

#' ### M = 31
#' 

polymod31 <- poly_model(31)
summary(polymod31)

# Predicted values
mt_plot + 
  geom_line(data = data_frame(y=predict(polymod31),x=mtcars$disp),
            aes(x=x,y=y)) +
  labs(subtitle="Polynomial LBFM, M = 31")

# Get the root-squared prediction error
sqrt(sum((mtcars$mpg - predict(polymod31))^{2}))

#' This should have fit the training data perfectly, except we got numerical overflow so some coefficients went to 0
#' 
#' You can get past this by scaling features and targets so they lie in (0,1)- we won't go into detail here
#' 
#' Cross-validation to pick M?
#' 
#' ### Pick M by CV
#' 
set.seed(13423)
m_grid <- 1:12 # Arbitrary
cv_results <- vector(mode="list",length=12)
names(cv_results) = stringr::str_c("m",m_grid)
folds <- 4 # 4-fold CV

# Split the data into "folds" # of folds
data_folds <- split(mtcars,sample(rep(1:folds,nrow(mtcars)/folds)))

# NOT the most computationally efficient implementation! Designed for clarity! This is a teaching example!
# Don't judge me please :(

for (m in m_grid) {
  # Repeat the CV procedure for each m
  cv_results[[m]] <- list()
  for (i in 1:folds) {
    holdout <- data_folds[[i]] # Hold out the ith fold
    training <- purrr::reduce(data_folds[-i],bind_rows)
    mod <- lm(mpg~poly(disp,m,raw=TRUE),data=training)
    cv_results[[m]][[i]] <- sqrt(sum((holdout$mpg - predict(mod,newdata = holdout))^2))
  }
}

# Compute the CVPE- crossvalidated predictive error. Just the average of the 4 test errors for each M
cv_error <- purrr::map_df(cv_results,
                       ~purrr::reduce(.x,sum) / folds) %>%
  tidyr::gather("m","error",m1:m12) %>%
  dplyr::mutate(m = as.numeric(stringr::str_replace_all(m,"m","")))

cv_error

cv_error %>%
  ggplot(aes(x = m,y = log(error))) +
  theme_light() +
  geom_point(fill="red",colour="black",pch=21,size=2) +
  geom_line(colour="black",alpha=0.5) +
  scale_x_continuous(breaks = 1:12) +
  labs(title = "Cross-validated Predictive Error",
       subtitle="Log Scale",
       x="M",
       y="log(error)")

best_m <- cv_error$m[which(cv_error$error == min(cv_error$error))]
best_m


#' ### Model with the best M = 3
polymod3 <- poly_model(3)
summary(polymod3)

# Predicted values
mt_plot + 
  geom_line(data = data_frame(y=predict(polymod3),x=mtcars$disp),
            aes(x=x,y=y)) +
  labs(subtitle="Polynomial LBFM, M = 3")

# Get the root-squared prediction error
sqrt(sum((mtcars$mpg - predict(polymod3))^2))
