# STA303 S19 Lecture 6: penalized regression
# 
# Load required packages
# 

library(dplyr)
library(ggplot2)
library(glmnet)


# Linear regression on the mtcars data.
mtcars <- mtcars %>%
  as_tibble() %>% # Removes rownames and pretty-prints
  select(disp,mpg)

# Plot
mt_plot <- mtcars %>%
  ggplot(aes(x=disp,y=mpg)) +
  theme_light() +
  geom_point(colour="black",fill="red",pch=21,size=2) +
  labs(title = "Plot of Fuel Economy vs Engine Displacement, mtcars Data",
       x="Displacement",
       y="Miles per Gallon")

mt_plot

# Looks curvy. Linear regression?
# Scale the data first so that y and x have mean 0 and variance 1
mtscaled <- mtcars %>%
  mutate_all(~(.x - mean(.x))/sd(.x))

mod1 <- lm(mpg~disp,data=mtscaled)
summary(mod1)

# Predicted values
mtscaled %>%
  ggplot(aes(x=disp,y=mpg)) +
  theme_light() +
  geom_point(colour="black",fill="red",pch=21,size=2) +
  geom_abline(intercept = coef(mod1)[1],slope=coef(mod1)[2],colour = "purple") +
  labs(subtitle="Linear Regression Model")

# Model fit- R^2
summary(mod1)$r.squared

# Now try a quadratic...
mod2 <- lm(mpg~poly(disp,degree = 2,raw = TRUE),data=mtscaled)
summary(mod2)

mtscaled %>%
  mutate(predvals = predict(mod2)) %>%
  ggplot(aes(x=disp,y=mpg)) +
  theme_light() +
  geom_point(colour="black",fill="red",pch=21,size=2) +
  geom_line(aes(y = predvals),colour = "purple") +
  labs(subtitle="Linear Regression Model with quadratic term")

# Coefficients
coef(mod2)

# Rsquared
summary(mod2)$r.squared

# Adding a quadratic term improved the fit a bit.
# Does adding more terms always improve the fit?

mod3 <- lm(mpg~poly(disp,degree = 20,raw = TRUE),data=mtscaled)
summary(mod3)

mtscaled %>%
  mutate(predvals = predict(mod3)) %>%
  ggplot(aes(x=disp,y=mpg)) +
  theme_light() +
  geom_point(colour="black",fill="red",pch=21,size=2) +
  geom_line(aes(y = predvals),colour = "purple") +
  labs(subtitle="Linear Regression Model with 20-degree polynomial terms")

# Those are the actual predicted values. We can also plot the functional relationship
# found by the model...
pred_big_polynomial <- function(x) {
  pp <- cbind(rep(1,length(x)),poly(x,20,raw = TRUE))
  beta <- unname(coef(mod3))
  beta[is.na(beta)] <- 0
  beta <- cbind(beta)
  as.numeric(pp %*% beta)
}
bigpolyplot <- mtscaled %>%
  ggplot(aes(x=disp,y=mpg)) +
  theme_light() +
  geom_point(colour="black",fill="red",pch=21,size=2) +
  stat_function(fun = pred_big_polynomial,colour = "purple") +
  labs(subtitle="Linear Regression Model with 20-degree polynomial terms")

bigpolyplot
# Woah! Blows up. Zoom in back to the original range of the data:
bigpolyplot + coord_cartesian(ylim = c(-2,2))
# Yikes. Not good.
# 
# Coefficients
unname(coef(mod3))

# Rsquared
summary(mod3)$r.squared

# The model fits the observed data better. But it gives ridiculous predictions.
# The coefficients are huge. The fitted curve hugs the observed data too closely.
# This model will not generalize well to new data, and the coefficients are highly
# sensitive/unstable (look at their standard errors/p-values)


### Penalized Regression ###

y <- mtscaled$mpg
X1 <- model.matrix(mod1)
X2 <- model.matrix(mod2)
X3 <- model.matrix(mod3)

# The first model fit well so the answer here shouldn't be too different
penalized_mod1 <- glmnet(x = X1,y = y,alpha = 0,nlambda = 100)
coef_pen_mod1 <- tibble(
  lambda = penalized_mod1$lambda,
  intercept = coef(penalized_mod1)[1, ],
  disp = coef(penalized_mod1)[3, ]
)
coef_pen_mod1

# Plot the disp coefficient as a function of lambda and compare to
# our unpenalized model
coef_pen_mod1 %>%
  ggplot(aes(x = lambda,y = disp)) +
  theme_light() +
  geom_line() +
  geom_hline(yintercept = coef(mod1)[2],linetype = "dashed",colour = "red") +
  coord_cartesian(xlim = c(0,10))

# Low lambda ==> limited penalization ==> answer is close to unpenalized
# High lambda ==> a lot of penalization ==> answer doesn't want to move far from zero

# Try the quadratic one yourself.

# Try the big one:

penalized_mod3 <- glmnet(x = X3,y = y,alpha = 0,nlambda = 100)
coef_pen_mod3 <- tibble(lambda = penalized_mod3$lambda) %>%
  bind_cols(as_tibble(as.matrix(t(coef(penalized_mod3)))))

options(scipen = 999) # Turn off scientific notation
View(coef_pen_mod3) # Opens in a spreadsheet

whichlambda <- penalized_mod3$lambda[length(penalized_mod3$lambda)]

# Plot predictions
mtscaled %>%
  mutate(predvals = predict(penalized_mod3,newx = X3,s = whichlambda)) %>%
  ggplot(aes(x=disp,y=mpg)) +
  theme_light() +
  geom_point(colour="black",fill="red",pch=21,size=2) +
  geom_line(aes(y = predvals),colour = "purple") +
  labs(subtitle="Linear Regression Model with 20-degree polynomial terms fit by penalized regression")


# Advanced: plot it for ALL the lambda values together! Woah!
# Requires the purrr package
library(purrr)
predframe <- purrr::map(penalized_mod3$lambda,
                        ~mtscaled %>% 
                          mutate(predvals = predict(penalized_mod3,newx = X3,s = .x)[,1],
                                 lambda = .x)) %>%
  purrr::reduce(rbind)

predframe

predframe %>%
  ggplot(aes(x=disp,y=mpg,group = lambda)) +
  theme_light() +
  geom_point(colour="black",fill="red",pch=21,size=2) +
  geom_line(aes(y = predvals),colour = "grey",alpha = .1) +
  labs(subtitle="Linear Regression Model with 20-degree polynomial terms fit by penalized regression")


