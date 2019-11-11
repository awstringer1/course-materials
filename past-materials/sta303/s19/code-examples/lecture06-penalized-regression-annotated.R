# STA303 S19 Lecture 6: penalized regression
# 
# # Load required packages
# 

library(dplyr)
library(ggplot2)
library(glmnet)


# Linear regression on the mtcars data.
mtcars <- mtcars %>%
  as_tibble() %>% # Removes rownames and pretty-prints
  select(disp,mpg)

# First thing to do: plot the data, see what type of relationship might be present.
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
# This is because the response and covariate are on different scales.
# So coefficient estimates are not immediately interpretable.
# Rescaling the data mitigates this.
mtscaled <- mtcars %>%
  mutate_all(~(.x - mean(.x))/sd(.x))

mod1 <- lm(mpg~disp,data=mtscaled)
summary(mod1)

# We have centred the data, so an intercept of 0 means that a car with average
# displacement (engine size) has average mpg (fuel economy).
# Further, beta(disp) = -.848. A one standard deviation increase in disp is associated with a .847
# standard deviation decrease in fuel economy, on average.
# Predicted values:
mtscaled %>%
  ggplot(aes(x=disp,y=mpg)) +
  theme_light() +
  geom_point(colour="black",fill="red",pch=21,size=2) +
  geom_abline(intercept = coef(mod1)[1],slope=coef(mod1)[2],colour = "purple") +
  labs(subtitle="Linear Regression Model")

# Definitely looks like a linear relationship is missing some of the structure.
# Can we pick this up in a residual plot?
tibble(x = mod1$fitted.values,
       y = residuals(mod1)) %>%
  ggplot(aes(x=x,y=y)) +
  theme_light() +
  geom_point(colour="black",fill="red",pch=21,size=2) +
  geom_hline(yintercept = 0,colour = "red") +
  labs(title = "Residual plot, linear model",
       x = "Fitted value",
       y = "Residual")

# The scale of the residuals looks pretty good- all fall
# between -2 and 2, which we expect since the data
# was scaled to have unit variance.
# But the pattern? Doesn't look evenly distributed about
# y = 0. Looks like... a quadratic?

# Model fit- R^2
summary(mod1)$r.squared # Fits pretty well.

# Now try a quadratic...
# The poly() function just creates a polynomial out of
# the input variable with the specified degree.
# By default it creates something called "orthogonal polynomials",
# but you can get regular ones using raw = TRUE.
mod2 <- lm(mpg~poly(disp,degree = 2,raw = TRUE),data=mtscaled)
summary(mod2)

# Harder to directly interpret the betas in a polynomial model
# but the standard errors should be not huge compared to the point estimates
# if the fitting procedure was nice and stable.
# (remember: fitting y=beta0 + beta1 x + beta2 x^2) is STILL linear regression
# because it's a linear function of the betas).

mtscaled %>%
  mutate(predvals = predict(mod2)) %>%
  ggplot(aes(x=disp,y=mpg)) +
  theme_light() +
  geom_point(colour="black",fill="red",pch=21,size=2) +
  geom_line(aes(y = predvals),colour = "purple") +
  labs(subtitle="Linear Regression Model with quadratic term")

# A quadratic gives a much better fit, but it's not that flexible.
# We see it starting to curve back up at the right-most x value
# Would a more flexible (higher degree) polynomial be appropriate?

# Residuals:
tibble(x = mod2$fitted.values,
       y = residuals(mod2)) %>%
  ggplot(aes(x=x,y=y)) +
  theme_light() +
  geom_point(colour="black",fill="red",pch=21,size=2) +
  geom_hline(yintercept = 0,colour = "red") +
  labs(title = "Residual plot, quadratic model",
       x = "Fitted value",
       y = "Residual")

# I don't see anything that alarming in this plot. It looks good!

# Coefficients
coef(mod2)

# Rsquared
summary(mod2)$r.squared # Fits a bit better than linear model.

# Adding a quadratic term improved the fit a bit.
# Does adding more terms always improve the fit?

mod3 <- lm(mpg~poly(disp,degree = 20,raw = TRUE),data=mtscaled)
summary(mod3)

# Woah. The coefficients (point estimates) are all over the place in magnitude,
# and direction. Their standard errors are huge.
# This is the "instability" we talked about in lecture.
# Whether it's statistical (sampling) error or numerical error, something is wrong here.

# Rather than plot a linear/quadratic using geom_line(),
# to do predictions for 20 degree polynomial, I use the predict() function
# and then plot its output.
mtscaled %>%
  mutate(predvals = predict(mod3)) %>%
  ggplot(aes(x=disp,y=mpg)) +
  theme_light() +
  geom_point(colour="black",fill="red",pch=21,size=2) +
  geom_line(aes(y = predvals),colour = "purple") +
  labs(subtitle="Linear Regression Model with 20-degree polynomial terms")

# While this model fits THESE data well, it fits them TOO well.
# Sacrifices generalizability to new datasets.
# STatistically: point estimates have very high variance. Confidence intervals
# would be wide. We're asking too much of this small dataset.
# Coefficients
unname(coef(mod3))

# Rsquared
summary(mod3)$r.squared # YEs, this model does fit THESE DATA very well.

# The model fits the observed data better. But it gives ridiculous predictions.
# The coefficients are huge. The fitted curve hugs the observed data too closely.
# This model will not generalize well to new data, and the coefficients are highly
# sensitive/unstable (look at their standard errors/p-values)


### Penalized Regression ###

# glmnet() wants X and y to be provided separately (no formula)
# so get the design matrices from each of the above models.
# If this is unfamiliar, print out each of the below X's and see what's inside.
y <- mtscaled$mpg
X1 <- model.matrix(mod1)
X2 <- model.matrix(mod2)
X3 <- model.matrix(mod3)

# The first model fit well so the answer here shouldn't be too different
# glmnet() picks a bunch of lambda values and tries them all
# We don't really have a good way of choosing lambda... yet.
# We will tackle this in later lectures.
penalized_mod1 <- glmnet(x = X1,y = y,alpha = 0,nlambda = 100)
coef_pen_mod1 <- tibble(
  lambda = penalized_mod1$lambda,
  intercept = coef(penalized_mod1)[1, ],
  disp = coef(penalized_mod1)[3, ]
)
coef_pen_mod1 # Gives estimated coefficients for each value of lambda
# Higher lambda ==> stronger penalty ==> smaller estimates.
View(coef_pen_mod1)
# As lambda gets smaller, the estimated coefficient approaches
# what it was in the unpenalized model.

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

# Pick any lambda- I chose the smallest one. You can try others!
whichlambda <- penalized_mod3$lambda[length(penalized_mod3$lambda)]

# Plot predictions
mtscaled %>%
  mutate(predvals = predict(penalized_mod3,newx = X3,s = whichlambda)) %>%
  ggplot(aes(x=disp,y=mpg)) +
  theme_light() +
  geom_point(colour="black",fill="red",pch=21,size=2) +
  geom_line(aes(y = predvals),colour = "purple") +
  labs(subtitle="Linear Regression Model with 20-degree polynomial terms fit by penalized regression")

# Looks a lot like the quadratic, except it's fine at the
# right end point.
# Very little effort on our part to build this very
# sophisticated model.
# The betas here should have higher variance than if we had
# just fit a linear/quadratic model. We pay a price for
# generality. However, they have MUCH lower variance than 
# the original 20-degree polynomial we tried.
# "No such thing as a free lunch". We pay a small price for
# being able to let the data tell us what model to use.
# I think it's worth it...

# Advanced: plot it for ALL the lambda values together! Woah!
# Requires the purrr package
# THIS WILL NOT BE TESTED
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

# That's a comparison of all the curves fit using different
# values of lambda.

