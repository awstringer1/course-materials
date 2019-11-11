# Supplementary R code for lecture 4, STA303 summer 2018
# Alex Stringer
# 

# If these commands don't work,
# type install.packages("faraway")
library(tidyverse)
library(faraway)

# Any time you get an error message in R,
# GOOGLE IT.
# ...then ask if you still can't figure it out.


# Load and plot the orings data
# What is the orings data?
# ?orings
# 
# We're going to build a regression model that
# allows us to quantify the association between
# temperature at launch and probability of 
# failure of an oring.

# First: load the data, and look at it.
# Always look at the data...
# There are 6 orings per shuttle
# So the MLE for prob of failure is
# (# that failed)/6
data(orings)
orings_tbl <- as_data_frame(orings) %>%
  mutate(prop_damaged = damage / 6) # Maximum likelihood estimate

glimpse(orings_tbl)

# So it kind of looks like lower temperatures
# have more failures. Plot the data? How do we
# plot binomial data...?
orings_tbl %>%
  ggplot(aes(x = temp,y = prop_damaged)) +
  theme_classic() +
  geom_point(pch=21) +
  labs(title = "Orings Data",
       subtitle = "Probability of o-ring failure for 23 space shuttle missions as a function of temperature on launch day",
       x = "Temperature (deg. far.)",
       y = "Probability of Damage") +
  scale_y_continuous(labels = scales::percent_format())

# There are overlapping points on this plot
# so it's hard to see the strength of any relationship
# because lots of points are on top of each other.
# ...could fix the plot.
# Or, do a model.

# Here's what happens if you do a linear regression:
orings_tbl %>%
  ggplot(aes(x = temp,y = prop_damaged)) +
  theme_classic() +
  geom_point(pch=21) +
  geom_smooth(method = "lm",se = FALSE,colour="blue") +
  labs(title = "Orings Data",
       subtitle = "Probability of o-ring failure for 23 space shuttle missions as a function of temperature on launch day",
       x = "Temperature (deg. far.)",
       y = "Probability of Damage") +
  scale_y_continuous(labels = scales::percent_format())

# What are some problems?
# 1) Linear relationship does not look appropriate
# 2) Negative probability, makes no sense.

# Fit a binomial GLM
# family = binomial ==> binomial regression
# family = poisson ==> poisson regression
# family = gaussian ==> same as lm()
# For binomial regression, the formula you specify has to
# have a matrix on the LHS (left hand side).
# Two columns: cbind("successes","failures")
# Confusing in this example. Because "success" is the word
# usually used when talking about the binomial distribution.
# But here, we're modelling the probability of an oring failing.
# So a binomial "success" is an oring failure.
glm1 <- glm(cbind(damage,6-damage) ~ temp,
            data=orings_tbl,
            family=binomial)
summary(glm1)
# IRWLS is sometimes referred to as "Fisher Scoring"
# Bottom of glm summary output tells us that
# IRWLS converged in 6 iterations.
# ...not really proactically important.

# glm1null <- glm(cbind(damage,6-damage) ~ 1,data=orings_tbl,family=binomial)
# summary(glm1null)

# Plot the fitted curve
# In general, you should make sure you can do all these calculations by
# hand, at least in small cases.
# Because you won't have a computer on the test.
# But, any questions that involve large or tedious computations,
# I will give you precalculated output.
# You should expect to have to do vector multiplications (i.e. dot products)
# and matrix multiplications of dimension 2 x 2.
orings_tbl %>%
  # Compute the predicted values.
  # ilogit() is an R function that implements the inverse link
  # So eta = coef(glm1)[1] + coef(glm1)[2]*temp
  # and p = ilogit(eta)
  mutate(predicted_prob = ilogit(coef(glm1)[1] + coef(glm1)[2]*temp)) %>%
  ggplot(aes(x = temp,y = prop_damaged)) +
  theme_classic() +
  geom_point(pch=21) +
  geom_line(aes(y = predicted_prob),colour="blue",size = 1) +
  labs(title = "Orings Data - Fitted Binomial Regression Model",
       subtitle = "Probability of o-ring failure for 23 space shuttle missions as a function of temperature on launch day",
       x = "Temperature",
       y = "Probability of Damage") +
  scale_y_continuous(labels = scales::percent_format())

# Looks better. Can we extrapolate below the range of observed data?

orings_tbl %>%
  ggplot(aes(x = temp,y = prop_damaged)) +
  theme_classic() +
  geom_point(pch=21) +
  geom_line(data = data_frame(
    temp = seq(25,90,by=0.1),
    prop_damaged = ilogit(coef(glm1)[1] + coef(glm1)[2]*temp)
    ),colour="blue",size = 1) +
  labs(title = "Orings Data - Fitted Binomial Regression Model",
       subtitle = "Probability of o-ring failure for 23 space shuttle missions as a function of temperature on launch day",
       x = "Temperature",
       y = "Probability of Damage") +
  scale_y_continuous(labels = scales::percent_format()) +
  scale_x_continuous(breaks = seq(30,80,by=10))


# Predict at a new temp
# E.g. 31F, the launch temperature on the day Challenger exploded

# Create a covariate vector for which you want a prediction.
# Make sure to include the intercept! Which means
# put a 1 as the first value.
# I always forget the intercept :(
xnew <- cbind(c(1,31)) # Intercept
# Get the predicted linear predictor, x^T beta
etapred <- t(xnew) %*% cbind(coef(glm1))
# What is that? That's eta!
etapred
# This is the log-odds of oring failure at 31F.
# I don't know how to interpret that number, do you?
# Predicted probability:
ilogit(etapred)
# About 99%- yikes
# Confidence interval for the prediction?
# Get the standard error for eta.
# The variance matrix of beta is obtained using the vcov()
# function:
varmat_beta <- vcov(glm1)
# diag(varmat_beta) is the vector of estimated variances
# of the estimates for beta
# varmat_beta[1,2] = -.17 is the estimated covariance
# Does it make sense that it's negative?
# The intercept is the log odds of an oring failure
# when the temperature = 0
# By the way, if you want an interpretable intercept,
# you could centre the data. For example, if you subtracted
# 32F from temperature and then ran the regression,
# the intercept would be the log odds of failure at
# temperature = 32F. (for example). 32F is 0C, the freezing
# point of water. 212F is the boiling point of water
# etc.
# So, a larger effect of temperature (bigger beta1)
# means lower baseline effect (beta0), because their
# estimates are negatively correlated.

# Okay build the confidence interval.
etapred_sd <- sqrt(t(xnew) %*% vcov(glm1) %*% xnew)
etapred_sd
etapred
eta_confint <- c(etapred - 2 * etapred_sd,etapred + 2 * etapred_sd)
eta_confint
# What we actually want though is a confidence
# interval for the predicted probability:
ilogit(eta_confint)
# The predicted probability is
ilogit(etapred)
# This is asymmetric, which we talked about, makes sense.

# In R, you don't have to calculate things manually like this
# (but you have to for the test!)
# The predict() function gives predicted linear predictors
# and probabilities.
predict(glm1,
        newdata = data_frame(temp = 31),
        se.fit = TRUE,
        type = "link") # Predict on the link scale

naturalpred <- predict(glm1,
        newdata = data_frame(temp = 31),
        se.fit = TRUE,
        type = "response") # Natural scale
# What happens if I try to make a CI based off
# of that SE?
c(
  naturalpred$fit - 2*naturalpred$se.fit,
  naturalpred$fit + 2*naturalpred$se.fit
)
# It's not the same as what we got...
# YIKES. Our confidence interval for eta is good.
# Why is our confidence interval for the probability so off from what
# predict() outputs?
# predict() uses approximations (something called "the delta method") when 
# working on the response scale. For binomial regression, especially in the tails,
# these approximations are not good.
# The best thing to do is always make your confidence interval on the link scale
# and transform it to the response scale.

# Confidence interval for beta(temp)
# Remember, the variance for beta_j is the jth diagonal element
# of the covariance matrix for beta, which is obtained as vcov(glm1)
betatemp_se <- sqrt(diag(vcov(glm1)))[2]
cint_temp <- c(coef(glm1)[2] - 1.96 * betatemp_se,coef(glm1)[2] + 1.96 * betatemp_se)
cint_temp
# But how is this beta interpreted?
# beta1 = -.12. A unit increase in temperature (so, increasing temp by 1F)
# is associated with a .12 unit DECREASE in the log-odds of oring failure.
# ...that pesky log-odds business again.
# How do we get something more interpretable?
# if eta = log(p/(1-p)) then exp(eta) = p/(1-p), the odds
# So exp(beta1) is the MULTIPLICATIVE FACTOR by which the ODDS change when
# temperature changes
exp(coef(glm1)[2]) # about .81
# So increasing temperature by 1F is associated with a new odds of .81 x the old odds
# A much better way to say this is increasing temp by 1F is
# associated with a (1 - .81 = .19) 19% reduction in
# odds of oring failure.
# It's still not completely easy to interpret, right?
# But better than log odds.
# 
# So how to get a confidence interval for exp(beta1)?
# Same exact trick as before. Compute the confidence interval
# on the link scale, and then you can apply whatever monotone
# transformation you want.
exp(cint_temp)
# That is a confidence interval for the predicted factor by which
# the odds of failure changes with a unit increase in temp.

# Binomial regression is good to try when you 
# have count data (so definitely not normal), and you
# know that each observed count has a small, fixed maximum
# possible value.
# So for the challenger data, we know there are 6 orings per flight
# So only 6 failures could possibly occur.
# 
# If you have really large counts (in the thousands), then
# normal linear regression might actually be fine (sort of)
# If you have smaller counts but you don't know if they
# have a maximum, try Poisson regression (next lecture).



### Binary logistic regression: complete example ###

# Logistic regression is the name given to binomial
# regression when m_i = 1.
# So you are technically responsible for knowing about it.
# However, there is a lot of stuff that is specific
# to logistic regression, e.g. ROC curves.
# You are NOT responsible for the "prediction" parts of the below.
# They are included in case you need this stuff for a job
# interview or whatever.

# We'll analyze the Wisconsin Breast Cancer Data, exercise 2 chapter 2 page 58 from ELMR
# Note the data is misspelled in the Faraway package
data(wbca) # Called wbcd in the book
wbca_tbl <- as_data_frame(wbca)

glimpse(wbca_tbl)

# First step: take a look at the response, Class, which is an indicator of malignancy of
# tumours (1 == benign, 0 == malignant)
with(wbca_tbl,table(Class))
443 / (443 + 238)
238 / (443 + 238)


# We have 9 predictors, so it's feasible to plot the data separately for each predictor
# We'll do this in a clever, "tidy" way

wbca_tbl %>%
  gather(variable,value,Adhes:USize) %>%
  ggplot(aes(x = value,y = Class)) +
  theme_classic() +
  facet_wrap(~variable) +
  geom_jitter(width=0.5,height=0.1,alpha = 0.3) +
  scale_y_continuous(breaks = c(0,1),labels = c("Malignant","Benign")) +
  labs(title = "Malignancy of Tumours, by Predictor",
       subtitle = "WBCA Data",
       x = "Predictor Value",
       y = "Malignant or Benign?")

# We have several predictors, so it is a good idea to look at their correlation matrix
round(cor(wbca_tbl %>% dplyr::select(-Class)),2)
corrplot::corrplot(cor(wbca_tbl %>% dplyr::select(-Class)),order="AOE")
# There are some highly correlated variables, especially UShap and USize

# Fit an initial binary logistic regression model
wbca_glm1 <- glm(Class ~ .,data = wbca_tbl,family = binomial)
summary(wbca_glm1)

# The coefficients all look reasonable
# What about the residual deviance?
# ALL THESE QUANTITIES ARE USELESS FOR BINARY REGRESSION!
# But you can still compute them... but you shouldn't
Dstat <- wbca_glm1$null.deviance - deviance(wbca_glm1)
Dstat
# Compare to a chisq(680 - 671) distribution
1 - pchisq(Dstat,9)

# So the model fits better than the null model
# NO! CAN'T SAY THAT
# Compare to the saturated model
# But wait... what is the deviance of the saturated model for BINARY logistic regression?
y <- wbca_tbl %>% pull(Class)
dev_sat <- sum( y*log(y) + (1 - y)*log(1 - y))
dev_sat
# What's the problem?
# 
# Take dev_sat = 0 (again, why?)
# Then a test of whether the model fits as well as the saturated model is simply a test of comparing
# the residual deviance to its degrees of freedom
deviance(wbca_glm1)
1 - pchisq(deviance(wbca_glm1),671)
# Small residual deviance ==> model fits the data "as well" as the saturated model... usually
# But here, we cannot use residual deviance, because it doesn't depend on the data. Only on p-hat
# Note though that the chisquare approximation is terrible for binary data, and there are other problems in using deviance with binary data
# Let's make sure the null model doesn't also fit the data as well as the saturated model
1 - pchisq(wbca_glm1$null.deviance,680)
# But don't trust these comparisons, for BINARY data
# The above would be a good thing to do for binomial data with bigger n per observation
# Or for count data
# But NOT for BINARY data
# 
# Now. Can we get a simpler model that fits just as well?

wbca_glm2 <- step(wbca_glm1,
                  lower = glm(Class~1,data=wbca_tbl,family=binomial))
summary(wbca_glm2)

# We removed Epith and USize. Remember that USize was highly correlated with UShap.
# It's likely that either of these variables would have been fine; e.g. if there are business/scientific
# reasons for wanting to include one over the other, you probably could choose
# 
# What next? Let's look at the actual predicted probabilities. The variables in the model were all pretty skewed, so what do
# we expect the distribution of predicted probabilities to look like?

# FYI, you should be able to compute the below from scratch, i.e. not using predict()
wbca_predicted_probs <- predict(wbca_glm2,type="response")

data_frame(x = wbca_predicted_probs) %>%
  ggplot(aes(x = x)) +
  theme_classic() +
  geom_histogram(bins = 100,colour = "black",fill = "orange") +
  labs(title = "Histogram of Predicted Probabilities",
       subtitle = "Predicted probability of tumour being benign, logistic regression on WBCA data",
       x = "Predicted Probability",
       y = "# of tumours") +
  scale_x_continuous(labels = scales::percent_format())

# Most of the predicted probabilities are near 0, or near 1
# How to make a hard 0/1 prediction for each case? The textbook suggests using a cutoff of 0.5.
# Let's look at the classification error if we do that

wbca_tbl %>%
  dplyr::select(Class) %>%
  bind_cols(data_frame(predprob = wbca_predicted_probs)) %>%
  mutate(ypred = as.numeric(predprob > .5)) %>%
  group_by(Class,ypred) %>%
  summarize(cnt = n(),
            pct = scales::percent(cnt / nrow(wbca_tbl)))

# What about using .9 as a cutoff?

wbca_tbl %>%
  dplyr::select(Class) %>%
  bind_cols(data_frame(predprob = wbca_predicted_probs)) %>%
  mutate(ypred = as.numeric(predprob > .9)) %>%
  group_by(Class,ypred) %>%
  summarize(cnt = n(),
            pct = scales::percent(cnt / nrow(wbca_tbl)))

# False Positive: ACTUAL negative, PREDICT positive
# False Negative: ACTUAL positive, PREDICT negative
# True Positive: ACTUAL positive, PREDICT positive
# True Negative: ACTUAL negative, PREDICT negative
# 
# False positive rate: FP / (All actual negatives)
# = FP / (FP + TN)
# True positive rate: TP / (All actual positives)
# = TP / (TP + FN)
# 
# Strongly recommend you read the wikipedia article too.
# And then copy this out 100 times.

# We see there is a tradeoff- When we increase the cutoff, we get less false positives, and more false negatives
# Since a positive here means benign, a false positive is classifying a malignant tumour as being benign, which seems worse
# We can plot a parametric curve of the false positives vs true positives for cutoffs ranging from 0 to 1- the ROC curve
# A wide ROC curve means there exists favourable cutoffs, i.e. the model is good
# 
# I won't require you to be able to create the below from scratch like I am here
# But you do need to know how to interpret the below graph.

fpr <- function(y,ypred) sum(ypred==1 & y==0) / sum(y==0)
tpr <- function(y,ypred) sum(ypred==1 & y==1) / sum(y==1)

wbca_with_predprob <- wbca_tbl %>%
  dplyr::select(Class) %>%
  bind_cols(data_frame(predprob = wbca_predicted_probs))

roc_data <- wbca_with_predprob %>%
  arrange(predprob) %>%
  pull(predprob) %>%
  round(3) %>%
  unique() %>%
  map(~c(.x,tpr(y = wbca_with_predprob$Class,ypred = as.numeric(wbca_with_predprob$predprob >= .x)),fpr(y = wbca_with_predprob$Class,ypred = as.numeric(wbca_with_predprob$predprob >= .x)))) %>%
  map(~.x %>% as_data_frame() %>% t() %>% as_data_frame()) %>%
  reduce(bind_rows) %>%
  rename(h=V1,tpr=V2,fpr=V3)

roc_data %>%
  ggplot(aes(x=fpr,y=tpr)) +
  theme_light() +
  geom_path(colour="#195FFF",size=1) +
  scale_x_continuous(labels=scales::percent_format()) +
  scale_y_continuous(labels=scales::percent_format()) +
  labs(title="ROC Curve",
       subtitle = "Wisconsin Breast Cancer Data - Logistic Regression Model",
       x="FPR",
       y="TPR") +
  geom_line(data = data_frame(x = c(0,1),y = c(0,1)),aes(x=x,y=y),linetype="dashed",colour="red")

# Every point on this curve is a (FPR,TPR)
# combination for a given cutoff
# Answers: can we pick a cutoff that separates the 0s and 1s?
# And, what FPR/TPR correspond to a given cutoff?
