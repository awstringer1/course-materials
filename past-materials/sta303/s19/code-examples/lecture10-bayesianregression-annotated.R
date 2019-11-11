### Bayesian Linear Regression example: STA303 Summer 2019 lecture 10 (?) ###

library(tidyverse) # Usual
library(INLA) # For computations
library(brinla) # For summaries + datasets
library(GGally) # For fancy plots

# Load the pollution data

data(usair,package = "brinla")
usair <- as_tibble(usair)
glimpse(usair)
# ?usair

## INITIAL DATA ANALYSIS ##

# Plot all the variables in the dataset:
pairs_chart <- GGally::ggpairs(usair[,-1],
                       lower = list(continuous = "cor"),
                       upper = list(continuous = "points", combo = "dot")) + 
  ggplot2::theme(axis.text = element_text(size = 6)) +
  ggplot2::theme_light()

pairs_chart

# What do we see from this?
# - manuf and pop are almost perfectly correlated
# - there appears to be one location which
# is "different" from the others- outlier?

# Take a look at the response. Histogram?
usair %>%
  ggplot(aes(x = SO2)) +
  theme_light() +
  geom_histogram(colour = "black",fill = "blue",alpha = .5) +
  labs(title = "Sulphur Dioxide in air in US",
       x = "SO2",
       y = "Count")

# It looks somewhat sparse, and not
# normally distributed (to me).
# EXERCISE: could you come up a transformation,
# like log() or a power transformation, that improves
# the "normality" of the data?
# Try a "boxcox" transformation (google it).
# or type ?boxcox.

# Scale the data to have mean 0 and variance 1
# NOTE: the choice of prior for beta assumes you've done this!!!
# If you didn't do this, what would happen? Think about the PRIOR BELIEFS you're
# imposing on beta. Think back to ridge regression too.
# Reason is: software usually assumes you're using a prior covariance matrix
# that is sigma^2_beta * I, where you specify sigma^2_beta.
# Which means, you set the variance of beta... but they are all the same!
# So if you had a variable measured in units of, say, grams, and one measured
# in kilograms, then your prior variance would be very different for these.
# If you scale all variables to have mean zero and variance one, you remove
# this difference.
# Also, you know how to make probability statements about the normal.
# Like P(|beta_0| > 2) = 5% if sigma^2_beta = 1.
# So scaling y lets you set priors using the normal distribution.
# For example, if sigma^2_beta = 4, then P(|beta_0|/2 > 2) = 5%.
# And so on.
# This is how you set the prior for beta, and covers a way to pick lambda
# in ridge regression.

usair_scaled <- usair %>%
  mutate_all(~(.x - mean(.x))/sd(.x))

apply(usair_scaled,2,mean)
apply(usair_scaled,2,sd)


# Basic linear model
usair_lm1 <- lm(SO2 ~ .,data = usair_scaled)
summary(usair_lm1)

# e.g. every one standard deviation increase in the number
# of manufacturing centres is associated with a 1.55 standard deviation
# increase in the concentration of SO2 in the air, on average.
# The "standard deviation" thing is because the data were scaled.

# Diagnostics?
tibble(x = fitted(usair_lm1),y = residuals(usair_lm1)) %>%
  ggplot(aes(x = x,y = y)) +
  theme_light() +
  geom_point() +
  geom_hline(yintercept = 0,colour = "red",linetype = "dashed") +
  labs(title = "Residual plot, US Air pollution data, basic linear model",
       x = "Fitted Values",y = "Residuals")

# Hm... okay. What assumption specifically looks violated?
# - constant variance?
# not surprising since the data had a long right tail.
# Values far above the mean tend to vary more wildly.
# 
# 
# Confidence intervals for betas?
lr_ciplot <- tibble(beta = coef(usair_lm1),
       coef = names(coef(usair_lm1)),
       cilower = beta - 2 * sqrt(diag(vcov(usair_lm1))),
       ciupper = beta + 2 * sqrt(diag(vcov(usair_lm1)))) %>%
  ggplot(aes(x = coef,y = beta)) +
  theme_light() + 
  geom_errorbar(aes(ymin = cilower,ymax = ciupper),width = .1) +
  geom_point(pch = 21,colour = "black",fill = "orange") +
  coord_flip() +
  labs(x = "Coefficient",y = "Estimate and 95% CI")

lr_ciplot

# Does it make sense that pop has a large,
# negative beta?
# Cities with higher populations have drastically
# LESS air pollution on average?
# I mean, maybe, maybe SO2 is somehow inversely related to
# pop size? But it seems more likely that the problem
# is the correlation with manuf and pop.
# 
# Try an inference methodology that accounts for this and see what happens.

## Bayesian Linear Regression ##
# Let's do a Bayesian LR with INLA and compare the output.
# You don't have to know how to USE INLA.
# But you have to read the output.

usair_inla1 <- inla(
  SO2 ~ negtemp + manuf + pop + wind + precip + days,
  data = usair_scaled,
  family = "gaussian"
)
inla_summary_fixed <- as_tibble(usair_inla1$summary.fixed[ ,1:5])
inla_summary_fixed$variable <- rownames(usair_inla1$summary.fixed)
inla_summary_fixed

# Or use the summary() method:
summary(usair_inla1)
# ...good summary of the betas, including all the
# information you need to construct point estimates (mean)
# and 95% min-width credible intervals.
# The hyperparameter summary gives you precision instead of
# standard deviation ==> pffft. Bad.

# Why did the results look a lot like what we had before?
# I used the default prior for beta!
# The default in INLA is N(0,1000)
# This is a sensible default if you have a really complicated model
# and a lot of data.
# Neither of those things is true here.
usair_inla2 <- inla(
  SO2 ~ negtemp + manuf + pop + wind + precip + days,
  data = usair_scaled,
  family = "gaussian",
  control.fixed = list(prec = 1)
)
inla_summary_fixed2 <- as_tibble(usair_inla2$summary.fixed[ ,1:5])
inla_summary_fixed2$variable <- rownames(usair_inla2$summary.fixed)
inla_summary_fixed2

# Using a prior precision (inverse variance) of 1
# gave these results. Can you change it to get even more
# sensible results?
# E.g. sigma^2_beta = 10 ==> prec = .1
nrow(usair_scaled)
# - not very much data. The prior will have a big impact on the posterior.

# The estimated variance is called a "hyperparameter" in INLA's terminology...
# It was assigned an inverse gamma prior by default.
bri.hyperpar.summary(usair_inla1)
bri.hyperpar.plot(usair_inla1)
# I can read right off this summary that
# P(0.493 < sigma < 0.786) = 95%
# This is a 95% credible interval for sigma.
# Compare to the estimate for the standard deviation from the linear regression:
summary(usair_lm1)$sigma
# Pretty close! Biased towards zero, by design.

# Plot the posteriors and compare the credible intervals to the confidence intervals
# INLA returns the posterior marginal distributions as a list of (x,y) pairs
# so we get those then plot each of them.
# Don't worry sooooo much about the below plotting code... unless
# you're super interested.
plot_posterior <- function(lst,nm) {
  lst %>%
    as_tibble() %>%
    ggplot(aes(x = x,y = y)) +
    theme_light() +
    geom_line(colour = "purple") +
    labs(title = nm,x = "beta",y = "density")
}


posteriorplots <- purrr::map2(usair_inla1$marginals.fixed,
                              names(usair_inla1$marginals.fixed),
                              plot_posterior)


cowplot::plot_grid(plotlist = posteriorplots,nrow = 2)

# Cool. What about the interval comparison?
# Grab the point estimate (mean) and the 2.5%
# and 97.5% quantiles of the posteriors.
postmeans <- usair_inla1$summary.fixed[,1]
post2.5 <- usair_inla1$summary.fixed[,3]
post97.5 <- usair_inla1$summary.fixed[,5]

post_cr_plot <- tibble(beta = postmeans,
                    coef = rownames(usair_inla1$summary.fixed),
                    cilower = post2.5,
                    ciupper = post97.5) %>%
  ggplot(aes(x = coef,y = beta)) +
  theme_light() + 
  geom_errorbar(aes(ymin = cilower,ymax = ciupper),width = .1) +
  geom_point(pch = 21,colour = "black",fill = "orange") +
  coord_flip() +
  labs(x = "Coefficient",y = "Estimate and 95% CI")

cowplot::plot_grid(
  lr_ciplot + labs(title = "Frequentist"),
  post_cr_plot + labs(title = "Bayesian"),
  nrow = 2
)

# The frequentist and Bayesian intervals are almost
# identical, because our prior was too wide.
# Try changing the prior, and seeing how the Bayesian
# intervals change.

# What is the interpretation of a frequentist confidence interval?
# What is the interpretation of a Bayesian credible interval?
# Frequentist: there is a 95% probability that the interval captures the
# true parameter. The interval here is random, and the probability refers to a
# long run relative frequency. If I sampled the data many times and calculated
# many intervals, I expect about 95% of them to contain the true parameter.
# Bayesian: there is a 95% chance that the parameter is in the interval.
# Here, probability is a subjective/degree of belief: how much money would YOU
# bet on the parameter being in the interval?
