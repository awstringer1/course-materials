### Analyze the rat data using a Bayesian mixed model ###

library(tidyverse)
library(INLA)

# Patrick Brown's Pmisc package:
# install.packages("Pmisc", repos='http://r-forge.r-project.org')

# EXERCISE: before next lecture, try to write down
# the full hierarchical model being fit here.
# 
# Recall: rat data from lectures 7/8
data(rat.growth)
rat <- as_data_frame(rat.growth)
glimpse(rat)

# 5 weekly measurements on 30 rats (150 obs total)
# y is weight, units unknown (grams?)

rat %>%
  ggplot(aes(x = week,y = y,group = rat)) +
  theme_classic() + 
  geom_line(alpha = 0.3) +
  geom_line(data = rat %>% group_by(week) %>% summarize(y = mean(y)),
            aes(x = week,y = y,group = 1),
            colour = "red",
            size = 1) +
  labs(title = "Growth by Week, for each rat",
       subtitle = "Rat Growth Data",
       x = "Week",
       y = "Weight (Units unknown)")

# We fit a mixed model using REML:
ratmod1 <- lmer(y ~ 1+(1|rat)+week,data = rat)
summary(ratmod1)
# This gives point estimates for betas:
summary(ratmod1)$coefficients
# ...and predictions for each rat's random intercept:
ranef(ratmod1)$rat
# ...and estimates for the between and within rat standard deviations
# (hey I finally figured out how to get these from the fitted object!)
summary(ratmod1)$varcor

# We know how to read this output. But we had to learn.
# The INLA output is the same for every type of
# hierarchical model fit, no matter how complicated.
# Mixed models, longitudinal, spatial models... it's all the same output.
# 
# 
# Let's fit a model with the same linear predictor, but using Bayesian inference

# Because we're Bayesian here, we need a prior for all unknown quantities
# We already were doing this for U, by modelling it as Gaussian
# We also don't know sigma_U and sigma though
# A popular modern choice is the "PC Prior". This is a whole concept on its
# own that we won't cover; it amounts to, here, putting an Exponential prior
# on the standard deviations sigma_U and sigma. So you'll just have to believe me
# that this is a good thing to do.
# You choose the parameter of the prior for the standard deviation by making a
# statement like "I think there's a 50% chance that the between subject variability
# is > 5" or something. I'll go with that for illustration, but you choose
# different values and see how the results change!

# Here is the point where I am less expecting that you'll understand the INLA code.
# 
# INLA's syntax for formulas is odd. Memorize the following syntax for
# fitting a mixed model:
ratformula <- y ~ 1 + week + f(rat,
                               model = "iid", # "IID" means "random intercept"
                               prior = "pc.prec",
                               param = c(5,.5))
# Can you interpret this PC Prior?
# P(sigma_U > 5) = 50%
ratinlamodel <- inla(
  ratformula,
  data = rat,
  family = "gaussian",
  control.family = list(prior = "pc.prec",param = c(2,.5))
  # control.family is how you set a prior on sigma.
)
# Summarize: set a prior on sigma_U, do it inside the f()
# To set the prior on sigma, do it inside the control.family()
summary(ratinlamodel)

# Some notes about this output:
# - because we're looking at marginal posteriors (i.e. each parameter
# one at a time), the credible intervals that are output here are still
# correct, regardless of how complicated the rest of the model was.
# This is a MAJOR simplification over calculating confidence intervals in
# complicated models.

# Get the posterior for the standard deviations, not the precisions
Pmisc::priorPostSd(ratinlamodel)$summary
# Ooo... compare to the non-Bayesian one?
summary(ratmod1)$varcor
# Pretty close. HOWEVER. Some experimenting shows that these results are actually
# pretty sensitive to the choice of the priors for BOTH sigma_U and sigma. I recommend
# playing around with this to see what I mean. This is important! The tradeoff between
# the two types of variability (between/within) is a very important part of inference in
# these models.

# Check out some posteriors. First do the between-rat standard deviation
Pmisc::priorPostSd(ratinlamodel)$posterior %>%
  as_tibble() %>%
  ggplot(aes(x = x,y = y)) +
  theme_light() +
  geom_line() + 
  geom_line(aes(y = prior),colour = "red",linetype = "dashed") + 
  labs(title = "Posterior between-rat standard deviation",
       subtitle = "Red: prior. Black: posterior. Purple: REML",
       x = "Value",
       y = "Density") +
  scale_x_continuous(breaks = seq(2,60,by = 2)) +
  geom_vline(xintercept = attr(summary(ratmod1)$varcor$rat,"stddev"),colour = "purple",alpha = .5)

# Fixed effects regression coefficients
ratinterceptplot <- ratinlamodel$marginals.fixed$`(Intercept)` %>%
  as_tibble() %>%
  ggplot(aes(x = x,y = y)) +
  theme_light() +
  geom_line() +
  labs(title = "Posterior for global intercept",
       subtitle = "Purple: REML",
       x = "Intercept",
       y = "Density") +
  geom_vline(xintercept = summary(ratmod1)$coefficients[1,1],colour = "purple",alpha = .5)

ratweekplot <- ratinlamodel$marginals.fixed$week %>%
  as_tibble() %>%
  ggplot(aes(x = x,y = y)) +
  theme_light() +
  geom_line() +
  labs(title = "Posterior for week fixed effect",
       subtitle = "Purple: REML",
       x = "Intercept",
       y = "Density") +
  geom_vline(xintercept = summary(ratmod1)$coefficients[2,1],colour = "purple",alpha = .5)

cowplot::plot_grid(ratinterceptplot,ratweekplot,nrow = 1)

# Each rat's random intercept also gets a posterior:
# For example, rat 1:
ratinlamodel$marginals.random$rat$index.1 %>%
  as_tibble() %>%
  ggplot(aes(x = x,y = y)) +
  theme_light() +
  geom_line() +
  labs(title = "Posterior for Rat 1 random intercept",
       x = "Intercept",
       y = "Density")

# You could do this for all the rats.

# IMPORTANT POINT. INLA returns MARGINAL posteriors. 
# But in a mixed model, the predicted effect is Xbeta + U.
# This is a linear combination of latent variables.
# I can give you a credible interval for U
# and for Xbeta.
# But not for their sum (INLA actually CAN do this, but it's not done by default).
# So, if I ask you for a credible interval for beta_1, say, you can give it based on the
# INLA output.
# But if I ask you for a credible interval for, say, y_{12} = x_{12}^T beta + U_{1},
# there is NOT ENOUGH INFO in the INLA output to calculate this, by default.
# i.e. there is no analogue to vcov() for INLA model objects.

# EXERCISE:
# What are some advantages:
# ...
# and disadvantages:
# ...
# to the Bayesian approach?
# We will discuss next lecture.


