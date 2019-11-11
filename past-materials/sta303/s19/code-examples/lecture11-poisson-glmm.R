### Analysis of the nitrofen data ###
# Poisson Bayesian GLMM with INLA
# 
# This dataset will appear on your final exam.
# Reference: BRI section 5.6
# 
## SETUP ##
library(tidyverse)
library(cowplot)
library(INLA)
library(brinla)

# Patrick Brown's Pmisc package:
# install.packages("Pmisc", repos='http://r-forge.r-project.org')

## LOAD DATA ##
# Data are from Bayesian Regression Models with INLA by Faraway
# 50 zooplankton were split into 10 groups of 5 each, 
# and exposed to different concentrations of nitrofen. 
# Each animal then gave birth to three broods, 
# and the number of live offspring in each brood was recorded.

data(nitrofen,package = "boot")
glimpse(nitrofen)
# Brood is in three separate columns, 
# change from wide to long
fleaclean <- nitrofen %>%
  as_tibble() %>%
  dplyr::select(-total) %>%
  mutate(id = 1:n(),
         conc = conc/300) %>%
  gather(brood,live,brood1:brood3) %>%
  mutate(brood = str_replace_all(brood,"brood",""))

glimpse(fleaclean %>% arrange(id,brood))

## INITIAL DATA ANLYSIS ##
# Five unique concentrations each measured on 10 fleas:
fleaclean %>%
  group_by(conc) %>%
  summarize(numfleas = length(unique(id)))
# What's a reasonable guess at the random effect variance?
# Look at standard deviation of within-flea 
# sample means, a crude proxy:
fleaclean %>%
  group_by(id) %>%
  summarize(fleamean = mean(live)) %>%
  summarize(fleameanvar = var(fleamean),
            cilower = fleameanvar*(n() -1)/qchisq(.975,n()-1),
            ciupper = fleameanvar*(n() -1)/qchisq(.025,n()-1)
  ) %>%
  # Transform CI for variance into CI for standard deviation
  mutate_all(sqrt)

# What kind of plot should we do here? What's missing?
# The primary element of interest here is the effect of
# concentration on number of offspring. That's the research question.
# We want our model to account for correlation within unit,
# but this isn't the subject of primary inferential interest.
# So I'd like to do a boxplot, grouped by concentration.
fleaclean %>%
  mutate_at("conc",as.character) %>%
  ggplot(aes(x = conc,y = live)) +
  theme_light() +
  geom_boxplot()
# Add your own title and labels!

## Fit the model ## 
# Choose PC prior to give 75% chance 
# the random effect SD is > 3.
# Again, PLAY AROUND with these parameters and 
# see how the results change.
# Default prior precision on the betas is .001
# (so default variance = 1000).
nitro_formula <- live ~ conc * brood +
  f(id,
    model = "iid", # IID = random intercept.
    prior = "pc.prec", # PC Precision prior
    param = c(3,.75)) # 75% chance that sigma_U > 3.

nitro_inla <- inla(nitro_formula,
                   data = fleaclean,
                   family = "poisson")

## ANALYZE FITTED MODEL ##

summary(nitro_inla)

# Model contains fixed effects for conc, and brood, and their interaction.
# What is the fitted linear predictor?
# log(lambda_ij) = beta_0 + beta_1 x conc_ij +
#                  beta_2 x brood2_ij (indicator of whether the brood is brood 2)
#                  + ...
#                  (You fill in the blank)
#                  
# How to get the linear predictor for flea 1 in brood 1?
# Random intercept for flea 1:
flea1_intercept <- nitro_inla$summary.random$id %>%
  filter(ID == 1) %>% pull(mean)
flea1_intercept
# That's the marginal posterior mean
# for U_1
# Here is the coefficients:
nitro_beta <- nitro_inla$summary.fixed[,1]
nitro_beta
# That's the marginal posterior mean
# for each beta
# ...now you multiply this by flea 1's 
# covariate. INLA doesn't create the design matrix automatically
# (I don't think?) so gotta do it ourselves:
X <- model.matrix(live ~ conc * brood,data = fleaclean)

flea1pred_linkscale <- rbind(X[1, ]) %*% cbind(nitro_beta) + flea1_intercept
flea1pred_linkscale
# Now transform back to the natural scale:
exp(flea1pred_linkscale)
# So flea 1 is expected to have 3.7 offspring in brood 1
# Check this against the actual observed value:
fleaclean %>% filter(id == 1,brood == 1)
# 3. So not bad.
# The INLA software does have a prediction interface, but its use is out of scope here.
# It's important that you know how to do the calculation manually.
# And that you understand why you can't get a confidence interval for this prediction 
# directly from the default INLA output (again, it's possible to do this with the INLA
# software, just not using the default output.) 

# Check the posterior of the standard deviation:
bri.hyperpar.summary(nitro_inla)

# So the posterior mean random effect
# standard deviation is about .327
# and a 95% credible interval is
# (.219,.456).
# Do you have to transform this?
# No. The GLM/link function thing only affects
# the linear predictor. The variance parameter has the
# same interpretation and is on the same scale as 
# the ordinary mixed model.

# Plot posterior marginals
plot_marginals <- function(marg,plottitle = "") {
  # marg is one of the dataframes from nitro_inla$marginals.fixed
  marg %>%
    as_tibble() %>%
    ggplot(aes(x = x,y = y)) +
    theme_light() + 
    geom_line() +
    labs(x = "Value",
         y = "Marginal Density",
         title = plottitle)
}

purrr::map2(nitro_inla$marginals.fixed,
            names(nitro_inla$marginals.fixed),
            ~plot_marginals(marg = .x,plottitle = .y)) %>%
  cowplot::plot_grid(plotlist = .,nrow = 2)

# Right from these plots, you can read off
# the point estimates (posterior means) and
# the interval estimates (given by the 95% credible intervals).
# BUT! These are all marginal posteriors for betas.
# So they are on the LINK scale.
# You read the estimates off the plots, and then
# transform back to the natural scale.
# So, for example, for the conc effect, it looks like about
# -0.05 for the estimate and (-.75,.75) for
# the credible interval. So we would transform back
exp(-.05)
exp(c(-.75,.75))
# to get point and interval estimates for the marginal effect of concentration.

# Prior/posterior
Pmisc::priorPostSd(nitro_inla)$posterior %>%
  as_tibble() %>%
  ggplot(aes(x = x,y = y)) +
  theme_light() +
  geom_line() + 
  geom_line(aes(y = prior),colour = "red",linetype = "dashed") + 
  labs(title = "Posterior standard deviation of random effect",
       subtitle = "Red: prior. Black: posterior",
       x = "Value",
       y = "Density")


# The Poisson GLMM uses every single concept from this course.
# - Poisson generalized linear model with log link
# - Mixed model
# - Bayesian inference + hierarchical modelling.
# So, if you can understand this example as well as possible, you
# are fairly well prepared for the exam.
# But don't make the mistake of thinking that this is "too hard"
# to appear on the exam.
# It's hard.
# But it will (may?) appear on the exam.


# Can you get a credible interval for the response directly from the INLA output?
# No. The linked mean response is a linear combination of the latent variables.
# So its variance/distribution depends on the posterior covariance between
# latent variables (beta and U).
# INLA does NOT output this by default.
# You can ask for it, but it's a confusing interface and I didn't want to include it.
 
