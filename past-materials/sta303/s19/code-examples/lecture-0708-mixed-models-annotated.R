# Supplementary R code for lectures 7 + 8, STA303 summer 2019
# Alex Stringer

library(tidyverse)
library(faraway)
library(lme4) # For mixed models
library(SMPracticals) # Has some good datasets

# Pulp data: experiment to test the paper brightness depending on a shift operator
# ?pulp
# Why use random effects? Because "operator"
# is a RANDOM SAMPLE from the population of
# possible operators of this machine.
# We want our analysis to generalize to NEW
# operators.
# So regard operators' intercepts as random.
# Another way of looking at it: we are more
# interested in the VARIANCE BETWEEN operators
# than we are in estimating exactly what each
# operator's mean is.

data(pulp)
pulp <- as_data_frame(pulp)
glimpse(pulp)

# 20 observations, how many operators and obs per operator? Plus brief summary
pulp %>%
  group_by(operator) %>%
  summarize(cnt = n(),
            meanbright = mean(bright),
            sdbright = sd(bright))
# Does it look like the operator means are
# different?

# Single continuous response and discrete factor- pairwise boxplots
pulp %>%
  ggplot(aes(x = operator,y = bright)) +
  theme_classic() + 
  geom_boxplot() +
  labs(title = "Brightness by Shift Operator",
       subtitle = "Pulp Data",
       x = "Operator",
       y = "Brightness")

# Wow! So the constant variance assumption doesn't look reasonable, mostly
# because of operator A (what are they even up to?)
# 
# Maximum Likelihood
# REML = TRUE is restricted ML, the default
# REML = FALSE gives regular ML (biased)
lmod_mixed_ml <- lmer(bright ~ 1 + (1|operator),data = pulp,REML = FALSE)
summary(lmod_mixed_ml)

# The 1 in the formula means beta0 (global intercept)
# The (1|operator) notation means a random intercept
# for operator.
# The overall (population-level) mean:
pulp %>% pull(bright) %>% mean()
# Why is it not the mean of the group means?
# Well...
pulp %>%
  group_by(operator) %>%
  summarize(mn = mean(bright),num = n()) %>%
  summarize(mn = mean(mn))

# It IS! Because there are the same number of
# observations in each group.
# This makes the MLE for beta0 exactly equal
# to the sample mean of all the observations.


# Random effects:
#   Groups   Name        Variance Std.Dev.
# operator (Intercept) 0.04575  0.2139  
# Residual             0.10625  0.3260

# Estimated sigma_u^2 = 0.04575
# Estimated sigma^2 = 0.10625
# You need to know how to read these
# numbers off the output and interpret them
# Do we have more between or within subject
# variability?
# Proportion of variance explained by each:
(0.04575/(0.04575 + 0.10625)) 
(0.10625/(0.04575 + 0.10625))
# About 30% between subject and 70% within subject.

# Variability due to operator?
.04575 / (.04575 + .10625)
# About 30%
# Total variance is sigma^2_b + sigma^2
# So proportion of variance due to operator
# is sigma^2_b / (sigma^2_b + sigma^2)

# What happens if we fit a fixed effects model?
pulp_fixed <- lm(bright ~ operator,data = pulp)
summary(pulp_fixed)

# Group means:
pulp %>%
  group_by(operator) %>%
  summarize(mn = mean(bright),num = n())
  
# Mean of operator A:
coef(pulp_fixed)[1]
# Mean of operator B:
coef(pulp_fixed)[1] + coef(pulp_fixed)[2]

# Look at the residual variance
summary(pulp_fixed)$sigma^2
# The residual variance from the fixed effects
# model is the same as the residual variance
# from the random effects model.
# The residual variance from an ordinary
# linear regression is unbiased.


# The fixed effect intercept is the grand mean
# The Residual variance is the same as that got from the MSE
# The residual variance is unbiased in ML.
# But the between subject variance...
# Fit with REML:
lmod_mixed_reml <- lmer(bright ~ 1 + (1|operator),data = pulp,REML = TRUE)
summary(lmod_mixed_reml)

# Between subject variance sigma^2_u:
# Before (ML): .04575
# Now (REML): 0.06808

# There we go. Notice fixed effects estimates unchanged but variance higher.
# Also notice this difference is different than what Faraway gets, because the package
# has changed (dramatically) in the 13 years since the book was published.

# Variability due to operator
.0681 / (.0681 + .1062)
# About 39%, much greater than in the ML model
.1062 / (.0681 + .1062)
# About 61%
# So using the unbiased estimation technique
# may have actually led to a different conclusion
# in this study.


### Rat growth data (SM page 459 example 9.18) ###
data(rat.growth)
rat <- as_data_frame(rat.growth)
glimpse(rat)

# 5 weekly measurements on 30 rats (150 obs total)
# y is weight, units unknown
# How to plot these data?

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

# The grey lines are each rat's growth
# The red line is the mean weight of all the
# rats at each time.
# So the red line is what we'd get if we ignored
# the grouping between rats
# But in mixed model, each rat gets their own line
# ==> better predictions!
# Also, the variances would be too low if we 
# ignored correlation between measurements on
# the same rat.
# You can eyeball...
# - beta0, beta(week)
# - U_i, i = 1..30
# - sigma^2_U
# but NOT sigma^2...
# ...because, the actual individual datapoints
# are not plotted.

# It looks like the actual amount of growth varies across rats (random intercept)
# but the growth rate/pattern does not (no random slope- later)
# 
# Can we graphically investigate whether the week itself induces variation?
# i.e. we COULD have done these measurements on ANY weeks, not
# just the ones we chose. Does it make sense to include week as
# a random effect?
# Conceptually, NO. We did not randomly choose weeks from some
# population of possible weeks. We chose 5 consecutive weeks.
rat %>%
  ggplot(aes(x = rat,y = y,group = week)) +
  theme_classic() + 
  geom_line() +
  labs(title = "Rat Growth by Rat, for each week",
       subtitle = "Rat Growth Data",
       x = "Rat",
       y = "Weight (Units unknown)")
# Be careful when interpreting a graph like this. The x axis is not ordered,
# so the lines don't mean anything.
# I think this plot is basically garbage.
# What DOES mean something is that the lines all have the same shape
# Better: pairwise boxplots for week
rat %>%
  mutate_at("week",as.factor) %>%
  ggplot(aes(x = week,y = y)) +
  theme_classic() + 
  geom_boxplot() +
  labs(title = "Pairwise boxplots of rat weights by week",
       subtitle = "Rat Growth Data",
       x = "Week",
       y = "Weight (Units unknown)")
# In this plot, the actual rat "number" is obscured.
# Which is good, because it doesn't mean anything.
# It looks like the differences in mean weight between
# each consecutive week are roughly the same.
# It looks like the variance is not changing across weeks.

# Model asumptions: normality of data?
# Usually we would fit the model and check normality of
# residuals, but here it's easier to just check the data
# because the residuals are now more complicated.

# This is how you do a QQ plot in scratch
# using ggplot...
rat %>%
  dplyr::select(y) %>%
  arrange(y) %>%
  mutate_at("y",funs( (. - mean(.)) / sd(.))) %>%
  mutate(q = qnorm(seq(1:nrow(rat)) / (1 + nrow(rat)))) %>%
  ggplot(aes(x = q,y = y)) +
  theme_classic() +
  geom_point() +
  geom_abline(slope = 1,intercept = 0,colour = "red") +
  labs(title = "Normal QQ Plot, Rat Growth",
       subtitle = "Evaluating normality of weight, rat growth data",
       x = "Theoretical Quantiles",
       y = "Sample Quantiles")

# Definitely some concern in the tails. We will have to proceed anyways.
# What do you think will happen because of this?
# - p-values won't be right. Because a p-value is a tail probability,
# and we use the normality assumption to calculate p-values.
# You NEED to know that the tails are the important part.
# 
# Model 1: random intercept for rat, fixed effect for week
# 1 + (1|rat) + week
ratmod1 <- lmer(y ~ 1+(1|rat)+week,data = rat)
summary(ratmod1)
# Did we use REML or ML? 
191.86 / (191.86 + 64.29) # % of variance due to rat
# 75%! Interpretation? Rats have very different birth weights
# on average, but similar growth across time.

# The "correlation of fixed effects" is scaled output
# of vcov(), and isn't new. You could get this in a regular
# lm(), it's just not printed by default.

# Sidebar: we should have considered week to be a discrete variable,
# not a continuous variable
# Because the labels don't mean anything
# But for illustration of LME models, we will proceed as-is.
# What does this assume about the relationship with size and week?


# Model 2: add a random slope for rat. Each rat has a different baseline weight,
# and grows as a different linear function of week.
# We didn't do this on the chalk board, because it's not that different.
# It corresponds EXACTLY to an INTERACTION between the fixed effect of week
# and the random effect of rat.
# 1+(1+week|rat)+week
# 1: global intercept
# week: fixed effect for week
# (1+week|rat): a random intercept U_i AND a random slope
# week_ij * U_i.
# 
ratmod2 <- lmer(y ~ 1+(1+week|rat)+week,data = rat)
summary(ratmod2)
# Within rat, we now have two terms.
# (Intercept): still the same as before
# week: variance due to week, within rat.
# Interpretation: how much do rats' growth patterns change on average
# from rat to rat?
# 
119.53 / (119.53 + 12.49 + 33.84) # 72%
12.49  / (119.53 + 12.49 + 33.84) # 7%

# We didn't gain much by adding in the random slope. Remember our
# phlosophy: fit the simplest possible model, but no simpler.
# It doesn't look like we NEED a model with random slopes.
# This agrees with what we thought based on looking at the plot
# To formally test this hypothesis is complicated (boundary problem,
# causes mathematical problems).
# Also makes no sense. A variance can't really be zero.
# Statistical significance is not really the question here.
# Remark: how can we have both a fixed and random effect for week? Does this
# make any sense?

# Test another model assumption: normality of the predicted random effects
# Remember, the normality of U is still an assumption,
# even though we essential invented U.
data_frame(b = ranef(ratmod1)$rat[ ,1]) %>%
  mutate_at("b",funs( (. - mean(.)) / sd(.))) %>%
  arrange(b) %>%
  mutate(q = qnorm(seq(1:nrow(ranef(ratmod1)$rat))/(1 + nrow(ranef(ratmod1)$rat)))) %>%
  ggplot(aes(x = q,y = b)) +
  theme_classic() +
  geom_point() +
  geom_abline(slope = 1,intercept = 0,colour = "red") +
  labs(title = "Normal QQ Plot, Predicted Random Intercepts",
       subtitle = "Evaluating normality of predicted random intercepts, rat growth data",
       x = "Theoretical Quantiles",
       y = "Sample Quantiles")

# Not bad!

# Illustration of lmer formula syntax, side-by-side

# Saw already: fixed intercept for week, random intercept for rat
ratmod1 <- lmer(y ~ 1+(1|rat)+as.factor(week),data = rat)
summary(ratmod1)

# Proportion of variance in growth that is attributable to rat?
194.46 / (194.46 + 51.29)
# About 79%

# Saw already: fixed intercept for week, random intercept for rat,
# and random rat x week slope
ratmod2 <- lmer(y ~ 1+(week|rat)+week,data = rat)
summary(ratmod2)

# What do we conclude? Some questions:
# - what does it mean that 75% of the variability in size is attributed to rat?
# - how do we interpret the point estimate for the week fixed effect?
# - how does this interpretation change if we include week as a random effect too?
# - what are the model assumptions? are they satisfied?

### Credit card data ###
# I simulated some credit card spending data that's kind of like
# stuff I would have seen at the bank.

# Create the credit dataset. I won't dwell too much on the below code...
# data processing stuff like this is important to know for your future
# industry/research activities but isn't part of the course content.
set.seed(8907)
credit <- tibble(
  id = sort(rep(seq(1:50),12)),
  month = rep(1:12,50)
)
# Subject-level covariate
subject_level <- tibble(
  id = 1:50,
  limit = sample(c(5000,10000,15000),50,replace=TRUE),
  subject_intercept = rnorm(50,0,1)
)
# Month-level covariate
month_level <- tibble(
  month = 1:12,
  holiday = c(0,1,0,0,0,0,1,1,0,0,0,1)
)
# Join them and generate response
credit <- credit %>%
  left_join(subject_level,by = "id") %>%
  left_join(month_level,by = "month") %>%
  mutate(log_spend = rnorm(50*12,
                           mean = log(50) + .1*log(limit) + 1*holiday + subject_intercept,
                           sd = log(5)),
         spend = exp(log_spend)
  ) %>%
  dplyr::select(-subject_intercept,-holiday) %>%
  mutate(month = as.character(lubridate::month(month,label = TRUE)))

glimpse(credit)

# How many subjects?
credit %>% pull(id) %>% unique() %>% length()
# How many months (lol)?
credit %>% pull(month) %>% unique() %>% length()
# Average spend per month:
credit %>% group_by(month) %>% summarize(avgspend = mean(spend),sdspend = sd(spend))
# Average spend for a few selected customers:
credit %>% 
  group_by(id) %>% 
  summarize(avgspend = mean(spend),sdspend = sd(spend)) %>%
  inner_join(tibble(id = sample(1:50,size = 10,replace = FALSE)),by = "id")

# Fit the model. 
# Rescale credit limit so it's on the 
# same scale as the others.
# Model the log of spend. Because spend
# is very right skewed (don't believe me? Make
# a plot!)
spendmodel <- lme4::lmer(
  log_spend ~ I(limit/5000) + month + (1|id),
  data = credit,
  REML = TRUE
)
summary(spendmodel)
# We included month as a categorical variable. Each month gets its own mean.
# Why wasn't it a random effect?
# 
# Take the following seriously:
# Could you write a short report explaining the above summary output?
# (hint hint)
# 
# Plot posterior expected spend
# coef(spendmodel) gives each subject's predicted intercept and the regression
# coefficients:
head(coef(spendmodel)$id)
# ...but I'm smart and will use the predict() method.
predvals <- credit %>%
  mutate(pred_log_spend = predict(spendmodel),
         pred_spend = exp(pred_log_spend))

# Plot predicted and actual, on log and natural scale
# This looks like a lot of code but it's repetitive, and you should be
# getting used to seeing this verbose ggplot() code.
predplot_log <- predvals %>%
  tidyr::gather(type,val,log_spend,pred_log_spend) %>%
  ggplot(aes(x = val,fill = type)) +
  theme_light() +
  geom_histogram(colour = "black",alpha = .8,bins = 50) +
  labs(title = "Observed vs Predicted Spend, Log Scale",
       x = "Log(spend)",
       y = "# customer - months",
       fill = "") +
  scale_fill_manual(labels = c("log_spend" = "Observed",
                               "pred_log_spend" = "Predicted"),
                    values = c("log_spend" = "lightgrey","pred_log_spend" = "darkgrey"))

predplot_natural <- predvals %>%
  tidyr::gather(type,val,spend,pred_spend) %>%
  filter(val < 2500) %>%
  ggplot(aes(x = val,fill = type)) +
  theme_light() +
  geom_histogram(colour = "black",alpha = .8,bins = 50) +
  labs(title = "Observed vs Predicted Spend, Natural Scale",
       x = "Spend",
       y = "# customer - months",
       fill = "") +
  scale_fill_manual(labels = c("spend" = "Observed",
                               "pred_spend" = "Predicted"),
                    values = c("spend" = "lightgrey","pred_spend" = "darkgrey"))

# Plot the predicted intercepts only
intercept_plot <- coef(spendmodel)$id %>%
  as_tibble() %>%
  dplyr::select(intercept = `(Intercept)`) %>%
  ggplot(aes(x = intercept)) +
  theme_light() +
  geom_histogram(bins = 15,colour = "black",fill = "darkgrey",alpha = .3) +
  labs(title = "Predicted customer intercepts",
       x = "Intercept",
       y = "# customers")

# Normal QQ plot of predicted intercepts
qqplot_intercepts <- coef(spendmodel)$id %>%
  as_tibble() %>%
  dplyr::select(intercept = `(Intercept)`) %>%
  arrange(intercept) %>%
  mutate_at("intercept",funs( (. - mean(.)) / sd(.))) %>%
  mutate(q = qnorm(seq(1:50) / (1 + 50))) %>%
  ggplot(aes(x = q,y = intercept)) +
  theme_light() +
  geom_point() +
  geom_abline(slope = 1,intercept = 0,linetype = "dashed",colour = "red") +
  labs(title = "Normal QQ Plot of predicted intercepts",
       x = "Theoretical Quantiles",
       y = "Observed Quantiles")

cowplot::plot_grid(
  predplot_log + guides(fill = FALSE) + theme(text = element_text(size = 8)),
  predplot_natural + theme(legend.position = c(.8,.8),text = element_text(size = 8)),
  intercept_plot + theme(text = element_text(size = 8)),
  qqplot_intercepts + theme(text = element_text(size = 8)),
  nrow = 2
)

# What do you think? Let's see some hands... :(
# Some points to note:
# - why are the predicted values lower on average than the observed values?
# - how do you interpret the customer intercepts?


### More repeated measures data: yearly income by age and gender in US ###
# Do this on your own time!
# 
# Yearly income, age, and gender information for American households
# from 1968 - 1990
data(psid)
psid <- as_data_frame(psid)
glimpse(psid)

# How many people, observations per person?
psid %>% pull(person) %>% unique() %>% length() # 85 households
psid %>%
  group_by(person) %>%
  summarize(nrows = n()) %>%
  group_by(nrows) %>%
  summarize(ntimes = n()) %>%
  ggplot(aes(x = nrows,y = ntimes)) +
  theme_classic() +
  geom_bar(stat="identity",colour = "black",fill = "lightblue") +
  labs(title = "Number of times each person appears in the dataset",
       subtitle = "PSID Data",
       x = "# of People",
       y = "# of Times") +
  scale_x_continuous(breaks = 11:23) +
  scale_y_continuous(breaks = seq(3,27,3))

# Let's look at 20 randomly selected people's yearly income for the period
# Make sure we select only people who have complete data (just for the purposes
# of this plot)
psid %>%
  inner_join(psid %>%
              group_by(person) %>%
              summarise(nrows = n()) %>%
              filter(nrows == 23) %>%
              sample_n(20),
            by = "person") %>%
  ggplot(aes(x = year,y = income,colour = sex,group = person)) +
  theme_classic() +
  facet_wrap(~person) +
  geom_line() +
  labs(title = "Income, 1978 - 1990, individual people",
       subtitle = "PSID Data",
       x = "Year",
       y = "Income",
       colour = "Gender") +
  scale_y_continuous(labels = scales::dollar_format())

# Faraway indicates that one goal of analysis is to compare incomes for
# males and females
psid %>%
  ggplot(aes(x = year,y = income,group = person)) +
  theme_classic() +
  facet_grid(~sex) +
  geom_line() +
  labs(title = "Individuals' Income by Year, for each gender",
       subtitle = "PSID Data",
       x = "Year",
       y = "Income") +
  scale_y_continuous(labels = scales::dollar_format())


# Interesting...  next let's do income in 1968 vs income in 1990
# by age, for each gender.
psid %>%
  filter(year == 68 | year == 90) %>%
  mutate_at("year",as.factor) %>%
  ggplot(aes(x = age,y = income,fill = year)) +
  theme_classic() +
  facet_grid(~sex) +
  geom_bar(stat="identity",position = "dodge") +
  labs(title = "Income by age, gender, and year",
       subtitle = "PSID Data",
       x = "Age",
       y = "Income") +
  scale_y_continuous(labels = scales::dollar_format())

# What does it look like the relationship between income and gender is?

# Now build a model
# Person is definitely a random effect
# Gender: fixed (not a random sample from population of all genders)
# Age: also fixed, not a random sample from population of possible ages
# Year: again, fixed
# May want to try year x age, gender x age and year x gender interactions
# First, a basic model
psid_lmer1 <- lmer(income ~ year + (1|person),data = psid)
summary(psid_lmer1)

114750124 / (114750124 + 70863916) # 62% of the variability in income is person-level

# Those variances are huge! Remember the units of the data though.
# This is what happens if you don't scale!!!
# You can try this analysis yourself scaling the data.
# 
# Faraway suggests a log transformation of income. This seems reasonable since
# income is strictly positive; let's make a quick plot though to see what values
# are actually in the data
psid %>%
  ggplot(aes(x = income)) +
  theme_classic() + 
  geom_histogram(colour = "black",fill = "#ff9933") +
  labs(title = "Histogram of Income",
       subtitle = "PSID Data",
       x = "Income",
       y = "# People") +
  scale_x_continuous(labels = scales::dollar_format())

# Definitely a large right skew, common in strictly positive data, and expected
# based on our rough socio-economic understanding of income dynamics (which for me can
# be summarized as "some people make a lot but most of us don't")
# Log transformation? Check if there are any zeroes
psid %>%
  filter(income == 0)
# No. What is lowest income?
psid %>% pull(income) %>% summary()
# Min income is 3... take a look at the bottom 100
psid %>%
  arrange(income) %>%
  slice(1:100) %>%
  View()

# It doesn't look like an outlier; there are just some really tiny incomes in
# the data. I want to remove these because I think people that make less than 1,000
# per year are not representative of American households.
# The decision to remove values from a dataset is ALWAYS subjective and must be
# justified in real applications.
# Try the transformation

psidfiltered <- psid %>% filter(income >= 1000)

psidfiltered %>%
  mutate_at("income",log) %>%
  ggplot(aes(x = income)) +
  theme_classic() + 
  geom_histogram(colour = "black",fill = "#ff9933") +
  labs(title = "Histogram of log(Income)",
       subtitle = "PSID Data",
       x = "log(Income)",
       y = "# People")

# Well that seems better. Check the qqplots...
psid_qq1 <- psidfiltered %>% # This one is not transformed
  filter(income >= 1000) %>%
  arrange(income) %>%
  mutate_at("income",funs( (. - mean(.)) / sd(.))) %>%
  mutate(q = qnorm(seq(1:nrow(psidfiltered)) / (1 + nrow(psidfiltered)))) %>%
  ggplot(aes(x = q,y = income)) +
  theme_classic() +
  geom_point() + 
  geom_abline(slope = 1,intercept = 0,colour = "red") +
  labs(title = "Normal QQ-Plot for un-transformed income",
       subtitle = "PSID Data",
       x = "Theoretical Quantiles",
       y = "Sample Quantiles")

psid_qq2 <- psidfiltered %>% # This one is transformed.
  mutate_at("income",log) %>%
  arrange(income) %>%
  mutate_at("income",funs( (. - mean(.)) / sd(.))) %>%
  mutate(q = qnorm(seq(1:nrow(psidfiltered)) / (1 + nrow(psidfiltered)))) %>%
  ggplot(aes(x = q,y = income)) +
  theme_classic() +
  geom_point() + 
  geom_abline(slope = 1,intercept = 0,colour = "red") +
  labs(title = "Normal QQ-Plot for log-transformed income",
       subtitle = "PSID Data",
       x = "Theoretical Quantiles",
       y = "Sample Quantiles")

cowplot::plot_grid(psid_qq1,psid_qq2)

# OH YEAH.
# Well... the second is clearly better than the first, but that long left
# tail is troubling.
# Anyways...

psid_lmer2 <- lmer(log(income) ~ year + (1|person),data = psidfiltered)
summary(psid_lmer2)

.5121 / (.5121 + .2224) # 69.7%
# How do you interpret this number?

# Here's the predicted regression line for each person:
as_data_frame(coef(psid_lmer2)$person)

# Predicted and actual
# Average income over the whole period
pred_avg_income <- as_data_frame(coef(psid_lmer2)$person) %>%
  bind_cols(psidfiltered %>% 
              group_by(person) %>% 
              summarize(income = mean(log(income)),
                        minyear = min(year),
                        maxyear = max(year))) %>%
  rename(intercept = `(Intercept)`) %>%
  mutate(predincome = intercept + year * (minyear + (maxyear - minyear)/2))
pred_avg_income

pred_avg_income %>%
  ggplot(aes(x = predincome,y = income)) +
  theme_classic() +
  geom_point(alpha = 0.3) +
  geom_abline(slope = 1,intercept = 0,colour = "orange") +
  labs(title = "Predicted vs Actual Log Income",
       subtitle = "PSID Data, Linear Mixed Effects Model 1",
       x = "Predicted",
       y = "Actual")

# This is common: LME are usually pretty fantastic for prediction, because they accurately
# harness trended data at the individual level. People's pasts are generally pretty predictive
# of their futures.

# Yearly Income- harder
pred_yearly_income <- pred_avg_income %>%
  rename(avgpredincome = predincome,
         avgincome = income,
         slopeyear = year) %>%
  left_join(psid,by = "person") %>%
  mutate(income = log(income),
         predincome = intercept + year * slopeyear)
pred_yearly_income

# Randomly select 20 people and plot their predicted vs actual
set.seed(3247)
people_range <- pred_yearly_income %>% pull(person) %>% unique()
pred_yearly_income %>%
  inner_join(data_frame(person = sample(people_range,20))) %>%
  ggplot(aes(x = year,y = income,group = person)) +
  theme_classic() +
  facet_wrap(~person) +
  geom_line(colour = "#ff9933") +
  geom_line(aes(y = predincome),colour = "#33ccff") +
  labs(title = "Predicted vs Actual Yearly Log Income",
       subtitle = "PSID Data, Linear Mixed Effects Model. Orange = Actual, Blue = Predicted",
       x = "Year",
       y = "Log(Income)")

# Notice anything interesting about the slopes of the lines...?


# Try a model with additional fixed effects
psid_lmer3 <- lmer(log(income) ~ year + age + sex + (1|person),data = psidfiltered)
summary(psid_lmer3)

exp(.879) # 2.41
# The data suggests that a man is expected to make 2.41 times as much as a woman on average
# if they are they same age and it's the same year
# How much of the variability in income is explained by individual variability?
.3733 / (.3733 + .5561) # 40%.

# Does that effect change across ages, or years?

psid_lmer4 <- lmer(log(income) ~ year * sex + age * sex + (1|person),data = psid)
summary(psid_lmer4)

# Older men can be expected to make more than older women (the wage gap increases with age)
# The effect is decreasing, though very slightly, with time (that's good!)
# Incomes are going up on average for everybody
# Note that the large correlation between the fixed effects estimates suggests that
# the actual point estimates for these coefficents is not very precise. In particular,
# look at the standard error for the sexM point estimate- even though it is 1.36, the standard
# error of 1.11 suggests anything between -0.86 and 3.58 is reasonable given the data.
# So I wouldn't put too much trust in this.


# What about a model where people's rate of income growth over time can be different?
# "Random slope"

psid_lmer5 <- lmer(log(income) ~ year + (year|person),data = psidfiltered)
summary(psid_lmer5)
# WOAH. Those intercepts got a LOT more variable. Why?

8.423 / (8.423 + .170) # 98.0%
# That's nuts!
# But look at how correlated the intercept and slope are.

# Here's the predicted regression line for each person:
as_data_frame(coef(psid_lmer5)$person) # Now they're different

# Predicted and actual
# Average income over the whole period
pred_avg_income <- as_data_frame(coef(psid_lmer5)$person) %>%
  bind_cols(psidfiltered %>% 
              group_by(person) %>% 
              summarize(income = mean(log(income)),
                        minyear = min(year),
                        maxyear = max(year))) %>%
  rename(intercept = `(Intercept)`) %>%
  mutate(predincome = intercept + year * (minyear + (maxyear - minyear)/2))
pred_avg_income

pred_avg_income %>%
  ggplot(aes(x = predincome,y = income)) +
  theme_classic() +
  geom_point(alpha = 0.3) +
  geom_abline(slope = 1,intercept = 0,colour = "orange") +
  labs(title = "Predicted vs Actual Log Income",
       subtitle = "PSID Data, Linear Mixed Effects Model 5",
       x = "Predicted",
       y = "Actual")


# Yearly Income- now the slopes should be different
pred_yearly_income <- pred_avg_income %>%
  rename(avgpredincome = predincome,
         avgincome = income,
         slopeyear = year) %>%
  left_join(psid,by = "person") %>%
  mutate(income = log(income),
         predincome = intercept + year * slopeyear)
pred_yearly_income

# Randomly select 20 people and plot their predicted vs actual
set.seed(3247)
people_range <- pred_yearly_income %>% pull(person) %>% unique()
pred_yearly_income %>%
  inner_join(data_frame(person = sample(people_range,20))) %>%
  ggplot(aes(x = year,y = income,group = person)) +
  theme_classic() +
  facet_wrap(~person) +
  geom_line(colour = "#ff9933") +
  geom_line(aes(y = predincome),colour = "#33ccff") +
  labs(title = "Predicted vs Actual Yearly Log Income",
       subtitle = "PSID Data, Linear Mixed Effects Model. Orange = Actual, Blue = Predicted",
       x = "Year",
       y = "Log(Income)")

# What do we conclude?
