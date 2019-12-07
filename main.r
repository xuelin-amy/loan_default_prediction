library(data.table)
library(dplyr)
library(summarytools)
library(ggplot2)
library(ggpubr)

#===== 1.load data =====
df <- fread('./data/train.csv')
head(df)

#===== 2.change data types =====
clean_data <- function(df){
  df %>%
    mutate(
      request_date = as.Date(request_date, format = '%d-%B-%y'),
      loan_amount = as.numeric(gsub('[$,]','', loan_amount)),
      insured_amount = as.numeric(gsub('[$,]','', insured_amount)))
}
df <- clean_data(df)
head(df)

#===== 3.statistic summary =====
dfSummary(df %>% select(-id), valid.col = F, graph.col = F) %>% print()

#===== 4.exploratory data analysis ======
# define theme for notebook display
options(repr.plot.width =2.5, repr.plot.height=1.5, repa.plot.res=300)
nb.theme <- theme(
  text = element_text(size = 3),
  element_line(size = 0.1),  
  legend.key.size = unit(0.2,'cm'),
  panel.background = element_rect(fill = "transparent"), # bg of the panel
  plot.background = element_rect(fill = "transparent"), # bg of the plot
  panel.grid.major = element_blank(), # get rid of major grid
  panel.grid.minor = element_blank(), # get rid of minor grid
  legend.background = element_rect(fill = "transparent", color = NA), # get rid of legend bg
  legend.box.background = element_rect(fill = "transparent", color = NA)) # get rid of legend panel

# 4.1 term histogram
df %>% 
  ggplot(aes(x= term, fill= factor(default_status))) +
  geom_histogram(bins=50) +
  facet_grid(~default_status) +
  labs(title='Term Distribution') +
  nb.theme

# 4.2 loan amount and insured amout histogram
p1 <- df %>% 
  ggplot(aes(x=loan_amount)) +
  geom_histogram(bins = 50, fill='lightblue') +
  labs(title = 'Loan Amount') +
  nb.theme
p2 <- df %>%
  ggplot(aes(x=insured_amount)) +
  geom_histogram(bins = 50, fill='lightblue') +
  labs(title = 'Insured Amount') +
  nb.theme
p3 <- p1 + scale_x_log10() + labs(title = 'Loan Amount (log scale)')
p4 <- p2 + scale_x_log10() + labs(title = 'Insured Amount (log scale)')
ggarrange(p1,p2,p3,p4, ncol = 2, nrow = 2)

# 4.3 scatter plot - insured amount vs. loan amount
df %>%
  ggplot(aes(x=loan_amount,y= insured_amount)) +
  geom_point(color='lightblue') +
  labs(title = 'Insured Amount vs. Loan Amount') +
  nb.theme

# 4.4 trend chart - default rate over time
df %>% 
  mutate(ym = format(request_date, '%Y-%m')) %>%
  group_by(ym) %>% 
  summarise(default_rate = mean(default_status)) %>%
  ggplot(aes(x=ym, y=default_rate)) +
  geom_line(stat = 'identity', color='lightblue', group = 1) +
  scale_y_continuous(labels = scales::percent) +
  labs(title = 'Default Rate over Time', x='Requested Date',y='Default Rate') +
  nb.theme

# 4.5 default rate over industry
df %>%
  group_by(industry) %>%
  summarise(default_rate= mean(default_status)) %>%
  ggplot(aes(x=reorder(industry, default_rate), y=default_rate)) +
  geom_bar(stat = 'identity', fill='lightblue') +
  scale_y_continuous(labels = scales::percent) +
  labs(title = 'Default Rate across Industry', x='Industry', y='Default Rate') +
  coord_flip() +
  nb.theme

# 4.6 default rate over new industry, business type
p1 <- df %>%
  group_by(business_new) %>%
  summarise(default_rate = mean(default_status)) %>%
  ggplot(aes(x=business_new, y=default_rate)) +
  geom_bar(stat = 'identity', fill='lightblue') +
  scale_y_continuous(labels=scales::percent) +
  labs(title = 'New Business Type', x='New Business',y='Default Rate') +
  coord_flip() +
  nb.theme

p2 <- df %>%
  group_by(business_type) %>%
  summarise(default_rate = mean(default_status)) %>%
  ggplot(aes(x = business_type, y = default_rate)) +
  geom_bar(stat = 'identity', fill = 'lightblue') +
  scale_y_continuous(labels = scales::percent) + 
  labs(title = 'Business Type Effect', x = 'Business Type', y = 'Default Rate') +
  coord_flip() + 
  nb.theme
ggarrange(p1,p2, ncol = 2)

# 4.7 loan amount effect on default rate
df %>% 
  ggplot(aes(x=log10(loan_amount), fill= factor(default_status))) +
  geom_density(alpha=0.5) +
  labs(title = 'Loan Amount effect on default') +
  nb.theme

#===== 5. Build Predictive Models =====

#combine train and valid before convert to factor
df$flag <- 'train'
df.test <- fread('./data/test.csv') %>% 
  clean_data() %>%
  mutate(default_status = 0, flag='test')
df.full <-rbind(df,df.test)

#creat features from request data
df.full <- df.full %>%
  mutate(request_ym = format(request_date, '%Y-%m'))

# init h2o library
library(h2o)
h2o.init(nthreads = 4, max_mem_size = '4g')

#convert data.table to h2o.frame
full.hex <- as.h2o(df.full)

#convert categorical variables to factor
full.hex$label <- h2o.asfactor(full.hex$default_status)
full.hex$industry <- h2o.asfactor(full.hex$industry)
full.hex$state <- h2o.asfactor(full.hex$state)
full.hex$business_new <- h2o.asfactor(full.hex$business_new)
full.hex$other_loans <- h2o.asfactor(full.hex$other_loans)
full.hex$request_ym <- h2o.asfactor(full.hex$request_ym)

#split train, test
train.hex <- full.hex[full.hex$flag == 'train',]
test.hex <- full.hex[full.hex$flag == 'test',]

#define feature set
feature.names <- c('industry','state','term','employee_count','request_ym','business_new', 
                   'business_type', 'other_loans','loan_amount','insured_amount')

# 5.1 Linear Model- Logistic Regression Baseline
model_ridge <- h2o.glm(x=feature.names, y='label', 
                       training_frame = train.hex, 
                       family = 'binomial',
                       nfolds = 5, alpha = 0, lambda_search = T)

# auc
h2o.auc(model_ridge, train=T, xval = T)

# accuracy at threshold = 0.5 for train and cv
model_ridge_perf_train <- h2o.performance(model_ridge, train = T)
model_ridge_perf_cv <- h2o.performance(model_ridge, xval = T)
acc <- c(h2o.accuracy(model_ridge_perf_train,thresholds = 0.5),
         h2o.accuracy(model_ridge_perf_cv,thresholds = 0.5))
names(acc) <- c('train','xval')
acc

h2o.varimp(model_ridge) %>% head(20)

# 5.2 Xgboost Model -Tree Based Ensemble Model
model_xgb <- h2o.xgboost(x=feature.names,y='label', 
                         training_frame = train.hex,
                         max_depth = 6, eta = 0.1, 
                         stopping_metric = 'AUC',stopping_rounds = 21, 
                         ntrees = 500, nfolds = 5)
#auc
h2o.auc(model_xgb, train = T, xval = T)

#accuracy at threshold = 0.5 for train and cv
model_xgb_perf_train <- h2o.performance(model_xgb, train = T)
model_xgb_perf_cv <- h2o.performance(model_xgb, xval = T)
acc <- c(h2o.accuracy(model_xgb_perf_train, thresholds = 0.5),
         h2o.accuracy(model_xgb_perf_cv, thresholds = 0.5))
names(acc) <- c('train','xval')
acc

h2o.varimp(model_xgb) %>% head(20)
h2o.shutdown(prompt = F)































