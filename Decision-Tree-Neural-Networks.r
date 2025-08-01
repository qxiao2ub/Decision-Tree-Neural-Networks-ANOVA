---
date: "`r Sys.Date()`"
output:
  pdf_document: default
  markdowntemplates::skeleton: default
footer:
- content: '[link1](http://example.com/) • [link2](http://example.com/)<br/>'
- content: Copyright blah blah
navlink: '[NAVTITLE](http://NAVLINK/)'
og:
  type: article
  title: opengraph title
  url: optional opengraph url
  image: optional opengraph image link
---

## I. Introduction 

---
title: "Project Documentary"
output: word_document
bibliography: refs.bib
---

```{r refs, include=FALSE}
db <- bibentry(
   bibtype = "Manual",
   title = "Ames Iowa: Alternative to the Boston Housing Data Set",
   author = person("AmesHousing"),
   organization = "Ames, Iowa Assessor’s Office",
   address = "Ames, Iowa Assessor’s Office",
   year = 2010,
   url = "http://jse.amstat.org/v19n3/decock/DataDocumentation.txt",
   key = "Data")
   
db1 <- bibentry(  
   bibtype = "Manual",
   title = "House Prices - Advanced Regression Techniques",
   author = person("kaggle"),
   organization = "kaggle.com",
   address = "kaggle.com",
   year = 2016,
   url = "https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/rules",
   key = "kaggle")

f <- function(cond, ref) if(cond) cite(ref, db) else invisible()
```

This project utilizes the Ames, Iowa dataset as an alternative to the Boston
Housing Data Set for data cleaning, sorting, organization, and analysis tasks. 
The Ames dataset comprises information from the Ames Assessor's Office, which 
is used to compute assessed values for individual residential properties sold 
in Ames, IA, from 2006 to 2010 `r f(TRUE, "Data")`. It contains 2,930 
observations with a total of 82 variables. In this project, we apply various 
analytical methods introduced in this course, including linear regression, 
decision tree analysis, neural networks, and analysis of variance (ANOVA), 
among others. The following sections provide a detailed explanation of the 
project's purpose, the training methods used, and the development pathway for 
the regression model.

The objective of this project is to develop a predictive model using the Ames, 
Iowa dataset, leveraging explanatory variables or predictors to estimate the 
selling price of a house. The response variable, "SalePrice," is continuous, 
and the model aims to help typical buyers or real estate agents estimate the 
selling price of a property. By analyzing these predictors, our model can
provide a reliable estimate for the sale price based on various 
factors (kaggle 2016).


```{r}
#Data Preparations
#install.packages("visdat")
library(readr)
library(visdat)
# Checking for NA, or missing data using graphics 
ames_data <- read_csv("ames.csv")
```

```{r}
vis_miss(ames_data)
```

```{r}
str(ames_data)
```




## II. Description of the data and quality 

Based on the data description, we treat the house price ("SalePrice") as the response variable, with all other available decision variables as predictors. To better understand the data, we first analyze the correlation relationships among these variables. The dataset contains a mixture of numeric (continuous) and categorical variables, with roughly half falling into each category. 

To prepare the data for analysis, we perform a series of data cleaning steps. This includes converting categorical variables to a numeric format where necessary and removing irrelevant or non-applicable data points. These preparatory steps are crucial for ensuring that the model-building process is accurate and robust. Following the data cleaning process, we will proceed with an in-depth analysis to determine which variables are most strongly associated with "SalePrice."

```{r}
ames_clean <- read_csv("ames_1.csv")
```
```{r}
ncol(ames_clean)
```

And we can get the correlation matrix model as below.

```{r}
round(cor(ames_clean),2)
```

```{r}
cor(ames_clean)
```

After doing the transformation for the categorical data, we have retrieved the correlation matrix among all factors shown above.




## III. Model Development Process 

In this process, we're building a regression model to predict prices. To start, we create a training dataset that consists of 70% of our data, using set.seed(1023) for reproducibility. The remaining 30% of the data will be used as the test set. At this stage, we examine the data, checking for categorical variables that may need to be combined or dropped. For now, we keep the data in its original form and will decide which variables to drop during the data processing phase.

```{r TS2}
set.seed(1023)
# train test split
n = nrow(ames_clean)
train_size = round(n*.7)

# split data
train_index = sample(n, train_size)
train_data = ames_clean[train_index, ]
test_data = ames_clean[-train_index, ]

if (nrow(train_data) + nrow(test_data) == nrow(ames_clean)) {
  print("train and test sets are good to go.")
} else {
  print("train and test data set sizes are wrong...")
}

```

Now that the data is split into train and test sets, we can clean the data to deal with any missing values.

```{r TS2 note}
train_columns = colnames(train_data)
na_values_train = c()
ncol(train_data)
for (i in 1:ncol(train_data)) {
  print(sprintf("%s: %i out of %i", train_columns[i], sum(is.na(train_data[i])), nrow(train_data)))
  na_values_train = append(na_values_train, sum(is.na(train_data[i])))
}

```

```{r TS3}
columns = colnames(test_data)
na_values_train = c()
ncol(train_data)
for (i in 1:ncol(train_data)) {
  print(sprintf("%s: %i out of %i", columns[i], sum(is.na(train_data[i])), nrow(train_data)))
  na_values_train = append(na_values_train, sum(is.na(train_data[i])))
}

```

First, we drop the "Alley," "Pool.QC," "Fence," "Fireplace.Qu," and "Misc.Feature" columns because they contain too many missing values, making them impractical for data analysis. After that, we perform linear regression per the training data.


```{r SX1 QX2}

#train_data_nonempty <- train_data[!apply(is.na(train_data) | train_data == "", 1, all),]
#Drop the Alley column as it has so many NAs 
train_data_nonempty=subset(train_data, select=-c(Alley,Pool.QC,Fence,Misc.Feature,Fireplace.Qu))
# remove the NA data
train_data_nonempty_1=na.omit(train_data_nonempty)
# do the regression
ames_model<-lm(price~., data=train_data_nonempty_1)
summary(ames_model)

```

The above steps allow us to create a linear regression model using the training dataset.

## IV. Model Performance Testing

We start by cleaning the testing dataset, removing columns with excessive missing values, similar to the process used for the training dataset.

```{r QX1-test data}
#Drop the columns as it has so many NAs 
test_data_nonempty=subset(test_data, select=-c(Alley,Pool.QC,Fence,Misc.Feature,Fireplace.Qu))
# remove the NA data
test_data_nonempty_1=na.omit(test_data_nonempty)
```

Next, we use a stepwise bidirectional selection method to identify the optimal multiple linear regression models.

```{r QX1-olsrr}
library(olsrr)
```

We first employ transformation techniques to determine if any transformations are needed. Additionally, we assess the data for unequal variances and multicollinearity.


```{r QX1-4plots}
par(mfrow=c(2,2))
plot(ames_model)
```

The R-squared value is 88%, indicating a strong correlation. However, the QQ plot shows a heavy tail, and the Box-Cox transformation suggests that a transformation might be beneficial.

```{r QX1-transform}
library(MASS)
par(mfrow=c(1,1))
boxcox(ames_model,lambda=seq(-1,1,0.1))
a<-boxcox(ames_model,lambda=seq(-1,1,0.1))
```

```{r QX1-maxlikelihood}
a$x[which.max(a$y)]
```

The maximum likelihood estimate for lambda is approximately 0.15, suggesting that a transformation using \(Y^{0.15}\) could be appropriate.

```{r QX1-transformation}
f1<-lm((price)^{0.15} ~ .,data=test_data_nonempty_1)
summary(f1)
```

```{r QX1-after4plots}
par(mfrow=c(2,2))
plot(f1)
```

After applying the transformation, the QQ plot shows that the distribution is approximately normal. There's no indication of unequal variances or influential observations. Additionally, the R-squared value increases from 88% to 91%.

Next, we perform a stepwise bidirectional selection method to identify the optimal multiple linear regression models.

```{r QX1-stepwise}
k5<-ols_step_both_p(ames_model,prem=0.05,details=FALSE)
k5$model
```

```{r QX1-stepwisestep2}
f2<-lm(price^{0.15}~ Overall.Qual+BsmtFin.SF.1+Exter.Qual+MS.SubClass+
         Garage.Cars+Kitchen.Qual+Bsmt.Exposure+Mas.Vnr.Area+Kitchen.AbvGr+
         Mas.Vnr.Type+Year.Built+Overall.Cond+Roof.Style+Bsmt.Full.Bath+
         Exterior.1st+Bsmt.Qual+Functional+Screen.Porch+X1st.Flr.SF+
         Wood.Deck.SF+Street+Fireplaces+PID+Pool.Area+Sale.Condition+Land.Slope
       +X2nd.Flr.SF+Bsmt.Unf.SF+Open.Porch.SF+House.Style+Bldg.Type+
         Bedroom.AbvGr+TotRms.AbvGrd+Enclosed.Porch+Heating+Lot.Frontage,
       data=test_data_nonempty_1)
summary(f2)
```

```{r QX1-4plotsstepwise}
par(mfrow=c(2,2))
plot(f2)
```

```{r QX1-cookdistance}
# Cooks distance plot
ols_plot_cooksd_bar(f2)
```

```{r QX1-Unequal Variances}
# Unequal Variances
ols_test_breusch_pagan(f2)
```

By above Breusch Pagan, we reject null hypothesis and indicate the variance is not constant.

```{r QX1-farawaypackage}
# Collinearity Diagnostics
library(MASS)
library(faraway)
```

```{r QX1-viftest}
vif(f2)
```

```{r QX1-Huberrobustregression}
# Huber robust regression method on the same variables selected above
stack.huber <- rlm(price^{0.15}~ Overall.Qual+BsmtFin.SF.1+Exter.Qual+
                     MS.SubClass+Garage.Cars+Kitchen.Qual+Bsmt.Exposure+
                     Mas.Vnr.Area+Kitchen.AbvGr+Mas.Vnr.Type+Year.Built+
                     Overall.Cond+Roof.Style+Bsmt.Full.Bath+Exterior.1st+
                     Bsmt.Qual+Functional+Screen.Porch+X1st.Flr.SF+
                     Wood.Deck.SF+Street+Fireplaces+PID+Pool.Area+
                     Sale.Condition+Land.Slope+X2nd.Flr.SF+Bsmt.Unf.SF+
                     Open.Porch.SF+House.Style+Bldg.Type+Bedroom.AbvGr+
                     TotRms.AbvGrd+Enclosed.Porch+Heating+Lot.Frontage
                   ,data=test_data_nonempty_1)
summary(stack.huber)
```

```{r QX1-checkimpactofoutlier}
#check impact of outliers by comparing coefficients
coeff1<-summary(f2)$coefficients[,1]
coeff2<-summary(stack.huber)$coefficients[,1]
coeffs<-cbind(coeff1,coeff2)
coeffs
```

```{r QX1-caretpackage}
library(caret)
```

```{r QX1-predictusingtest}
PredictedTest<-(predict(f2,test_data_nonempty_1))^{(1/.15)}
ModelTest2<-data.frame(obs = test_data_nonempty_1$price, pred=PredictedTest)
defaultSummary(ModelTest2)
```

From the above we can see the for the testing dataset R-squared is high as 92.3% as well as RMSE and MAE is low, so it is reliable using stepwise both ways selection method for the best multiple linear models builtup.

```{r Qingyang1-predictusingtrain}
PredictedTest1<-(predict(f2,train_data_nonempty_1))^{(1/.15)}
ModelTest3<-data.frame(obs = train_data_nonempty_1$price, pred=PredictedTest1)
defaultSummary(ModelTest3)
```

The results from the training data set indicate that the RMSE is 70,406, while the R-squared value is 57.4% lower compared to the testing data. Furthermore, the MAE is also higher, by 17,988, than that of the testing data.

## V. Challenger Models 

To establish the regression framework, we initially construct a decision tree model, taking into account all relevant explanatory variables as predictors. The constructed decision tree model is then applied to generate forecasts, which are subsequently compared with the results from the test data set.

```{r QX1-Challenger Models1}
library(caret)
library(rpart)
library(rpart.plot)

tree_model <- rpart(price ~ ., data = train_data_nonempty_1)

rpart.plot(tree_model, digits = 3)

rpart.plot(tree_model, digits = 4, fallen.leaves = TRUE,type = 3, extra = 101)

## View the splits of the tree and the cptable
tree_model
tree_model$cptable

plotcp(tree_model)

#regression tree on train data
PredictedTrain<-predict(tree_model,train_data_nonempty_1)
ModelTrain<-data.frame(obs = train_data_nonempty_1$price, pred=PredictedTrain)
defaultSummary(ModelTrain)

#regression tree on test data
PredictedTest<-predict(tree_model,test_data_nonempty_1)
ModelTest<-data.frame(obs = test_data_nonempty_1$price, pred=PredictedTest)
defaultSummary(ModelTest)


```

We also implement a cross-validation procedure to evaluate the performance of the trained decision tree model. The final $R^2$ is $62\%$.

```{r QX1-Challenger Models2}
# Define the training control for cross-validation
train_control <- trainControl(method = "cv", number = 10)  

# Train the regression tree model using cross-validation
tree_model_cv <- train(price ~ ., data = train_data_nonempty_1, method = "rpart", trControl = train_control)

# View the cross-validated model results
print(tree_model_cv)

#get the RMSE of the cross-validated model
cv_rmse <- sqrt(tree_model_cv$results$RMSE)
cv_rmse

# Get the R-squared value of the cross-validated model
cv_r_squared <- summary(tree_model_cv)$r.squared
cv_r_squared

```

The analysis of variable importance reveals that Overall.Qual is the most significant predictor, while Bsmt.Qual, Garage.Cars, Garage.Area, Total.Bsmt.SF, and Kitchen.Qual exhibit moderate importance. Conversely, Full.Bath, Year.Built, Area, Garage.Yr.Blt, and Foundation show relatively low importance.

## VI. Model Limitation and Assumptions

Next, we employ a neural network (NN) model for regression analysis, incorporating two hidden layers with 4 neurons in the first layer and 6 neurons in the second layer. This model is used to predict house prices on the training data set, and its performance is evaluated on the testing data set.

```{r QY1-NN training}
library(neuralnet)
library(rpart.plot)
library(rpart)
library(caret)
library(MASS)
library(faraway)
library(olsrr)
set.seed(1023)
normalize <- function(x) {return((x - min(x)) / (max(x) - min(x)))}
dat=rbind(train_data_nonempty_1,test_data_nonempty_1)
q1_norm <- as.data.frame(lapply(dat, normalize))
q1_norm_train <- q1_norm[train_index,]
q1_norm_test <- q1_norm[-train_index,]
set.seed(1023)
q1_norm_train_1=na.omit(q1_norm_train)
q1_norm_train_2=q1_norm_train_1[1:1557,]
q1_norm_test_1=na.omit(q1_norm_test)
q1_norm_test_2=rbind(q1_norm_test_1,q1_norm_test_1)
q1_norm_test_3=q1_norm_test_2[1:661,]
q1_nn<- neuralnet(price ~ ., data = q1_norm_train_2, hidden =c(4, 6))
plot(q1_nn)
```

```{r QY1-NNTrainTest}
set.seed(1023)
#performance on train data set
model_results = neuralnet::compute(q1_nn, q1_norm_train_2[-1])
predicted_y <- model_results$net.result
unnormalize <- function(x) {return((x * (max(dat$price)) -min(dat$price)) + min(dat$price))}

#transforming back to training original scale
pred_new <- unnormalize(predicted_y)
PredictedTest<-pred_new
ModelTest1<-data.frame(obs = train_data_nonempty_1$price, pred=PredictedTest)
defaultSummary(ModelTest1)
```
```{r QY1-NNTestDataPerformance}
set.seed(1023)
#performance on test data set
model_results <- neuralnet::compute(q1_nn, q1_norm_test_3[-1])
predicted_y <- model_results$net.result
#transforming back to testing original scale
pred_new <- unnormalize(predicted_y)
PredictedTest<-pred_new
ModelTest1<-data.frame(obs = test_data_nonempty_1$price, pred=PredictedTest)
defaultSummary(ModelTest1)
```

Given that our neural network (NN) model employs only two layers—one with 4 neurons and the other with 6 neurons—the prediction accuracy may be limited. This configuration might not capture complex patterns in the data effectively, leading to lower predictive performance.

Following we employ Lasso regression to predict the target variable house price on the training data set, while evaluating the performance of Ridge regression on the test data set.

```{r QY1}
library(glmnet)
```

```{r QY1-Lasso}
#this delete 4th column i.e., price
x <- model.matrix(price~., train_data_nonempty_1)[,-c(4)] 
y <- train_data_nonempty_1$price
LassoMod <- glmnet(x, y, alpha=1, nlambda=100,lambda.min.ratio=0.0001)
plot(LassoMod,xvar="norm",label=TRUE)
```

```{r QY1-plotLasso}
CvLassoMod <- cv.glmnet(x, y, alpha=1, nlambda=100,lambda.min.ratio=0.0001)
plot(CvLassoMod)
```

```{r QY1-LassoLambda}
best.lambda.lasso <- CvLassoMod$lambda.min
best.lambda.lasso
```

```{r QY1-LassoLambdaMin}
coef(CvLassoMod, s = "lambda.min")
```

```{r QY1-Lasso performance on train data}
#Lasso performance on train data
PredictedTest<-predict(LassoMod, s = best.lambda.lasso, newx = x)
ModelTest2<-data.frame(obs = train_data_nonempty_1$price, pred=c(PredictedTest))
defaultSummary(ModelTest2)
```

```{r QY1-Lasso performance on test data}
#Lasso performance on test data
#this delete test 4th column i.e., Price
x_test <- model.matrix(test_data_nonempty_1$price~., test_data_nonempty_1)[,-c(4)]
PredictedTest<-predict(LassoMod, s = best.lambda.lasso, newx = x_test)
ModelTest2<-data.frame(obs = test_data_nonempty_1$price, pred=c(PredictedTest))
defaultSummary(ModelTest2)
```

Given the absence of strong multicollinearity in the data set, the Lasso regression results closely align with the full regression model developed earlier. The R-squared value for the training data set is 85.2%, while the R-squared value for the test data set is 86.2%, indicating stable model performance across both sets.

Evaluate the performances of models built all above on both train and test data sets.

```{r QY1-Whole Evaluation-Rsquare}
pr<-data.frame(cbind(Model=c("Regression Model_R-Square","Regression Tree_R-Square", "Lasso_R-Square"),TrainData=c(.574,.804,.852),
TestData=c(.921,.707,.862)))
pr
```

From these results, it is evident that Lasso regression yields significantly higher and more stable R-squared values compared to other benchmark models.

```{r QY1-Whole Evaluation-RMSE}
pr<-data.frame(cbind(Model=c("Regression Model_RMSE","Regression Tree_RMSE","NN_RMSE", "Lasso_RMSE"),TrainData=c(70406,38034,128690,33236),
TestData=c(21553,41689,121820,28593)))
pr
```

Based on the RMSE results, Lasso regression demonstrates the lowest RMSE among all four regression methods.

```{r QY1-Whole Evaluation-MAE}
pr<-data.frame(cbind(Model=c("Regression Model_MAE","Regression Tree_MAE","NN_MAE", "Lasso_MAE"),TrainData=c(17988,27212,93736,20117),
TestData=c(14769,29766,90124,20322)))
pr
```

From the MAE analysis, we find that while another regression model has the lowest MAE, the Lasso regression's MAE is also reasonably low. Thus, when comparing R-squared, RMSE, and MAE, Lasso regression appears to be the best choice for this analysis.

## VII. Ongoing Model Monitoring Plan 

Monitoring and maintaining regression models entails a multifaceted endeavor, demanding a systematic approach to track performance, detect anomalies, and discern moments when the model's stability and accuracy are compromised. This paper delineates essential components of model monitoring, delineates quantitative thresholds and triggers signaling the need for model replacement, and scrutinizes underlying assumptions pivotal for the model's continuous operation.

First, a comprehensive model monitoring framework is imperative to ensure efficacy. This framework necessitates consideration of diverse metrics and analytical tools, encompassing: performance metrics: tracking indicators such as accuracy, precision, recall, and the F1-Score provides insight into the model's discriminative power and regression accuracy. Additional metrics like Area Under the Curve (AUC-ROC) or Mean Squared Error (MSE) can further augment understanding. Detection of data drift and concept drift: Statistical methodologies like the Kolmogorov-Smirnov or Chi-Square tests are employed to identify shifts in data distribution. Monitoring performance enables timely detection of concept drift through variations in prediction accuracy. Model explainability: leveraging interpretability techniques such as SHAP or LIME facilitates understanding of influential features, thereby illuminating potential sources of instability or bias.

Secondly, the establishment of quantitative thresholds and triggers is paramount for effective data monitoring. To uphold optimal model performance: performance thresholds delineate acceptable ranges for key metrics, with deviations prompting comprehensive reviews or preemptive measures. Drift detection thresholds, defined statistically, identify significant deviations in data distribution or increases in false positives, necessitating deeper investigation or model retraining. Ensuring model stability entails implementing continuous monitoring systems capable of detecting real-time behavioral variations. Back-testing with historical data further validates stability and informs proactive adjustments.

Lastly, for continuous model use, adherence to specific assumptions is indispensable to ensure longevity and reliability: consistent data quality: maintaining consistent data quality, structure, and characteristics over time is imperative. Regular model maintenance: scheduled retraining and fine-tuning address evolving data patterns and business requirements. Compliance and regulatory adherence: alignment with relevant legal, ethical, and industry-specific regulations is mandatory.

Generally speaking, effective monitoring and maintenance of regression models necessitate a comprehensive approach encompassing performance tracking, anomaly detection, and corrective action implementation. Adherence to these best practices guarantees the model's reliability and sustained success in real-world applications.

## VIII. Conclusion 

The goal of this study is to investigate the factors affecting the sale prices of residential properties in Ames, Iowa, during the period from 2006 to 2010. We use a variety of statistical techniques and machine learning models to understand the relationship between different variables and the sale prices of these properties.

To ensure data quality, we removed columns with a high proportion of missing values—namely, Alley, Pool.QC, Fence, Misc.Feature, and Fireplace.Qu as their significant amount of NA values hindered effective analysis. The dataset was then split into a training set (70%) and a testing set (30%) for model development and validation.

To identify significant predictors, we initially applied a stepwise both-ways selection method. The residuals from the initial regression model exhibited a heavy-tailed distribution, which led us to use a Box-Cox transformation to address this issue. We then conducted outlier detection using Cook's distance and Breusch-Pagan tests to check for influential points and heteroscedasticity.

As an initial regression approach, we implemented a decision tree model to estimate house sale prices. We evaluated the model's performance using Root Mean Square Error (RMSE) and R-squared values on both the training and testing datasets. We also performed cross-validation to assess model stability.

To understand the limitations of the decision tree model, we compared its performance against neural networks (NN), Lasso regression, and linear regression models. This analysis considered multiple performance metrics, including RMSE, Mean Absolute Error (MAE), and R-squared values. The comparison revealed that Lasso regression provided the most reliable results among the four models.

The final Lasso regression model indicated that several factors had significant effects on house sale prices. Among these, Street, Overall.Qual, and Garage.Cars were identified as the most influential variables. Other factors had some impact, but their influence was relatively weak compared to the top three.

Based on our analysis, we conclude that Lasso regression is the most suitable model for predicting house sale prices in this dataset. Further research could explore additional variables and interactions to improve model accuracy. The key determinants of house sale prices identified in this study are Street, Overall.Qual, and Garage.Cars that should be prioritized in future analyses and property evaluations.


## Bibliography 

[1] Ames, Iowa Assessor’s Office. Ames Iowa: Alternative to the Boston Housing Data Set. http://jse.amstat.org/v19n3/decock/DataDocumentation.txt, amstat.org, 2010.

[2] kaggle.com. House Prices - Advanced Regression Techniques.
https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/rules, kaggle.com, 2016.

[3] Ranstam, J. and Cook, J.A., 2018. LASSO regression. Journal of British Surgery, 105(10), pp.1348-1348.

## Appendix 

We plot correlation matrix in the Appendix as below.

```{r}
library(GGally)
```

```{r}
Correlation_matrix<-round(cor(train_data_nonempty_1), 2)
Correlation_matrix
```

```{r}
Correlation_matrix<-round(cor(test_data_nonempty_1), 2)
Correlation_matrix
```

```{r bib, include=FALSE}
# KEEP THIS AT THE END OF THE DOCUMENT TO GENERATE A LOCAL bib FILE FOR PKGS USED
knitr::write_bib(sub("^package:", "", grep("package", search(), value=TRUE)), file='skeleton.bib')
```