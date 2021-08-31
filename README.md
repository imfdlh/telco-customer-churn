# Telco Customer Churn
Predictive modeling of telco customer churn tendency using Sequential NN.

# Deployment Site
https://fadilah-milestone1p2.herokuapp.com/

Template by Fadilah, powered by Bootstrap v5.1.

# Objective
The objective of this project is to build predictive modeling of customer churn using an Artificial Neural Network (ANN). If the target variable corresponds to "Churn" is yes, it means this customer has no longer subscribed to this telco company's services.

In this project, we want to minimize the number of False Negative predictions which means we failed to predict the customer that most likely will churn and lost the revenue from them. If we failed to catch a lot of the customer who intends to churn, means the company will lose a big amount of revenue they previously get from those customers which are bad for the company. Thus, the proper metric for the model evaluation is Recall since recall considers the amount of FN that the model can reduce.

In addition to the recall, we also have to evaluate the overall performance of the model. We can use accuracy or area under the curve score. But first, we have to check on the target variable distribution. If the target distribution is imbalanced, we will choose the area under curve as the evaluation metrics.

# EDA
**Distribution**

- The scale or range of numbers amongst the numerical columns is quite different, especially between tenure and MonhtlyCharges.
- Most of the customer has used the telco service for ~29 months, has MonthlyCharges ~70, and TotalCharges ~1397.
- Approximately 70% of customers don't have dependents.
- Approximately 90% of customers use phone services.
- Approximately 78% of customers used internet service.
- Approximately 59% of customers (majority) were billed paperless and 41% weren't.

**Relationship with Customer Churn**

- tenure of those customers who churned is smaller than those who didn't.
- MonthlyCharges of those customers who churned is greater than those who didn't.
- TotalCharges of those customers who churned is smaller than those who didn't.
- The majority of customers who churned is not SeniorCitizen (SeniorCitizen=0), the senior citizen only contributed 25.5% of the total churned customers.
- The number of customers who churned doesn't affect by gender, because the contribution ratio of each gender towards the customer churn is approximately equal (i.e. 27% Female churned, 26% Male churned).
- Most of the customer who churned doesn't have a partner.
- Most of the customer who churned doesn't have dependents.
- The contribution of customer churned based on contract type is mostly contributed by the customer who has a month-to-month contract.
- The contribution of customer churned based on billing type is mostly contributed by the customer who was billed paperless.
- The contribution of customer churned based on payment method is mostly contributed by the customer who paid through electronic check.
- Out of the total customer who didn't use phone service, only 25%. of them were churned.
- Between-group of customer internet service type, the customer who used fiber optic internet service contributed the most to the churned rate.

**Customer Churn**

The target variable distribution is moderately imbalanced. The customer who churned is about 1/4 of the total customer.

**Insights**

-	tenure of those customers who churned is smaller than those who didn't. It means that most of the customers who have stayed longer have a lower tendency to churn.
-	MonthlyCharges of those customers who churned are greater than those who didn't. Although that they stayed only for a short time, most of them were charged higher in monthly term than the others who didn't churn.
-	Most of the customer who churned doesn't have a partner and dependents.
-	The contribution of customer churned based on contract type is mostly contributed by the customer who has a month-to-month contract.
-	The contribution of customer churned based on billing and payment type is mostly contributed by the customer who was billed paperless and paid through electronic check.
-	Between-group of customer internet service type, the customer who used fiber optic internet service contributed the most to the churned rate.
-	The churned rate for the customer who used internet service, is mostly from the customer who didn’t use any additional add-ons (except streaming add-ons) like online security, online backup, device protection, and tech support. For the other streaming add-ons like streaming TV and movies, most of the customer who churned is the one who uses the streaming add-on.

**Recommendations**

- Offer a discount for those who are charged greater if they choose to pay for a yearly contract.
- Create a special bundling for those who don't have either a partner or a dependent, for example, like offer a special price if they want to add streaming add-ons.
- Do more extensive research on why the customer who billed paperless or paid through an electronic check churn rate is greater. Find out whether the churn is voluntary or involuntary. If it's involuntary, try to send more emails several days before the end of the subscription. If it's voluntary, try to ask for feedback.
- Create a special bundling price for those who choose to use a streaming add-on.

## Model Analysis
The main metrics that will be evaluated in this churn classification problem are AUC and Recall due to the class imbalance and the most important metrics for the business as we want to reduce the number of False Negatives.

From the initial experiments, we get a quite good AUC score but a poor Recall score. To improve the model, there are several experiments conducted, they are:
1.  Training initial Sequential API and Functional API models without sampling and hyperparameter tuning (random initializer and default adam learning rate).
2.  Training initial Sequential API and Functional API models with SMOTE oversampling and without hyperparameter tuning (random initializer and default adam learning rate).
3.  Training initial Sequential API and Functional API models with Random undersampling and without hyperparameter tuning (random initializer and default adam learning rate).
4.  Training Sequential API model (as it was decided as the best model) with Random undersampling and with hyperparameter tuning (get Random Normal and optimizer adam lr=0.0001 as the best params).

The best model was achieved in the 4th experiment.

After training the model using the best sequential API model with parameters as follows:
- 3 hidden layers, consisted of 50, 20, and 10 neurons.
- hidden layers activation function: ReLU
- hidden layer initializer: Random Normal
- optimizer and learning rate: Adam with lr=0.0001

The best model is achieved. 

As we see from the plot of the final model above, previously we set the epochs to 100 with early stopping patience=20. The iteration has stopped at epoch 68 which means even in epoch 48, the model already achieved a quite good score.

Even if the loss still looks similar to the previous experiment, the evaluation metrics consisted of AUC and Recall, have improved a lot from the initial experiments, especially for the Recall score.

The recall score achieved about 75% on the validation set and 80% on the training set, which improved a lot from the initial score (~50%). Since the recall is the most important metric for the business as it tells how good the model (or how small the FN can a model has) in detecting the customer who will churn, it's quite crucial to have a greater recall score. This score means, out of the total customer who will churn in the future, we can guess 75% of them. It's beneficial for the business because they can create a strategy to prevent their customer from churning.

The AUC scores achieve ~85% on the training set and 80% on the validation set, the scores stayed at the ~80% range which means that the model tends to be less overfitted compared to the previous experiments.
