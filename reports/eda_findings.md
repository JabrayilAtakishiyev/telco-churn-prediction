# EDA Findings

- Overall churn rate is 26.5% (1869 of 7043 customers).
- Contract: churn is highest for 'Month-to-month' (42.7%) and lowest for 'Two year' (2.8%).
- InternetService: churn is highest for 'Fiber optic' (41.9%) and lowest for 'No' (7.4%).
- PaymentMethod: churn is highest for 'Electronic check' (45.3%) and lowest for 'Credit card (automatic)' (15.2%).
- TechSupport: churn is highest for 'No' (31.2%) and lowest for 'Yes' (15.2%).
- OnlineSecurity: churn is highest for 'No' (31.3%) and lowest for 'Yes' (14.6%).
- Customers in the 0-12 month tenure group churn at 47.4%, compared with 6.6% in the 61+ month group.
- Average monthly charges are 74.44 for churned customers versus 61.27 for retained customers.
- Tenure and TotalCharges are strongly correlated (0.83), which indicates multicollinearity for linear models.
- Strongest categorical associations with churn (Cramer's V): Contract (0.41), tenure_group (0.35), InternetService (0.32), PaymentMethod (0.30), PaperlessBilling (0.19).
