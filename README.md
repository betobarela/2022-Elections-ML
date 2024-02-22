# Predicting the ratings of a presidential candidate in Scikit-Learn

This project is an exploratory endeavor whose goal is to build a reasonable model to predict the ratings of Candidate 1 in each Brazilian city in the Second Round of the 2022 Elections. 

In practice, used Python, Pandas and Scikit-learn to produce a regression model that predicts the ratings of the presidential candidates in the second round of the 2022 Brazilian Elections. Each city is an instance and the features are comprised by socioeconomical data from the last census manually scraped from the [IBGE page](https://www.ibge.gov.br/estatisticas/downloads-estatisticas.html). The winning estimator is a simple squared model on the more than 40 numerical features plus one 26-classes categorical one (the state feature). ElasticNet is the ansatz for regularization, with an l1-ratio of 0.736 and a regularization strenght of 
&alpha;<span> = 6.9 x 10<sup>-4</sup></span>. The model presents a score of around 0.85 and an average error that, cast as a distribution of the average population, can be seen below 
<p align="center">
  <img src="https://github.com/betobarela/webpage/blob/main/assets/img/error_population_plot.png?raw=true" width="64%" />
</p>

Feature importances are strongly connected to the coefficients of the polynomial model, which can be visualized below  
<p align="center">
  <img src="https://github.com/betobarela/webpage/blob/main/assets/img/feature_importances.png?raw=true" width="90%" />
</p>

From the plot above, we may choose eight among the most important features to visualize their combined influence on the predicted rating for Candidate 1, as below

<p align="center">
  <img src="https://github.com/betobarela/webpage/blob/main/assets/img/correlations.png?raw=true" width="100%" />
</p>

## In the scope of this stage of this project, we will content ourselves with the model above. Among the obvious next steps that could be easily taken in order to improve it, are:

1. _Dimensionality reduction_: Could use, e.g., Principal Component Analysis (or manifold methods) to alleviate the complexity of the dataset, transforming it to a lower dimensional version of itself;
2. _Outlier treatment_: Could use, e.g., Bayesian Gaussian Mixture Matrices to rid the model of most extreme outliers (with the care to define a procedure which will not exceedingly reduce the cardinality of the data);
3. _Turn the estimator into an ensemble_: Could train a few more good models, very different than our polynomial one, and define an ensemble with their combination. We could try, to start, a kernelized SVM regressor, a Random Forest and a Logistic Regressor. These models could be ensembled through Bagging. One more step that could be taken would be to try a boosting strategy on the ensemble, particularly, the great XGBoost.
