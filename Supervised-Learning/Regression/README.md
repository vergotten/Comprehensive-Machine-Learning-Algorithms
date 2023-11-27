# Linear Regression / Линейная регрессия

Linear Regression is the simplest form of regression where we assume a linear relationship between the independent and dependent variables. It’s used to predict a continuous outcome variable (Y) based on one or more predictor variables (X). The goal is to find the best-fitting straight line through the data points.

Линейная регрессия - это самый простой вид регрессии, где мы предполагаем линейную связь между независимыми и зависимыми переменными. Она используется для прогнозирования непрерывной результативной переменной (Y) на основе одной или нескольких предикторных переменных (X). Цель состоит в том, чтобы найти наилучшую прямую линию, которая проходит через точки данных.

# Logistic Regression / Логистическая регрессия

Unlike linear regression, logistic regression is used when the dependent variable is binary (0 or 1). It estimates the probability of an event occurring based on given independent variables. The output is a probability that the given input point belongs to a certain class, which can be used for binary classification problems.

В отличие от линейной регрессии, логистическая регрессия используется, когда зависимая переменная является бинарной (0 или 1). Она оценивает вероятность наступления события на основе заданных независимых переменных. Выходные данные - это вероятность того, что данная входная точка принадлежит определенному классу, что может быть использовано для бинарной классификации.

# Ridge Regression / Гребневая регрессия

Ridge Regression is a technique used for analyzing multiple regression data that suffer from multicollinearity (high correlation between predictor variables). By adding a degree of bias to the regression estimates (L2 regularization), Ridge regression reduces the standard errors which can help it to outperform the Ordinary Least Squares (OLS) estimator in terms of prediction error.

Гребневая регрессия - это техника, используемая для анализа множественных регрессионных данных, которые страдают от мультиколлинеарности (высокой корреляции между предикторными переменными). Добавив степень смещения к оценкам регрессии (L2-регуляризация), гребневая регрессия уменьшает стандартные ошибки, что может помочь ей превзойти оценщик обычных наименьших квадратов (OLS) с точки зрения ошибки прогнозирования.

# Lasso Regression / Лассо регрессия

Similar to ridge regression, lasso (Least Absolute Shrinkage and Selection Operator) regression not only helps in avoiding overfitting but can also be used for feature selection. It does this by introducing a penalty term (L1 regularization) that forces some of the coefficient estimates, with a minor contribution to the model, to be exactly equal to zero. This effectively reduces the number of features upon which the given solution is dependent.

Подобно гребневой регрессии, регрессия лассо (Least Absolute Shrinkage and Selection Operator) не только помогает избежать переобучения, но также может быть использована для выбора признаков. Она делает это путем введения штрафного члена (L1-регуляризация), который заставляет некоторые оценки коэффициентов, вносящие незначительный вклад в модель, быть строго равными нулю. Это эффективно уменьшает количество признаков, от которых зависит данное решение.

# Elastic Net Regression / Регрессия Elastic Net

Elastic Net Regression is a hybrid of Ridge Regression and Lasso Regression. It incorporates penalties from both L1 and L2 regularization which allows for learning a sparse model where few of the weights are non-zero like Lasso, while still maintaining the regularization properties of Ridge. Elastic-net is useful when there are multiple features which are correlated. Lasso is likely to pick one of these at random, while elastic-net is likely to pick both. This makes the Elastic Net preferred over the Lasso Regression in some cases as it can generalize better.

Регрессия Elastic Net - это гибрид гребневой регрессии и регрессии Lasso. Она включает в себя штрафы как от L1, так и от L2 регуляризации, что позволяет обучать разреженную модель, где немногие веса не равны нулю, как в Lasso, сохраняя при этом свойства регуляризации Ridge. Elastic-net полезен, когда есть несколько коррелированных признаков. Lasso, вероятно, выберет один из них случайным образом, в то время как elastic-net, вероятно, выберет оба. Это делает Elastic Net предпочтительнее регрессии Lasso в некоторых случаях, поскольку он может лучше обобщать.