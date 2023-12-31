# Supervised Learning / Обучение с учителем

Supervised Learning is a type of machine learning where models are trained using labeled data. The model makes predictions based on this data and the accuracy of the predictions is improved over time. It's called "supervised" because the process of an algorithm learning from the training dataset can be thought of as a teacher supervising the learning process. We know the correct answers, the algorithm iteratively makes predictions on the training data and is corrected by the teacher.

Обучение с учителем - это тип машинного обучения, при котором модели обучаются с использованием размеченных данных. Модель делает прогнозы на основе этих данных, и точность прогнозов улучшается со временем. Это называется "обучение с учителем", потому что процесс обучения алгоритма на обучающем наборе данных можно рассматривать как процесс обучения под руководством учителя. Мы знаем правильные ответы, алгоритм итеративно делает прогнозы на обучающих данных и исправляется учителем.

## Regression / Регрессия

Regression is a statistical method used to understand the relationship between dependent and independent variables. It’s commonly used to make projections, such as for sales revenue for a given business. Regression analysis is widely used for prediction and forecasting, where its use has substantial overlap with the field of machine learning.

Регрессия - это статистический метод, используемый для понимания взаимосвязи между зависимыми и независимыми переменными. Он обычно используется для создания прогнозов, например, для выручки от продаж для данного бизнеса. Регрессионный анализ широко используется для прогнозирования, где его использование имеет существенное перекрытие с областью машинного обучения.

## Classification / Классификация

Classification, on the other hand, is a process in machine learning where we categorize data into a given number of classes. The main goal of a classification problem is to identify the category/class to which a new data will fall under. Classification can be performed on both structured or unstructured data. Classification is a two-step process, learning step and prediction step. In the learning step, the model is developed based on given training data. In the prediction step, the model is used to predict the response for given data.

Классификация, с другой стороны, - это процесс в машинном обучении, где мы классифицируем данные в заданное количество классов. Основная цель задачи классификации - определить категорию/класс, к которому будет относиться новые данные. Классификация может быть выполнена как на структурированных, так и на неструктурированных данных. Классификация - это двухшаговый процесс, шаг обучения и шаг прогнозирования. На этапе обучения модель разрабатывается на основе заданных обучающих данных. На этапе прогнозирования модель используется для прогнозирования ответа для заданных данных.

## Ensemble Methods / Методы ансамбля

Ensemble methods are machine learning techniques that combine several base models in order to produce one optimal predictive model. They work by generating multiple classifiers/models which learn and make predictions independently. Those predictions are then combined into a single (mega) prediction that should be as good or better than the prediction made by any one classifier.

Методы ансамбля - это техники машинного обучения, которые объединяют несколько базовых моделей для создания одной оптимальной прогностической модели. Они работают путем создания нескольких классификаторов/моделей, которые обучаются и делают прогнозы независимо. Затем эти прогнозы объединяются в один (мега) прогноз, который должен быть таким же хорошим или лучше прогноза, сделанного любым одним классификатором.

Ensemble methods can be divided into two groups:
1. **Sequential ensemble methods** where the base learners are generated sequentially (e.g. AdaBoost). This technique tries to exploit the dependence between the base learners since the overall performance can be boosted by weighing previously mislabeled examples with higher weight.
2. **Parallel ensemble methods** where the base learners are generated in parallel (e.g. Random Forest). This technique tries to exploit independence between the base learners since the error can be reduced dramatically by averaging.

Методы ансамбля можно разделить на две группы:
1. **Последовательные методы ансамбля**, где базовые модели генерируются последовательно (например, AdaBoost). Эта техника пытается использовать зависимость между базовыми моделями, поскольку общая производительность может быть увеличена за счет взвешивания ранее неправильно маркированных примеров с большим весом.
2. **Параллельные методы ансамбля**, где базовые модели генерируются параллельно (например, Random Forest). Эта техника пытается использовать независимость между базовыми моделями, поскольку ошибка может быть существенно уменьшена путем усреднения.

Most ensemble methods use a single base learning algorithm to produce homogeneous base learners, i.e. learners of the same type leading to *homogeneous ensembles*. There are also some methods that use heterogeneous learners, i.e., learners of different types, leading to *heterogeneous ensembles*.

Большинство методов ансамбля используют один базовый алгоритм обучения для создания однородных базовых моделей, т.е. моделей одного типа, что приводит к *однородным ансамблям*. Существуют также некоторые методы, которые используют гетерогенные модели, т.е. модели разных типов, что приводит к *гетерогенным ансамблям*.
