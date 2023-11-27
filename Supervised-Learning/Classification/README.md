# K-Nearest Neighbors (KNN) / K-ближайших соседей (KNN)

K-Nearest Neighbors (KNN) is a type of instance-based learning, or lazy learning, where the function is only approximated locally and all computation is deferred until function evaluation. It works by classifying a data point based on how its neighbors are classified. KNN stores all available cases and classifies new cases based on a similarity measure.

K-ближайших соседей (KNN) - это тип обучения на основе экземпляров или ленивого обучения, где функция аппроксимируется только локально, и все вычисления откладываются до оценки функции. Он работает, классифицируя точку данных на основе того, как классифицируются ее соседи. KNN хранит все доступные случаи и классифицирует новые случаи на основе меры сходства.

# Support Vector Machines (SVM) / Метод опорных векторов (SVM)

Support Vector Machines (SVM) are supervised learning models with associated learning algorithms that analyze data for classification and regression analysis. SVMs are effective in high dimensional spaces and are versatile in the sense that different Kernel functions can be specified for the decision function. They work by finding a hyperplane that best divides the dataset into classes.

Метод опорных векторов (SVM) - это модели обучения с учителем с соответствующими алгоритмами обучения, которые анализируют данные для классификации и регрессионного анализа. SVM эффективны в пространствах высокой размерности и универсальны в том смысле, что для функции принятия решений могут быть указаны различные ядерные функции. Они работают, находя гиперплоскость, которая наилучшим образом разделяет набор данных на классы.

# Naive Bayes / Наивный Байес

Naive Bayes is a simple technique for constructing classifiers: models that assign class labels to problem instances, represented as vectors of feature values, where the class labels are drawn from some finite set. It's based on applying Bayes' theorem with strong (naive) independence assumptions between the features. Each feature contributes independently to the probability of the class, which allows for efficient computation.

Наивный Байес - это простая техника для построения классификаторов: моделей, которые присваивают классовые метки экземплярам проблем, представленным в виде векторов значений признаков, где классовые метки извлекаются из некоторого конечного набора. Он основан на применении теоремы Байеса с сильными (наивными) предположениями о независимости между признаками. Каждый признак независимо вносит вклад в вероятность класса, что позволяет эффективно вычислять.

# Decision Trees / Деревья решений

A decision tree is a decision support hierarchical model that uses a tree-like model of decisions and their possible consequences, including chance event outcomes, resource costs, and utility. It's one of the easiest and popular classification algorithms to understand and interpret. It works by creating a model, which predicts the value of a target variable by learning simple decision rules inferred from the data features.

Дерево решений - это иерархическая модель поддержки принятия решений, которая использует древовидную модель решений и их возможных последствий, включая случайные исходы событий, затраты ресурсов и полезность. Это один из самых простых и популярных алгоритмов классификации для понимания и интерпретации. Он работает, создавая модель, которая прогнозирует значение целевой переменной, изучая простые правила принятия решений, выведенные из признаков данных.

# Ensemble Methods / Ансамблевые методы

Ensemble methods are techniques that aim at improving the accuracy of results in models by combining multiple models instead of using a single model. They are often used to reduce overfitting, improve robustness and improve prediction performance. The main principle behind ensemble methods is that a group of weak learners can come together to form a strong learner.

Ансамблевые методы - это техники, которые направлены на улучшение точности результатов в моделях путем объединения нескольких моделей вместо использования одной модели. Они часто используются для уменьшения переобучения, улучшения устойчивости и улучшения производительности прогнозирования. Основной принцип за методами ансамбля заключается в том, что группа слабых учеников может объединиться, чтобы сформировать сильного ученика.

# Quadratic Discriminant Analysis (QDA) / Квадратичный дискриминантный анализ (QDA)

Quadratic Discriminant Analysis (QDA) is a statistical algorithm that uses a quadratic function to make a decision about the classification. It's a variant of Linear Discriminant Analysis (LDA), where the difference is that LDA assumes that the covariance of each class is identical, while QDA assumes that each class has its own covariance matrix. This makes QDA more flexible than LDA and can lead to better classification performance when this assumption holds.

Квадратичный дискриминантный анализ (QDA) - это статистический алгоритм, который использует квадратичную функцию для принятия решения о классификации. Это вариант линейного дискриминантного анализа (LDA), где разница состоит в том, что LDA предполагает, что ковариация каждого класса идентична, в то время как QDA предполагает, что у каждого класса есть своя матрица ковариации. Это делает QDA более гибким, чем LDA, и может привести к лучшей производительности классификации, когда это предположение соблюдается.

# Gradient Boosting Machines (GBM) / Машины градиентного бустинга (GBM)

Gradient Boosting Machines (GBM) are a machine learning technique for regression and classification problems, which produces a prediction model in the form of an ensemble of weak prediction models. GBM builds the model in a stage-wise fashion, and it generalizes them by allowing optimization of an arbitrary differentiable loss function.

Машины градиентного бустинга (GBM) - это техника машинного обучения для задач регрессии и классификации, которая создает модель прогнозирования в виде ансамбля слабых моделей прогнозирования. GBM строит модель поэтапно и обобщает их, позволяя оптимизацию произвольной дифференцируемой функции потерь.

# LightGBM

LightGBM is a gradient boosting framework that uses tree-based learning algorithms. It is designed to be distributed and efficient with the following advantages: faster training speed and higher efficiency, lower memory usage, better accuracy, support of parallel and GPU learning, capable of handling large-scale data.

LightGBM - это фреймворк для градиентного бустинга, который использует алгоритмы обучения на основе деревьев. Он разработан для распределенного и эффективного использования с следующими преимуществами: более быстрая скорость обучения и более высокая эффективность, меньшее использование памяти, лучшая точность, поддержка параллельного и GPU обучения, способность обрабатывать данные большого масштаба.

# CatBoost

CatBoost is a machine learning algorithm that uses gradient boosting on decision trees. It is designed to work with categorical data. One of the key benefits of CatBoost is its advanced handling of categorical features. It can automatically deal with categorical variables and does not require extensive data preprocessing like other machine learning algorithms.

CatBoost - это алгоритм машинного обучения, который использует градиентный бустинг на деревьях решений. Он разработан для работы с категориальными данными. Одним из ключевых преимуществ CatBoost является его продвинутая обработка категориальных признаков. Он может автоматически работать с категориальными переменными и не требует обширной предварительной обработки данных, как другие алгоритмы машинного обучения.