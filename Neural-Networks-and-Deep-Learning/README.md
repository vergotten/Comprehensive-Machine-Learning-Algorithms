# Neural Networks and Deep Learning / Нейронные сети и глубокое обучение

Neural Networks and Deep Learning are a subset of machine learning methods based on artificial neural networks with representation learning. These methods have dramatically improved the state-of-the-art in speech recognition, visual object recognition, object detection, and many other domains.

Нейронные сети и глубокое обучение - это подмножество методов машинного обучения, основанных на искусственных нейронных сетях с обучением представления. Эти методы значительно улучшили передовые методы в области распознавания речи, распознавания визуальных объектов, обнаружения объектов и многих других областях.

## Perceptron / Перцептрон

The Perceptron is a type of artificial neural network invented in 1957 at the Cornell Aeronautical Laboratory by Frank Rosenblatt. It can be seen as the simplest kind of feedforward neural network: a linear classifier.

Перцептрон - это тип искусственной нейронной сети, изобретенный в 1957 году в Корнеллской аэронавтической лаборатории Фрэнком Розенблаттом. Его можно рассматривать как самый простой вид прямой нейронной сети: линейный классификатор.

## Multi-Layer Perceptron (MLP) / Многослойный перцептрон (MLP)

A Multi-Layer Perceptron (MLP) is a class of feedforward artificial neural network. An MLP consists of at least three layers of nodes: an input layer, a hidden layer, and an output layer. Except for the input nodes, each node is a neuron that uses a nonlinear activation function.

Многослойный перцептрон (MLP) - это класс прямых искусственных нейронных сетей. MLP состоит как минимум из трех слоев узлов: входного слоя, скрытого слоя и выходного слоя. За исключением входных узлов, каждый узел является нейроном, который использует нелинейную активационную функцию.

## Convolutional Neural Networks (CNN) / Сверточные нейронные сети (CNN)

Convolutional Neural Networks (CNN) are a class of deep neural networks, most commonly applied to analyzing visual imagery. They are also known as shift invariant or space invariant artificial neural networks (SIANN), based on their shared-weights architecture and translation invariance characteristics.

Сверточные нейронные сети (CNN) - это класс глубоких нейронных сетей, наиболее часто применяемых для анализа визуальных изображений. Они также известны как сдвиговые инвариантные или пространственно инвариантные искусственные нейронные сети (SIANN) на основе их архитектуры с общими весами и характеристиками инвариантности к переводу.

## Recurrent Neural Networks (RNN) / Рекуррентные нейронные сети (RNN)

Recurrent Neural Networks (RNN) are a class of artificial neural networks where connections between nodes form a directed graph along a temporal sequence. This allows it to exhibit temporal dynamic behavior. Unlike feedforward neural networks, RNNs can use their internal state (memory) to process sequences of inputs.

Рекуррентные нейронные сети (RNN) - это класс искусственных нейронных сетей, где связи между узлами образуют направленный граф вдоль временной последовательности. Это позволяет ему проявлять временное динамическое поведение. В отличие от прямых нейронных сетей, RNN могут использовать свое внутреннее состояние (память) для обработки последовательностей входов.

## Long Short-Term Memory (LSTM) / Долгая краткосрочная память (LSTM)

Long Short-Term Memory (LSTM) is an artificial recurrent neural network (RNN) architecture used in the field of deep learning. Unlike standard feedforward neural networks, LSTM has feedback connections. It can not only process single data points (such as images), but also entire sequences of data (such as speech or video).

Долгая краткосрочная память (LSTM) - это архитектура искусственной рекуррентной нейронной сети (RNN), используемая в области глубокого обучения. В отличие от стандартных прямых нейронных сетей, LSTM имеет обратные связи. Он может обрабатывать не только отдельные точки данных (например, изображения), но и целые последовательности данных (например, речь или видео).

## Autoencoders / Автоэнкодеры

Autoencoders are a type of artificial neural network used to learn efficient data codings in an unsupervised manner. The aim of an autoencoder is to learn a representation (encoding) for a set of data, typically for the purpose of dimensionality reduction.

Автоэнкодеры - это тип искусственной нейронной сети, используемой для изучения эффективных кодировок данных без учителя. Цель автоэнкодера - изучить представление (кодирование) набора данных, обычно с целью уменьшения размерности.

## Generative Adversarial Networks (GAN) / Генеративно-состязательные сети (GAN)

Generative Adversarial Networks (GANs) are a class of artificial intelligence algorithms used in unsupervised machine learning, implemented by a system of two neural networks contesting with each other in a zero-sum game framework.

Генеративно-состязательные сети (GAN) - это класс алгоритмов искусственного интеллекта, используемых в машинном обучении без учителя, реализованных системой из двух нейронных сетей, соревнующихся друг с другом в рамках игры с нулевым итогом.

## Word2Vec / Word2Vec

Word2Vec is a group of related models that are used to produce word embeddings. These models are shallow, two-layer neural networks that are trained to reconstruct linguistic contexts of words.

Word2Vec - это группа связанных моделей, которые используются для создания вложений слов. Эти модели представляют собой поверхностные двухслойные нейронные сети, которые обучаются восстанавливать лингвистические контексты слов.

## BERT / BERT

BERT (Bidirectional Encoder Representations from Transformers) is a transformer-based machine learning technique for natural language processing (NLP) pre-training. It was created and published in 2018 by Jacob Devlin and his colleagues from Google.

BERT (двунаправленные представления кодировщика от трансформаторов) - это техника машинного обучения на основе трансформаторов для предварительного обучения обработки естественного языка (NLP). Он был создан и опубликован в 2018 году Джейкобом Девлином и его коллегами из Google.

## Transformers / Трансформеры

Transformers are a type of model architecture introduced in a paper called "Attention is All You Need". As opposed to directional models, which read the text input sequentially (left-to-right or right-to-left), the Transformer encoder reads the entire sequence of words at once. This characteristic allows the model to learn the context of a word based on all of its surroundings (left and right of the word).

Трансформеры - это тип архитектуры модели, представленной в статье под названием "Внимание - все, что вам нужно". В отличие от направленных моделей, которые читают входной текст последовательно (слева направо или справа налево), кодировщик трансформера читает всю последовательность слов сразу. Эта характеристика позволяет модели учиться контексту слова на основе всех его окружений (слева и справа от слова).