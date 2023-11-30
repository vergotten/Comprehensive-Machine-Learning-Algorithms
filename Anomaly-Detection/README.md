## Anomaly Detection / Обнаружение аномалий

Anomaly detection is the process of identifying rare items, events, or observations in data that significantly deviate from the majority of the data. It's widely used in various domains such as cyber security, medicine, machine vision, and statistics.

Обнаружение аномалий - это процесс идентификации редких элементов, событий или наблюдений в данных, которые значительно отличаются от большинства данных. Он широко используется в различных областях, таких как кибербезопасность, медицина, машинное зрение и статистика.

### Isolation Forest / Изоляционный лес

Isolation Forest is an anomaly detection algorithm that uses binary trees to isolate anomalous points based on their path length. It has linear time complexity, low memory requirement and works well with high-dimensional data. The algorithm relies upon the characteristics of anomalies, i.e., being few and different, in order to detect anomalies.

Изоляционный лес - это алгоритм обнаружения аномалий, который использует бинарные деревья для изоляции аномальных точек на основе длины их пути. Он имеет линейную временную сложность, небольшие требования к памяти и хорошо работает с данными высокой размерности. Алгоритм опирается на характеристики аномалий, то есть на то, что их мало и они отличаются, чтобы обнаружить аномалии.