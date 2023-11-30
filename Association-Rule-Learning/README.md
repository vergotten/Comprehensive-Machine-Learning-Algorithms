## Association Rule Learning / Обучение ассоциативным правилам

Association Rule Learning is a rule-based machine learning method for discovering interesting relations between variables in large databases. It aims to identify strong rules discovered in databases using measures of interestingness.

Обучение ассоциативным правилам - это метод обучения с учителем, основанный на правилах, для выявления интересных связей между переменными в больших базах данных. Он предназначен для определения сильных правил, обнаруженных в базах данных с использованием некоторых мер интересности.

## Apriori / Априори

Apriori is an algorithm for frequent item set mining and association rule learning over relational databases. It identifies the frequent individual items in the database and extends them to larger and larger item sets as long as those item sets appear sufficiently often in the database. The Apriori algorithm uses a "bottom up" approach, where frequent subsets are extended one item at a time, and groups of candidates are tested against the data.

Априори - это алгоритм для поиска часто встречающихся наборов элементов и обучения ассоциативным правилам над реляционными базами данных. Он работает, определяя часто встречающиеся отдельные элементы в базе данных и расширяя их до больших и больших наборов элементов, пока эти наборы элементов не появляются в базе данных достаточно часто. Алгоритм Apriori использует "снизу вверх" подход, где часто встречающиеся подмножества расширяются по одному элементу за раз, и группы кандидатов проверяются против данных.

## FP-Growth / Рост FP

The FP-Growth Algorithm is an alternative way to find frequent item sets without using candidate generations, thus improving performance. For so much, it uses a divide-and-conquer strategy. The core of this method is the usage of a special data structure named frequent-pattern tree (FP-tree), which retains the item set association information. The FP-Growth reduces the search costs by recursively looking for short patterns and then concatenating them into the long frequent patterns.

Алгоритм FP-Growth - это альтернативный способ найти часто встречающиеся наборы элементов без использования генерации кандидатов, тем самым улучшая производительность. Для этого он использует стратегию "разделяй и властвуй". Основой этого метода является использование специальной структуры данных, называемой деревом часто встречающихся шаблонов (FP-tree), которое сохраняет информацию об ассоциации набора элементов. FP-Growth сокращает затраты на поиск, рекурсивно ища короткие шаблоны, а затем объединяя их в длинные часто встречающиеся шаблоны.