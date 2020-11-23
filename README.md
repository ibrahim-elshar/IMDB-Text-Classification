# IMDB Text Classification

### CMU 10-701 Project

We propose a novel text representation for language classification problems. The models and techniques presented were evaluated using the IMDB Movie Review Dataset (Large Movie Review Dataset v1.0). Our aim is to design and implement a binary classifier to predict a movie review sentiment as positive or negative. The dataset is composed of 50k of movie reviews from the IMDB website, half of them are positive (having a rating ≥ 7), other half is labeled as negative (having a rating ≤ 4).

The input to our classifier will be raw english text, and our output will be the class probability/prediction. The raw text will have to be converted into a representation that can be processed computationally. Existing techniques, traditional and neural-net based, rely on word-level representations of the data. Our hypothesis is that word-level representations may not sufficiently capture nuanced contextual information from the text. We believe that sentence-level representations may capture better inter-word dependencies, and yield potentially better results.

We present multiple novel *sentence-based text representations*, building on state-of-the-art Natural Language Processing (NLP) techniques. Furthermore, we evaluate their efficacy as sentiment representations using a variety of traditional binary classifiers and analyze the relative performance on the sentiment classification task.


