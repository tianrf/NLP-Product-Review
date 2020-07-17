# NLP-Product-Review
This is an end-to-end sentimental analysis for switch product reviews on JingDong. The final trained model can classify product review as good, normal, bad reviews with an accuracy of 85% on dev set.

## Data
The data folder includes the original data scraped from [JingDong]:(https://item.jd.com/100010343850.html). You can see the web scrape code from **Web Scrape JingDong.ipynb**
Data is randomly split into training and test sets with a ratio of 4:1.

## Word Embedding
The model built in this project is esentially a transfer model implementing pretrained word embedding. The chinese word embedding is downloaded from [Tencent AI Lab]:(https://ai.tencent.com/ailab/nlp/en/index.html). It is a very large text with 8 million Chinese words.

## Sentence Preprocess
The **utilities.py** includes 3 functions and one class for data processing. The class **sentence_process** is meant to split chinese sentence into words and convert the words into index according to wordembeddings.

## Model
There are three models built through tensorflow. All of 3 models obtained similar performance on dev set. Here, the architeture of model 3 are shown:
![Model3 Architect](https://github.com/tianrf/NLP-Product-Review/image/model3 architect.jpeg)
