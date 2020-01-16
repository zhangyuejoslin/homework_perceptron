from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer


#vectorization text
class news_datareader():
    def __init__(self):
        print("reading data")

    def TF_IDF_datareader(self):
        newsgroups_train = fetch_20newsgroups(subset='train')
        vectorizer = TfidfVectorizer(max_features=1000)
        vect_train = vectorizer.fit_transform(newsgroups_train.data)
        vect_train_dense = vect_train.todense()
        newsgroups_test = fetch_20newsgroups(subset='test')
        vect_test = vectorizer.transform(newsgroups_test.data)
        vect_test_dense = vect_test.todense()
        Y_train = newsgroups_train.target
        Y_test = newsgroups_test.target
        return [vect_train_dense, Y_train, vect_test_dense, Y_test]