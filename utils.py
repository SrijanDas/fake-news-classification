import re
import string
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import one_hot
import numpy as np

nltk.download('stopwords')


def preprocess(text):
    voc_size = 10000
    max_length = 1000

    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = re.sub('[^a-zA-Z]', ' ', text)

    stop_words = set(stopwords.words("english"))
    ps = PorterStemmer()
    text = [ps.stem(word) for word in text.split() if not word in stop_words]

    # One hot encoding
    onehot_repr = []
    for words in text:
        for i in one_hot(words, voc_size):
            onehot_repr.append(i)

    embedded_docs = pad_sequences([onehot_repr], padding='pre', maxlen=max_length)
    return np.array(embedded_docs)
