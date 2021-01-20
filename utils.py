import re
import string
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import one_hot
import numpy as np
from tensorflow.keras.models import load_model

nltk.download('stopwords')


def get_prediction(embedded_docs):
    x = np.array(embedded_docs)
    model = load_model('fake_news_bid_lstm.model')
    prediction = model.predict_classes(x)
    # print("\n\n_____Prediction_____________________:", prediction)
    return prediction[0][0]


def preprocess(text):
    voc_size = 10000
    max_length = 100

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
#     print("___________Txt__________", text)
    # One hot encoding
    onehot_repr = []
    for words in text:
        for i in one_hot(words, voc_size):
            onehot_repr.append(i)

    embedded_docs = pad_sequences([onehot_repr], padding='pre', maxlen=max_length)
#     print("----------------------Embedded Docs:-------------\n", embedded_docs)
    return embedded_docs
