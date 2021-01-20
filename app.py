from flask import Flask, render_template, request, url_for
from tensorflow.keras.models import load_model
from utils import preprocess

app = Flask(__name__)
model = load_model('fake_news_lstm.model')


@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        if request.method == 'POST':
            news = request.form['news']
            if news == '':
                return render_template('index.html')
            else:
                embedded_docs = preprocess(news)

                class_names = ['Real News', 'Fake News']
                alert_class_names = ['success', 'danger']            

                pred = model.predict_classes(embedded_docs)

                # print("-----------------------------------")
                # print("News: ", news)
                # print("Embedded Docs:\n", embedded_docs)
                # print("prediction: ", pred)
                # print("-----------------------------------")
                
                prediction = class_names[pred[0][0]]
                alert_class = alert_class_names[pred[0][0]]

                return render_template('index.html', news=news, prediction=prediction, alert_class=alert_class)

        else:
            return render_template('index.html')

    except Exception as e:
        print(e)


if __name__ == '__main__':
    app.run(debug=True)
