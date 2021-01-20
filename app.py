from flask import Flask, render_template, request, url_for
from utils import get_prediction, preprocess

app = Flask(__name__)


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
                prediction = get_prediction(embedded_docs)

                class_names = ['Real News', 'Fake News']
                alert_class_names = ['success', 'danger']

                prediction_class = class_names[prediction]
                alert_class = alert_class_names[prediction]

                return render_template('index.html', news=news, prediction=prediction_class, alert_class=alert_class)

        else:
            return render_template('index.html')

    except Exception as e:
        print(e)


if __name__ == '__main__':
    app.run(debug=True)
