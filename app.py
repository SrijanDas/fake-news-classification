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
            class_names = ['Real News', 'Fake News']
            alert_class_names = ['success', 'danger']

            if news == '':
                return render_template('index.html')
            elif len(str(news).split()) <= 4:
                prediction_class = class_names[1]
                alert_class = alert_class_names[1]
                return render_template('index.html', news=news, prediction=prediction_class, alert_class=alert_class)

            else:
                embedded_docs = preprocess(news)
                prediction = get_prediction(embedded_docs)

                prediction_class = class_names[prediction]
                alert_class = alert_class_names[prediction]

                return render_template('index.html', news=news, prediction=prediction_class, alert_class=alert_class)

        else:
            return render_template('index.html')

    except Exception as e:
        print(e)


if __name__ == '__main__':
    app.run(debug=True)
