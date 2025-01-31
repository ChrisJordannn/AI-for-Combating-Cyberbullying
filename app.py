from flask import Flask, render_template, request, jsonify
import pickle
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.layers import LSTM
from keras.saving import register_keras_serializable
from source.config import *
from source.data_cleaning import clean_text

app = Flask(__name__)

# Register the LSTM layer
@register_keras_serializable()
class RegisteredLSTM(LSTM):
    def __init__(self, *args, **kwargs):
        kwargs.pop("time_major", None)
        super().__init__(*args, **kwargs)

# Load model and tokenizer
rnn_model = load_model(MODEL_LOC, custom_objects={'LSTM': RegisteredLSTM})
with open(TOKENIZER_LOC, 'rb') as handle:
    tokenizer = pickle.load(handle)

def make_prediction(input_comment):
    input_comment = clean_text(input_comment)
    input_comment = input_comment.split(" ")
    sequences = tokenizer.texts_to_sequences(input_comment)
    sequences = [[item for sublist in sequences for item in sublist]]
    padded_data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    result = rnn_model.predict(padded_data, len(padded_data), verbose=0)
    return {
        "Toxic": str(result[0][0]),
        "Very Toxic": str(result[0][1]),
        "Obscene": str(result[0][2]),
        "Threat": str(result[0][3]),
        "Insult": str(result[0][4]),
        "Identity Hate": str(result[0][5]),
        "Neutral": str(result[0][6])
    }

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/usersupport')
def user_support():
    return render_template('usersupport.html')

@app.route('/helpline')
def helpline():
    return render_template('helpline.html')

@app.route('/aboutus')
def about_us():
    return render_template('aboutus.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form['comment']
    prediction = make_prediction(data)
    return jsonify(prediction)

if __name__ == '__main__':
    app.run(debug=True)


# from flask import Flask, render_template, request, jsonify
# from keras.models import load_model
# from keras.preprocessing.sequence import pad_sequences
# from keras.layers import LSTM
# from keras.saving import register_keras_serializable
# import pickle
# import os
# from source.config import *
# from source.data_cleaning import clean_text

# # Initialize Flask app
# app = Flask(__name__, static_folder='static', template_folder='templates')

# # Register the LSTM layer
# @register_keras_serializable()
# class RegisteredLSTM(LSTM):
#     def __init__(self, *args, **kwargs):
#         kwargs.pop("time_major", None)
#         super().__init__(*args, **kwargs)

# # Load the model and tokenizer
# rnn_model = load_model(MODEL_LOC, custom_objects={'LSTM': RegisteredLSTM})
# with open(TOKENIZER_LOC, 'rb') as handle:
#     tokenizer = pickle.load(handle)

# def make_prediction(input_comment):
#     """
#     Predicts the toxicity of the specified comment.
#     """
#     input_comment = clean_text(input_comment)
#     input_comment = input_comment.split(" ")
#     sequences = tokenizer.texts_to_sequences(input_comment)
#     sequences = [[item for sublist in sequences for item in sublist]]
#     padded_data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
#     result = rnn_model.predict(padded_data, len(padded_data), verbose=1)

#     return {
#         "Toxic": result[0][0],
#         "Very Toxic": result[0][1],
#         "Obscene": result[0][2],
#         "Threat": result[0][3],
#         "Insult": result[0][4],
#         "Hate": result[0][5],
#         "Neutral": result[0][6]
#     }

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     comment = request.form['comment']
#     input_seq = tokenizer.texts_to_sequences([comment])
#     input_padded = pad_sequences(input_seq, maxlen=200)
    
#     prediction = model.predict(input_padded)[0]  # Get predictions
#     prediction = {label: float(pred) for label, pred in zip(label_names, prediction)}  # Convert to Python float
    
#     return jsonify(prediction)

# if __name__ == '__main__':
#     app.run(debug=True)
