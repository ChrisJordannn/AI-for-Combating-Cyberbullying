# -------------------------------------------------------------------------
#                           Import Libraries
# -------------------------------------------------------------------------
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pickle

import gradio as gr
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.layers import LSTM
from keras.saving import register_keras_serializable

# Register the LSTM layer
@register_keras_serializable()
class RegisteredLSTM(LSTM):
    def __init__(self, *args, **kwargs):
        # Remove 'time_major' if present
        kwargs.pop("time_major", None)
        super().__init__(*args, **kwargs)

from source.config import *
from source.data_cleaning import clean_text

# -------------------------------------------------------------------------
#                     Load Existing Model and Tokenizer
# -------------------------------------------------------------------------

# Load the trained model and pass the custom objects
rnn_model = load_model(MODEL_LOC, custom_objects={'LSTM': RegisteredLSTM})

# Load the tokenizer
with open(TOKENIZER_LOC, 'rb') as handle:
    tokenizer = pickle.load(handle)


# -------------------------------------------------------------------------
#                           Main Application
# -------------------------------------------------------------------------

def make_prediction(input_comment):
    """
    Predicts the toxicity of the specified comment
    :param input_comment: the comment to be verified
    """
    input_comment = clean_text(input_comment)
    input_comment = input_comment.split(" ")

    sequences = tokenizer.texts_to_sequences(input_comment)
    sequences = [[item for sublist in sequences for item in sublist]]

    padded_data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    result = rnn_model.predict(padded_data, len(padded_data), verbose=1)

    return {
        "Toxic": str(result[0][0]),
        "Very Toxic": str(result[0][1]),
        "Obscene": str(result[0][2]),
        "Threat": str(result[0][3]),
        "Insult": str(result[0][4]),
        "Hate": str(result[0][5]),
        "Neutral": str(result[0][6])
    }


# comment = gr.inputs.Textbox(lines=17, placeholder="Enter your comment here")
comment = gr.Textbox(lines=17, placeholder="Enter your comment here")


title = "Comments Toxicity Detection"
description = (
    "This application uses a Bidirectional Long Short-Term Memory (LSTM) Recurrent Neural Network (RNN) "
    "model to predict the inappropriateness of a comment"
)

gr.Interface(fn=make_prediction,
             inputs=comment,
             outputs=gr.Label(),
             title=title,
             description=description).launch()







# import sys
# import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# import pickle

# import gradio as gr
# from keras.models import load_model
# from keras.preprocessing.sequence import pad_sequences

# from source.config import *
# from source.data_cleaning import clean_text

# # -------------------------------------------------------------------------
# #                     Load Existing Model and Tokenizer
# # -------------------------------------------------------------------------

# # load the trained model
# rnn_model = load_model(MODEL_LOC)

# # load the tokenizer
# with open(TOKENIZER_LOC, 'rb') as handle:
#     tokenizer = pickle.load(handle)


# # -------------------------------------------------------------------------
# #                           Main Application
# # -------------------------------------------------------------------------

# def make_prediction(input_comment):
#     """
#     Predicts the toxicity of the specified comment
#     :param input_comment: the comment to be verified
#     """
#     input_comment = clean_text(input_comment)
#     input_comment = input_comment.split(" ")

#     sequences = tokenizer.texts_to_sequences(input_comment)
#     sequences = [[item for sublist in sequences for item in sublist]]

#     padded_data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
#     result = rnn_model.predict(padded_data, len(padded_data), verbose=1)

#     return \
#         {
#             "Toxic": str(result[0][0]),
#             "Very Toxic": str(result[0][1]),
#             "Obscene": str(result[0][2]),
#             "Threat": str(result[0][3]),
#             "Insult": str(result[0][4]),
#             "Hate": str(result[0][5]),
#             "Neutral": str(result[0][6])
#         }


# comment = gr.inputs.Textbox(lines=17, placeholder="Enter your comment here")

# title = "Comments Toxicity Detection"
# description = "This application uses a Bidirectional Long Short-Term Memory (LSTM) Recurrent Neural Network (RNN) " \
#               "model to predict the inappropriateness of a comment"

# gr.Interface(fn=make_prediction,
#              inputs=comment,
#              outputs="label",
#              title=title,
#              description=description) \
#     .launch() . WHERE TO REGISTER THE LSTM LAYER.                                                                                                     from keras.layers import LSTM
# from keras.saving import register_keras_serializable

# @register_keras_serializable()
# class RegisteredLSTM(LSTM):
#     pass