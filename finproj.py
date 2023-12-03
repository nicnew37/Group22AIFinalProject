import nltk
from nltk.stem.snowball import SnowballStemmer
import numpy as np
import tensorflow as tf
import random
import json
import pickle
import os
import streamlit as st
import tensorflow as tf
from tensorflow import keras


st.title("Emotional Sentiment ChatBot")
st.write()
model='/Users/brian/Desktop/Python/model1.h5'

#Loading the saved model
loaded_model = tf.keras.models.load_model(model)

#Loading the preprocessed data
with open("/Users/brian/Desktop/Python/data1.pickle", "rb") as f:
    words, labels, training, output = pickle.load(f)

#Loading our intents dataset
with open('/Users/brian/Desktop/Python/intents.json') as file:
    data = json.load(file)

#Creating an instance of a Stemmer
ss = SnowballStemmer(language='english')


#Defining the bag of words function
def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [ss.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
            
    return np.array(bag)


#Defining the get repsonse function
def get_responses(tag):
    for tg in data["intents"]:
        if tg['tag'] == tag:
            return tg['responses']
    return []


#The main function block
def main():
    st.write("Start talking with the bot (Type quit to stop)!")

    user_input = st.text_input("You: ")
    if user_input.lower() == "quit":
        st.stop()

    input_data = bag_of_words(user_input, words)
    #Check if the array length is correct
    if len(input_data) != 281:
        st.warning("Error: Input data does not have the expected length.")
    else:
        input_data = input_data.reshape(1, -1)
        results = loaded_model.predict(input_data)
        results_index = np.argmax(results)
        tag = labels[results_index]

        responses = get_responses(tag)
        st.write(random.choice(responses))

if __name__ == "__main__":
    main()

