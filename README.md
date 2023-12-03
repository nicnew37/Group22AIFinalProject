# AIFinalProject

## Emotional Sentiment Chatbot Model(Functionalities and Model Training)
Our Final AI Project is an Emotional Sentiment Chatbot that helps users who are feeling down. The project includes intent classification using a neural network
and a streamlit application for the chatbot.A dataset containing labeled examples of patterns and their corresponding intents was used to train the model. The chatbot that 
responds based on user input was created using the sentiment analysis model.

The intent classification model is trained on a dataset with patterns and corresponding intents. Hyperparameter tuning is performed using GridSearchCv to find the best model.
Based on user input, Using a pre-trained model and a bag of words approach, our model then generates responses. 
NLP tokenization was used to breakdown text into individual words to thus tokenize the patterns from the dataset, thereby converting the raw words to 
an easily processible format. Here, nltk.word_tokenize() was applied to each pattern to break it down into a list of words which are then used to build the vocabulary to train
the neural network. 
In our code, 'SnowballStemmer' from the NLTK library was used for stemming so that words of different forms are treated the same to effectively capture the underlying meaning.


## Hosting on your local server
## Requirements 
A python setup will be needed 

##  Post python initialization and Setup 
The following have to be pip-installed
nltk pandas keras plotly scikit-learn matplotlib streamlit



## Installation into  python script

`
pip install nltk pandas keras plotly scikit-learn matplotlib streamlit
`

In your preferred environment, Open and run your  python script.

## Project Structure 
'intents.json' was the dataset containing the patterns and intents
'model.h5' is our  trained neural network
'data1.pickle' is the pickle file containing the essential data(words, labels, training data, and output).
'finproj.py' is the Streamlit application

## Streamlit Chatbot
Run the Streamlit application using 'finproj.py"
Interact with the chatbot by typing in the input box and when done with your session,  type quit.

## Contact us
Contact us  for any questions or feedback at nicnew202@gmail.com and @brian.antwi@ashesi.edu.gh

Thank you!



