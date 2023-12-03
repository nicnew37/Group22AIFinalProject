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



Installation into  python script
`
pip install nltk pandas keras plotly scikit-learn matplotlib streamlit

`

In your preferred environment, Open and run your  python script.




