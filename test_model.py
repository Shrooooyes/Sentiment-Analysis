import pickle
import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

def removeLinks(text):
    text = re.sub(r'https?://\S+|www\.\S+','',str(text)) 
    return text

def removePunctuation(text):
    for punctuation in string.punctuation:
        text = text.replace(punctuation, '')
    return text

def removeStoppingWords(text):
    nlp = stopwords.words("english")
    text = ' '.join([word for word in text.split() if word not in nlp])
    return text

def convertToStem(text):
    port_stem = PorterStemmer()
    content = re.sub('[^a-zA-Z]', ' ', text)#removing all values that is not alphabet
    content = content.split()
    
    content = ' '.join([port_stem.stem(word) for word in content])
    
    return content

def vectorizing(text):
    vectorizer = pickle.load(open('vectorizer.sav','rb'))
    vector = vectorizer.transform(text)
    return vector

def cleanInput(text):
    text = text.lower()
    text = removeLinks(text)
    text = removePunctuation(text)
    text = removeStoppingWords(text)
    text = convertToStem(text)
    
    return text


text = input("Enter your comment: ")
text = cleanInput(text)
vector = vectorizing([text])

model =  pickle.load(open('trained_model.sav', 'rb'))

prediction = model.predict(vector)

if prediction == 1:
    print("respponse: Positive")
else:
    print("response: Negative")