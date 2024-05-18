import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
nltk.download('omw-1.4')

def clean_text(text):
    compiled_for_brackets = re.compile('[/(){}[]|@,;]')

    bad_chars = re.compile('[^a-z]')

    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    text = str(text).lower()
    text = re.sub(r"\s+[a-zA-Z]\s+"," ",text)
    text= re.sub(bad_chars," ",text)
    text=compiled_for_brackets.sub(' ',text)
    cleaned_text_field = ' '.join(lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words)
    return cleaned_text_field