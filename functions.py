import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
# nltk.download('punkt_tab')
# nltk.download('stopwords')

def get_ngrams(column, n):
    text = ' '.join(word for word in column.str.lower() if word not in stopwords.words('portuguese'))
    tokens = word_tokenize(text, language='portuguese')
    return list(ngrams(tokens, n))