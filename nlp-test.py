import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer#, CountVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import string
import json

# import nltk
# from nltk.corpus import stopwords
# nltk.download('stopwords')

# url_trb = 'https://raw.githubusercontent.com/FakeNewsChallenge/fnc-1/master/train_bodies.csv'
url_trb = 'nlp_csv/nlp_csv/train_bodies.csv'
train_bodies = pd.read_csv(url_trb)

# url_trs = 'https://raw.githubusercontent.com/FakeNewsChallenge/fnc-1/master/train_stances.csv'
url_trs = 'nlp_csv/nlp_csv/train_stances.csv'
train_stances = pd.read_csv(url_trs)

# url_teb = 'https://raw.githubusercontent.com/FakeNewsChallenge/fnc-1/master/test_bodies.csv'
url_teb = 'nlp_csv/nlp_csv/test_bodies.csv'
test_bodies = pd.read_csv(url_teb)

# url_tes = 'https://raw.githubusercontent.com/FakeNewsChallenge/fnc-1/master/test_stances_unlabeled.csv'
url_tes = 'nlp_csv/nlp_csv/test_stances_unlabeled.csv'
test_stances = pd.read_csv(url_tes)

train = train_bodies.merge(train_stances, on='Body ID')
test = test_bodies.merge(test_stances, on='Body ID')

train['articleBody'] = train['articleBody'].str.lower()
train['Headline'] = train['Headline'].str.lower()

test['articleBody'] = test['articleBody'].str.lower()
test['Headline'] = test['Headline'].str.lower()


#Now read the file back into a Python list object
with open('nlp_csv/stop.txt', 'r') as f:
    stop = json.loads(f.read())
stop = set(stop)

# # stop = set(stopwords.words('english'))
#
train['articleBody'] = train['articleBody'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
train['Headline'] = train['Headline'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

test['articleBody'] = test['articleBody'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
test['Headline'] = test['Headline'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

def remove_punct(text):
    table = str.maketrans('', '', string.punctuation)

    return text.translate(table)

example = "I am #king"
print(remove_punct(example))

train['articleBody'] = train['articleBody'].apply(lambda x: remove_punct(x))
train['Headline'] = train['Headline'].apply(lambda x: remove_punct(x))

test['articleBody'] = test['articleBody'].apply(lambda x: remove_punct(x))
test['Headline'] = test['Headline'].apply(lambda x: remove_punct(x))

train['articleBody'] = train['articleBody'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
train['Headline'] = train['Headline'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

test['articleBody'] = test['articleBody'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
test['Headline'] = test['Headline'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))


from sklearn.feature_extraction.text import TfidfVectorizer

TFIDF_VOCAB_SIZE = 5000 # lim_unigram

STOP_WORDS = [
        "a", "about", "above", "across", "after", "afterwards", "again", "against", "all", "almost", "alone", "along",
        "already", "also", "although", "always", "am", "among", "amongst", "amoungst", "amount", "an", "and", "another",
        "any", "anyhow", "anyone", "anything", "anyway", "anywhere", "are", "around", "as", "at", "back", "be",
        "became", "because", "become", "becomes", "becoming", "been", "before", "beforehand", "behind", "being",
        "below", "beside", "besides", "between", "beyond", "bill", "both", "bottom", "but", "by", "call", "can", "co",
        "con", "could", "cry", "de", "describe", "detail", "do", "done", "down", "due", "during", "each", "eg", "eight",
        "either", "eleven", "else", "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone",
        "everything", "everywhere", "except", "few", "fifteen", "fifty", "fill", "find", "fire", "first", "five", "for",
        "former", "formerly", "forty", "found", "four", "from", "front", "full", "further", "get", "give", "go", "had",
        "has", "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself",
        "him", "himself", "his", "how", "however", "hundred", "i", "ie", "if", "in", "inc", "indeed", "interest",
        "into", "is", "it", "its", "itself", "keep", "last", "latter", "latterly", "least", "less", "ltd", "made",
        "many", "may", "me", "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly", "move", "much",
        "must", "my", "myself", "name", "namely", "neither", "nevertheless", "next", "nine", "nobody", "now", "nowhere",
        "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours",
        "ourselves", "out", "over", "own", "part", "per", "perhaps", "please", "put", "rather", "re", "same", "see",
        "serious", "several", "she", "should", "show", "side", "since", "sincere", "six", "sixty", "so", "some",
        "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", "such", "system", "take",
        "ten", "than", "that", "the", "their", "them", "themselves", "then", "thence", "there", "thereafter", "thereby",
        "therefore", "therein", "thereupon", "these", "they", "thick", "thin", "third", "this", "those", "though",
        "three", "through", "throughout", "thru", "thus", "to", "together", "too", "top", "toward", "towards", "twelve",
        "twenty", "two", "un", "under", "until", "up", "upon", "us", "very", "via", "was", "we", "well", "were", "what",
        "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon",
        "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "will",
        "with", "within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves"
        ]

########################### NOT RUNNING DUE TO MEMORY ISSUES
tfidf_vectorizer = TfidfVectorizer(max_features = TFIDF_VOCAB_SIZE, stop_words = STOP_WORDS).\
        fit(np.concatenate((train['articleBody'].values.astype('U'), train['Headline'].values.astype('U'),
                test['articleBody'].values.astype('U'), test['Headline'].values.astype('U')), axis=0))
##########################
dictionary = np.asarray(tfidf_vectorizer.get_feature_names())

print('tfidf_train_body')
tfidf_train_body = tfidf_vectorizer.transform(train['articleBody'].drop_duplicates().values.astype('U'))
tfidf_tb = pd.DataFrame.sparse.from_spmatrix(tfidf_train_body)
tfidf_tb.columns = dictionary
tfidf_tb = tfidf_tb.assign(articleBody=train['articleBody'].drop_duplicates().tolist())
tfidf_tb.to_csv('nlp_csv/tfidf_train_body.csv', index=False)

print('tfidf_train_headlines')
tfidf_train_head = tfidf_vectorizer.transform(train['Headline'].drop_duplicates().values.astype('U'))
tfidf_th = pd.DataFrame.sparse.from_spmatrix(tfidf_train_head)
tfidf_th.columns = dictionary
tfidf_th = tfidf_th.assign(Headline=train['Headline'].drop_duplicates().tolist())
tfidf_th.to_csv('nlp_csv/tfidf_train_head.csv', index=False)

print('tfidf_test_body')
tfidf_test_body = tfidf_vectorizer.transform(test['articleBody'].drop_duplicates().values.astype('U'))
tfidf_teb = pd.DataFrame.sparse.from_spmatrix(tfidf_test_body)
tfidf_teb.columns = dictionary
tfidf_teb = tfidf_teb.assign(articleBody=test['articleBody'].drop_duplicates().tolist())
tfidf_teb.to_csv('nlp_csv/tfidf_test_body.csv', index=False)

print('tfidf_test_head')
tfidf_test_head = tfidf_vectorizer.transform(test['Headline'].drop_duplicates().values.astype('U'))
tfidf_teh = pd.DataFrame.sparse.from_spmatrix(tfidf_test_head)
tfidf_teh.columns = dictionary
tfidf_teh = tfidf_teh.assign(Headline=test['Headline'].drop_duplicates().tolist())
tfidf_teh.to_csv('nlp_csv/tfidf_test_head.csv', index=False)

print('WORKS!')
