import nltk, string, numpy
from sklearn.feature_extraction.text import TfidfVectorizer

from gensim.matutils import softcossim
from gensim import corpora
import gensim.downloader as api

#nltk.download('wordnet')
lemmer = nltk.stem.WordNetLemmatizer()

def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))


TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
def cos_similarity(textlist):
    tfidf = TfidfVec.fit_transform(textlist)
    return (tfidf * tfidf.T).toarray()[0][1]

def text_similarity(documents):
    print("The cosine similarity is:")
    print(int(cos_similarity(documents)*100), '%')
    print()

    print("The soft cosine similarity is:")
    print('working on it...')
    j=0
    doc =[]
    for i in range(len(documents)):
        doc.append(documents[i].split())
        j += 1
    documents = doc
    dictionary = corpora.Dictionary(documents)

    fasttext_model300 = api.load('fasttext-wiki-news-subwords-300')
    similarity_matrix = fasttext_model300.similarity_matrix(dictionary, tfidf=None, threshold=0.0, exponent=2.0, nonzero_limit=100)

    sent_1 = dictionary.doc2bow(documents[0])
    sent_2 = dictionary.doc2bow(documents[1])
    print(int((softcossim(sent_1, sent_2, similarity_matrix))*100), "%")
