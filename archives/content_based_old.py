import pandas as pd
import nltk
import re
import pickle
import mysql.connector as mysql
from nltk.tokenize import RegexpTokenizer
from datetime import datetime, timedelta

# Below libraries are for text processing using NLTK
from nltk.corpus import stopwords

# Below libraries are for similarity matrices using sklearn
from sklearn.metrics.pairwise import cosine_similarity

# Below libraries are for feature representation using sklearn
from sklearn.feature_extraction.text import TfidfVectorizer


nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

#Load data
news_articles = pd.read_csv("data/news_data.csv")

#Preprocessing of item data
news_articles = news_articles[news_articles['is_active'] == 'yes']
news_articles.rename(columns = {'main_title':'headline'}, inplace = True)

#remove duplicates and shorter headlines
duplicated_articles_series = news_articles.duplicated('headline', keep = False)
news_articles = news_articles[~duplicated_articles_series]
news_articles = news_articles[news_articles['headline'].apply(lambda x: len(x.split())>5)]
#drop na values
news_articles.dropna(inplace = True)

#Replace category_id with category
di = {
     1:"Fashion",
     2:"Entertainment",
     3:"Buisness",
     4:"Sports",
     9:"Technology",
     12:"Test",
     13:"Elections",
     14:"Test",
     15:"World",
     19:"Security",
     20:"Big Data",
     21:"Cloud",
     22:"AI",
     23:"IOT",
     24:"Blockchain",
     25:"Automation",
     26:"Digital Transformation",
     27:"AR/VR",
     28:"Others",
     29:"Buisness",
     30:"Buisness",
     31:"People",
     32:"NASSCOM Research",
     33:"Startup",
     34:"Case Study"
     }

news_articles.replace({"category_id": di},inplace= True)
news_articles.rename(columns = {'category_id':'category'}, inplace = True)
news_articles['created_at'] = pd.to_datetime(news_articles['created_at'],format='%Y-%m-%d',errors='coerce')
news_articles['date'] = news_articles['created_at'].dt.date

#Keywords extractor
cv = pickle.load(open('/Users/vaidehibhagwat/Downloads/TechshotApi/data/cv.pkl','rb'))
tfidf_transformer = pickle.load(open('/Users/vaidehibhagwat/Downloads/TechshotApi/data/tfidf_transformer.pkl','rb'))

def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    """get the feature names and tf-idf score of top n items"""

    #use only topn items from vector
    sorted_items = sorted_items[:topn]

    score_vals = []
    feature_vals = []

    for idx, score in sorted_items:
        fname = feature_names[idx]

        #keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])

    #create a tuples of feature,score
    #results = zip(feature_vals,score_vals)
    results= {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]]=score_vals[idx]

    return results

feature_names=cv.get_feature_names_out()


def extract_topn_keywords(text):
  # Create tf-idf vector for current row
  tf_idf_vector = tfidf_transformer.transform(cv.transform([text]))

  # Sort the tf-idf vectors by descending order of scores
  sorted_items = sort_coo(tf_idf_vector.tocoo())

  # Extract only the top 10 keywords
  keywords = extract_topn_from_vector(feature_names, sorted_items, 10)

  return keywords

#Cleaning and tokenisation
stop= set(stopwords.words('english'))

news_articles["headline"] = news_articles["headline"].apply(lambda words: ' '.join(word.lower() for word in words.split() if word not in stop))
w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()
def lemmatize_text(text):
    return [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)]

news_articles["headline"] = news_articles["headline"].apply(lemmatize_text)
news_articles["headline"] = news_articles["headline"].apply(lambda x : " ".join(x))


# Function for removing NonAscii characters
def _removeNonAscii(s):
    return "".join(i for i in s if  ord(i)<128)
# Function for converting into lower case
def make_lower_case(text):
    return text.lower()
# Function for removing stop words
def remove_stop_words(text):
    text = text.split()
    stops = set(stopwords.words("english"))
    text = [w for w in text if not w in stops]
    text = " ".join(text)
    return text
# Function for removing punctuation
def remove_punctuation(text):
    tokenizer = RegexpTokenizer(r'\w+')
    text = tokenizer.tokenize(text)
    text = " ".join(text)
    return text
#Function for removing the html tags
def remove_html(text):
    html_pattern = re.compile('<.*?>')
    return html_pattern.sub(r'', text)
#Function for removing rdquo, ldquo, quot
def remove_words(text):
  word_list = ["rdquo","ldquo","quot","href","p","h2","style","text","align"]
  tokenizer = RegexpTokenizer(r'\w+')
  words = tokenizer.tokenize(text)
  text = ' '.join([word for word in words if word not in word_list])
  return text


# Applying all the functions in description and storing as a cleaned_desc
news_articles['cleaned_desc'] = news_articles['short_description'].apply(_removeNonAscii)
news_articles['cleaned_desc'] = news_articles.cleaned_desc.apply(func = make_lower_case)
news_articles['cleaned_desc'] = news_articles.cleaned_desc.apply(func = remove_stop_words)
news_articles['cleaned_desc'] = news_articles.cleaned_desc.apply(func=remove_punctuation)
news_articles['cleaned_desc'] = news_articles.cleaned_desc.apply(func=remove_words)
news_articles['cleaned_desc'] = news_articles.cleaned_desc.apply(func=remove_html)
news_articles['keyword_extracted'] = news_articles['cleaned_desc'].apply(extract_topn_keywords)

news_articles['keys'] = news_articles['keyword_extracted'].apply(lambda x: ' '.join(x.keys()))
news_articles.drop(['short_description','is_active','keyword_extracted'],axis=1, inplace=True)

# create content-based recommendation
tfidf = TfidfVectorizer(stop_words='english')
news_articles['content'] = news_articles['headline'] + ' ' + news_articles['category'] + ' ' + news_articles['cleaned_desc'] + ' ' + news_articles['keys'].fillna('')
tfidf_matrix = tfidf.fit_transform(news_articles['content'])
content_similarity = cosine_similarity(tfidf_matrix, tfidf_matrix)

def recommend_articles(news_id, top_k=11):
    #Find the index of the given news_id
    article_index = news_articles[news_articles['id'] == news_id].index[0]

    # Get the similarity scores for the given article
    article_scores = content_similarity[article_index]

    # Sort the articles based on similarity scores
    top_indices = article_scores.argsort()[::-1][:top_k]

    # Get the news_ids of the recommended articles
    recommended_articles = news_articles.loc[top_indices, ['id', 'created_at']]

    # Filter the recommended articles to include only those not older than 2 weeks
    target_created_at = news_articles.loc[news_articles['id'] == news_id, 'created_at'].iloc[0]
    two_weeks_ago = datetime.now() - timedelta(weeks=2)
    latest_recommended_articles = recommended_articles[
        # (recommended_articles['created_at'] > two_weeks_ago) &
        (recommended_articles['created_at'] > target_created_at)
    ]
    
    return latest_recommended_articles['id'].tolist()

