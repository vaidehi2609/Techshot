import json
from flask import Flask, jsonify,request
import content_based
from flask import Flask, jsonify,request
import requests
from bs4 import BeautifulSoup
import json
import pandas as pd
import html
import openai
import os
import csv
import pickle
import sklearn
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Set up OpenAI API credentials
OPENAI_API_KEY = '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'
openai.api_key = OPENAI_API_KEY
# RSS scraping and data extraction
def parse_content(cfinal):
    ret = ''
    skip1c = 0
    skip2c = 0
    for i in cfinal:
        if i == '<':
            skip1c += 1
        elif i == '>' and skip1c > 0:
            skip1c -= 1
        elif skip1c == 0 and skip2c == 0:
            ret += i
    ret = ret.replace('&#8217;', '\'')
    return ret

def parse_content2(cfinal):
    text = html.unescape(cfinal)
    return text

def news_rss(link):
    article_list = []
    csv_filename = 'dataf.csv'
    
    # Check if the CSV file exists
    if not os.path.isfile(csv_filename):
        # Create an empty DataFrame and save it as the CSV file
        empty_df = pd.DataFrame(columns=['id', 'title', 'link', 'published_date', 'description', 'content','summary','keywords'])
        empty_df.to_csv(csv_filename, index=False)

    # Read the existing CSV file to initialize the counter
    existing_data = pd.read_csv(csv_filename)
    counter = len(existing_data) if existing_data is not None else 0

    try:
        r = requests.get(link)
        soup = BeautifulSoup(r.content, features='xml')
        articles = soup.findAll('item')

        for a in articles:
            counter += 1  # Increment the counter for each article

            title = a.find('title').text
            link = a.find('link').text
            published = a.find('pubDate').text
            desc = a.find('description').text
            content = a.find('content:encoded').text

            article = {
                'id': counter,  # Add the counter as 'id'
                'title': title,
                'link': link,
                'published_date': published,
                'description': parse_content(desc),
                'content': parse_content(content)
            }

            article_list.append(article)

        return article_list
    except Exception as e:
        print('The scraping job failed. See exception: ')
        print(e)

def append_to_csv(dataframe, filename):
    file_exists = os.path.isfile(filename)

    with open(filename, 'a+', newline='', encoding='utf-8') as csvfile:
        if not file_exists:
            dataframe.to_csv(csvfile,index=False,)
        else:
            dataframe.to_csv(csvfile,index=False, header=False, mode='a')

def appendsumm_to_csv(id, summary):
    filename = 'dataf.csv'
    df = pd.read_csv(filename)

    # Update the 'summary' column of the row with matching ID
    df.loc[df['id'] == id, 'summary'] = summary

    # Save the DataFrame back to the CSV file
    df.to_csv(filename, index=False)

    print("Summary added to CSV successfully.")
            
def appendkeyw_to_csv(id,keywords):
    filename = 'dataf.csv'
    df = pd.read_csv(filename)

    # Convert the dictionary to a string representation
    keywords_str = json.dumps(keywords)

    # Update the 'summary' column of the row with matching ID
    df.loc[df['id'] == id, 'keywords'] = keywords_str

    # Save the DataFrame back to the CSV file
    df.to_csv(filename, index=False)

    print("Keywords added to CSV successfully.")
        
def save_dataframe(news):
    # print(news)
    df = pd.DataFrame.from_records(news)
    df['content'] = df['content'].apply(parse_content2)
    csv_filename = 'dataf.csv'
    # summarize_text(df)
    # extract_topn_keywords(df)
    
    append_to_csv(df, csv_filename)
    # df.to_csv('file1.csv')
    return 'DataFrame saved successfully.'

def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    sorted_items = sorted_items[:topn]
    score_vals = []
    feature_vals = []

    for idx, score in sorted_items:
        fname = feature_names[idx]
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])

    results = {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]]=score_vals[idx]

    return results

# def load_pickle_file(file_path):
#     with open(file_path, 'rb') as file:
#         data = pickle.load(file)
#     return data




def extract_topn_keywords(id,summary):
    #FILE PATHS HERE
    tfidf_transformer=pickle.load(open("data/tfidf_transformer.pkl",'rb'))
    cv = pickle.load(open('data/cv.pkl','rb'))
# Create CountVectorizer object

# Create feature names list

# Create empty list to store results
    # results = []
    feature_names=cv.get_feature_names_out()
    # Loop through each row in content column
    # for summary in df['summary']:

        # Create tf-idf vector for current row
    tf_idf_vector = tfidf_transformer.transform(cv.transform([summary]))

    # Sort the tf-idf vectors by descending order of scores
    sorted_items = sort_coo(tf_idf_vector.tocoo())

    # Extract only the top 10 keywords
    keywords = extract_topn_from_vector(feature_names, sorted_items, 10)

    # Append keywords to results list
    # results.append(keywords)

    # Create new column in DataFrame with keywords
    appendkeyw_to_csv(id,keywords)
    return keywords

def generate_summary(id,title,text): 
   
    word_limit = 800
    words = text.split()
    if len(words) > word_limit:
        newtext = ' '.join(words[:word_limit])
    else:
        newtext=text    

    prompt = f"Summarize the following text in 400 characters\n\n{newtext}\ntl;dr:"
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=100,
        n=1,
        stop='\n'
    )
    # print(response)
   
    summary =  response['choices'][0]['text']
    # print(summary)
    # return summary

    print("Text Summarized Successfully")
    appendsumm_to_csv(id,summary)
    return summary

    

#ROUTES
@app.route("/")
def hello_world():
    return "Techshot Api"

@app.route('/recommend',methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        try:
            if not request.data:
                return jsonify({"error": "No input data"}),400
            data = json.loads(request.data)
            if not data.get("news_id"):
                return jsonify({"error": "No news_id in request data"}),400
            input_data = int(data["news_id"])
            output=content_based.recommend_articles(input_data,11)
            return jsonify(output),200
        except Exception as e:
            return jsonify({"error": str(e)}),400
    else:
        return "Not a valid method"

@app.route('/scrape_news', methods=['POST'])
def scrape_news():
    news_link = 'https://techcrunch.com/feed/'
    news = news_rss(news_link)
    print(news)
    # save_dataframe(news)
    

    return jsonify(news)

# Summarization using OpenAI API
@app.route('/summarize_text', methods=['POST'])
def summarize_text():
    article_data =request.get_json()
    # print(article_data)
    if article_data is None or 'id' not in article_data or 'title' not in article_data or 'content' not in article_data:
        return jsonify({'error': 'Invalid request. Required fields missing.'}), 400

    id = article_data['id']
    title = article_data['title']
    text = article_data['content']
  
    summary = generate_summary(id,title,text)  
    keywords = extract_topn_keywords(id,summary)  
    return jsonify({'id':id,'title':title,'summary': summary,'keywords':keywords})


if __name__ == '__main__':
    app.run(debug=True,port=8001)
