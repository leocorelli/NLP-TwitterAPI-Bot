from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
import uvicorn
import requests
import pandas as pd
from decouple import config
from transformers import pipeline
import numpy as np
from tqdm import tqdm

bearer_token = config('BEARER_TOKEN') # env variable
templates = Jinja2Templates(directory='htmldirectory')

def initialize_parameters(term, max_results = 10, bearer_token = bearer_token):
    search = f'{term} -is:retweet'
    query = f'https://api.twitter.com/2/tweets/search/recent?query={search}&max_results={max_results}'
    my_headers = {'Authorization' : 'Bearer ' + bearer_token}
    print(f'\nSearch term: {term}')
    return query, my_headers


def call_api(query, my_headers, times = 1):
    unclean_data = []
    for i in range(times):
        if i == 0:
            response = requests.get(query, headers = my_headers)
            response = response.json()
            data = response['data']
            meta = response['meta']
            next_token = meta['next_token']
            
            unclean_data.extend(data)

        else:
            new_query = query + '&next_token=' + next_token
            
            response = requests.get(new_query, headers = my_headers)
            response = response.json()
            data = response['data']
            meta = response['meta']
            next_token = meta['next_token']
            
            unclean_data.extend(data)
    
    return unclean_data


def convert_to_list(messy_data):
    '''Turns messy json repsonse into a list of extracted tweets
    
    Args:
        messy_data (list of dicts extracted from json response): 'data' part of .json() parse
        
    Returns:
        list: the cleaned list that has the extracted tweets as elements
    '''
    clean_list = []
    
    for i in range(len(messy_data)):
        clean_list.append(messy_data[i]['text'])
    return clean_list

def convert_to_df(lst_of_tweets):
    df = pd.DataFrame(lst_of_tweets, columns = ['tweet'])
    for i in range(len(df)):
        df['tweet'].iloc[i] = df['tweet'].iloc[i].split(' https')[0]

    df = df.drop_duplicates(ignore_index = True)
    return df

def detect_sentiment_new(tweets):
    vals = {'LABEL_0': 0, 'LABEL_1': 0, 'LABEL_2': 0} # 0 --> Negative, 1 --> Neutral, 2 --> Positive

    classifier = pipeline('sentiment-analysis', model = "cardiffnlp/twitter-roberta-base-sentiment")
    
    print(f'Analyzing {len(tweets)} total tweets\n')

    for i in tqdm(range(len(tweets))):
        sentiment = classifier(tweets['tweet'].iloc[i])[0]
        vals[sentiment['label']] += 1
    
    print('\nEnd of DetectSentiment\n')

    total_classified = len(tweets)
    percent_positive = np.round((vals['LABEL_2']/total_classified) * 100,2) # positive
    percent_negative = np.round((vals['LABEL_0']/total_classified) * 100,2) # negative
    percent_neutral = np.round((vals['LABEL_1']/total_classified) * 100,2) # neutral

    return f'{percent_positive}%', f'{percent_negative}%', f'{percent_neutral}%', total_classified

app = FastAPI()

@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

@app.get("/RealNLP")
async def realNLP(request: Request, term: str):
    """New and improved NLP sentiment analysis"""

    query, my_headers = initialize_parameters(term, max_results = 100)
    unclean_data = call_api(query, my_headers, times = 1)
    tweets = convert_to_list(unclean_data)
    tweets = convert_to_df(tweets)
    positive, negative, neutral, total = detect_sentiment_new(tweets)
    return templates.TemplateResponse("away.html", {"request": request, "term":term, "positive":positive, "negative":negative, "neutral":neutral, "total":total})

if __name__ == '__main__':
    uvicorn.run(app, port=8080, host='0.0.0.0')
