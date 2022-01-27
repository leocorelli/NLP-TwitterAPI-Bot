from fastapi import FastAPI
import uvicorn
import requests
import pandas as pd
import boto3
from decouple import config

bearer_token = config('BEARER_TOKEN') # coming from .env file (it's an environment variable)
access_key = config('AWS_ACCESS_KEY_ID')
secret_key = config('AWS_SECRET_ACCESS_KEY')

def initialize_parameters(term, max_results = 10, bearer_token = bearer_token):
    search = f'{term} -is:retweet'
    query = f'https://api.twitter.com/2/tweets/search/recent?query={search}&max_results={max_results}'
    my_headers = {'Authorization' : 'Bearer ' + bearer_token}
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

def detect_sentiment(tweets):
    comprehend = boto3.client(service_name='comprehend', region_name='us-east-1', aws_access_key_id = access_key, aws_secret_access_key = secret_key)
    vals = {'POSITIVE': 0, 'NEUTRAL': 0, 'NEGATIVE': 0, 'MIXED': 0}
    
    print('Calling DetectSentiment')
    print(f'Analyzing {len(tweets)} total tweets\n')

    for i in range(len(tweets)):
        if i%50 == 0:
            print(f'{i}/{len(tweets)}')
        sentiment = comprehend.detect_sentiment(Text=tweets['tweet'].iloc[i], LanguageCode='en')
        vals[sentiment['Sentiment']] += 1
    
    print('\nEnd of DetectSentiment\n')

    return vals

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello Duke"}

@app.get("/add/{num1}/{num2}")
async def add(num1: int, num2: int):
    """Add two numbers together"""

    total = num1 + num2
    return {"total": total}

@app.get("/sub/{num1}/{num2}")
async def sub(num1: int, num2: int):
    """Subtract num2 from num1"""

    total = num1 - num2
    return {"total": total}

@app.get("/multiply/{num1}/{num2}")
async def sub(num1: int, num2: int):
    """multiply num1 and num2"""

    total = num1 * num2
    return {"total": total}


@app.get("/nlp/{term}")
async def nlp(term: str):
    """Do all NLP twitter api search ops"""

    query, my_headers = initialize_parameters(term, max_results = 100)
    unclean_data = call_api(query, my_headers, times = 2)
    tweets = convert_to_list(unclean_data)
    tweets = convert_to_df(tweets)
    sentiment = detect_sentiment(tweets)
    return {"sentiment": sentiment}

if __name__ == '__main__':
    uvicorn.run(app, port=8080, host='0.0.0.0')