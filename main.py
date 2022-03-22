from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from searchtweets import load_credentials, gen_request_parameters, collect_results
import uvicorn
import pandas as pd
from transformers import pipeline
import numpy as np
from tqdm import tqdm

templates = Jinja2Templates(directory='htmldirectory')

def get_tweets(term):
    credentials = load_credentials(env_overwrite=True)
    query = gen_request_parameters(f"{term} -is:retweet", results_per_call=100, granularity=None)
    tweets = collect_results(query, max_tweets = 100, result_stream_args=credentials)
    return tweets

def convert_to_list(messy_tweets):
    '''Turns messy json repsonse into a list of extracted tweets
    
    Args:
        messy_tweets (list of dicts extracted from json response): 'data' part of .json() parse
        
    Returns:
        list: the cleaned list that has the extracted tweets as elements
    '''
    clean = []
    
    for i in range(len(messy_tweets)):
        for j in range(len(messy_tweets[i]['data'])):
            clean.append(messy_tweets[i]['data'][j]['text'])
    return clean

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

    messy_tweets = get_tweets(term)
    tweets = convert_to_list(messy_tweets)
    tweets = convert_to_df(tweets)
    positive, negative, neutral, total = detect_sentiment_new(tweets)
    return templates.TemplateResponse("away.html", {"request": request, "term":term, "positive":positive, "negative":negative, "neutral":neutral, "total":total})

if __name__ == '__main__':
    uvicorn.run(app, port=8080, host='0.0.0.0')
