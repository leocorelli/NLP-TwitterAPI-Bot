from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
import uvicorn
from searchtweets import gen_request_parameters, load_credentials, collect_results 
import pandas as pd
from transformers import pipeline
import numpy as np
import time
from decouple import config
import os

os.environ["SEARCHTWEETS_BEARER_TOKEN"] = config('SEARCHTWEETS_BEARER_TOKEN') 
os.environ["SEARCHTWEETS_CONSUMER_KEY"] = config('SEARCHTWEETS_CONSUMER_KEY')
os.environ["SEARCHTWEETS_CONSUMER_SECRET"] = config('SEARCHTWEETS_CONSUMER_SECRET')
os.environ["SEARCHTWEETS_ENDPOINT"] = config('SEARCHTWEETS_ENDPOINT')

templates = Jinja2Templates(directory='htmldirectory')

def get_tweets(term, max_tweets = 200):
    v2_search_args = load_credentials()
    query = gen_request_parameters(f"{term} -is:retweet -is:reply", results_per_call=100, granularity = None) 
    tweets = collect_results(query,result_stream_args=v2_search_args, max_tweets = max_tweets)
    lst = []
    for i in range(len(tweets)):
        for j in range(len(tweets[i]['data'])):
            lst.append(tweets[i]['data'][j]['text'])
    df = pd.DataFrame(lst,columns = ['tweet'])
    df = df.drop_duplicates(ignore_index = True)
    return df

def detect_sentiment_new(tweets):
    start = time.time()
    vals = {'POSITIVE': 0, 'NEGATIVE': 0}

    classifier = pipeline('sentiment-analysis', model = "distilbert-base-uncased-finetuned-sst-2-english")

    print('Calling sentiment-analysis')
    print(f'Analyzing {len(tweets)} total tweets\n')

    for i in range(len(tweets)):
        if i%50 == 0:
            print(f'{i}/{len(tweets)}')
        sentiment = classifier(tweets['tweet'].iloc[i])[0]
        if sentiment['score'] <= 0.6:
            continue
        else:
            vals[sentiment['label']] += 1
    
    print('\nEnd of DetectSentiment\n')

    total_not_classified = len(tweets) - (vals['POSITIVE'] + vals['NEGATIVE'])
    total_classified = len(tweets) - total_not_classified
    
    percent_positive = np.round((vals['POSITIVE']/total_classified) * 100,2)
    percent_negative = np.round((vals['NEGATIVE']/total_classified) * 100,2)
    
    print('**************')
    print(f'{percent_positive}% positive')
    print(f'{percent_negative}% negative\n')
    print(f'Total classified: {total_classified}')
    print('**************\n')
    end = time.time()
    print(f'Total time elapsed {np.round(end - start,2)} seconds.\n')
    return f'{percent_positive}%', f'{percent_negative}%', total_classified

app = FastAPI()

@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

@app.get("/RealNLP")
async def realNLP(request: Request, term: str):
    """New and improved NLP sentiment analysis"""
    try:
        tweets = get_tweets(term)
        positive, negative, total = detect_sentiment_new(tweets)
        return templates.TemplateResponse("away.html", {"request": request, "term":term, "positive":positive, "negative":negative, "total":total})
    except:
        return templates.TemplateResponse("third.html", {"request": request, "term":term})

if __name__ == '__main__':
    uvicorn.run(app, port=8080, host='0.0.0.0')