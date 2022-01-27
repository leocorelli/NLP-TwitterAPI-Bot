<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/4/4f/Twitter-logo.svg" width="175" />
  <img src="https://upload.wikimedia.org/wikipedia/commons/a/a1/NeuralNetwork.png" width="220" /> 
  <img src="https://upload.wikimedia.org/wikipedia/commons/f/fc/Happy_face.svg" width="160" /> 
</p>

# NLP TwitterAPI Bot - Real Time Sentiment Analysis

[![Python application test with Github Actions](https://github.com/leocorelli/NLP_TwitterAPI_Bot/actions/workflows/main.yml/badge.svg)](https://github.com/leocorelli/NLP_TwitterAPI_Bot/actions/workflows/main.yml)

Website to check out my microservice: https://tinyurl.com/NLPSentimentAnalysis

## Project Description
This microservice performs natural langauge processing (NLP) to detect sentiment of the most recent 1,000 tweets related to a particular topic. This can be used to extrapolate how the general population is feeling at a particular time about a given topic (for example: a politician, athlete, or stock).

## Automated Workflow
This project was an excellent opportunity to practice my professional software engineering skills. Upon receipt of a push, my workflow does the following:
1. **Continuous integration**: Linting, testing, and formatting
2. **Continuous delivery**: Automatic containerization and push to AWS Elastic Container Registry (ECR)
3. **Continuous deployment**: Automatic deployment pipeline utilizing AWS App Runner
