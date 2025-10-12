# TruthMate
A fake news and job postings detector,just by pasting the text or URLs.
This app is a trained model based Logistic Regression algorithm.
It uses the gemini API keys as a backup for more accurate results.
Two datasets were used for training the model, one consists of fake news articles and other is fake job postings.
The code was written on jupyter notebook and deployed in Gradio - web UI framework
It takes two different inputs, news text and URL similarily for job postings.
The output includes whether its real or fake and the ML confidence and Gemini's credibility score.
