import matplotlib
matplotlib.use('Agg')

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import io
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import re
import pandas as pd
import nltk
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer
from mlflow.tracking import MlflowClient
import matplotlib.dates as mdates
import pickle
from wordcloud import WordCloud

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

def download_nltk_data():
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        print("Downloading stopwords...")
        nltk.download('stopwords')
    
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        print("Downloading wordnet...")
        nltk.download('wordnet')
    
    try:
        nltk.data.find('corpora/omw-1.4')
    except LookupError:
        print("Downloading omw-1.4...")
        nltk.download('omw-1.4')

# Download the data
download_nltk_data()

# Now import after downloading
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def preprocess_comment(comment: str) -> str:

    comment = comment.lower()
    comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)
    comment = re.sub(r'\s+', ' ', comment)
    comment = comment.strip()
    comment = re.sub(r'\n', ' ', comment)
    stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
    comment = ' '.join([word for word in comment.split() if word not in stop_words])

    # Lemmatize the words
    lemmatizer = WordNetLemmatizer()
    comment = ' '.join([lemmatizer.lemmatize(word) for word in comment.split()])

    return comment

# def load_model_and_vectorizer(model_name, model_version, vectorizer_path):
#     mlflow.set_tracking_uri("http://ec2-16-52-85-241.ca-central-1.compute.amazonaws.com:5000")
#     client = MlflowClient()
#     model_uri = f"models:/{model_name}/{model_version}"
#     model = mlflow.pyfunc.load_model(model_uri)
#     with open(vectorizer_path, 'rb') as f:
#         vectorizer = pickle.load(f)

#     return model, vectorizer


# #Initialize model and vectorizer
# model, vectorizer = load_model_and_vectorizer("sentiment_analysis_yt_comments", 1, "tfidf_vectorizer.pkl")
def load_model(model_path, vectorizer_path):
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)

        return model, vectorizer
    except Exception as e:
        raise

model, vectorizer = load_model("./lgbm_model.pkl", "./tfidf_vectorizer.pkl")

@app.route('/')
def home():
    return "Welcome to the Sentiment Analysis API"

# @app.route('/test', methods=['GET'])
# def test():
#     return jsonify({'status': 'ok'}), 200

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    comments = data.get('comments')
    print('I am the comment', comments)
    print('I amd the comment type', type(comments))

    if not comments:
        return jsonify({'error': 'Comment is required'}), 400
    
    try:
        preprocessed_comment = [preprocess_comment(comment) for comment in comments]
        transformed_comment = vectorizer.transform(preprocessed_comment)
        dense_comment = transformed_comment.toarray()

        prediction = model.predict(dense_comment).tolist()
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
    response = [{"comment": comment, "sentiment": sentiment} for comment, sentiment in zip(comments, prediction)]
    return jsonify(response)


@app.route('/predict_with_timestamps', methods=['POST', 'OPTIONS'])
def predict_with_timestamp():
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200
    data = request.json
    comments_data = data.get('comments')
   
    if not comments_data:
        return jsonify({'error': 'Comments are required'}), 400
    
    try:
        comments = [item['text'] for item in comments_data]
        timestamps = [item['timestamp'] for item in comments_data]

        preprocessed_comment = [preprocess_comment(comment) for comment in comments]
        transformed_comment = vectorizer.transform(preprocessed_comment)
        dense_comment = transformed_comment.toarray()
        predictions = model.predict(dense_comment).tolist()
        predictions = [str(pred) for pred in predictions]

    except Exception as e:
        return jsonify({'error': f'prediction failed {str(e)}'}), 500
    
    
    response = [{"comment": comment, "sentiment": sentiment, "timestamp": timestamp} for comment, sentiment, timestamp in zip(comments, predictions, timestamps)]
    return jsonify(response)

@app.route('/generate_chart', methods=['POST'])
def generate_chart():
    try:
        data = request.get_json()
        sentiment_counts = data.get('sentiment_counts')
        
        if not sentiment_counts:
            return jsonify({"error": "No sentiment counts provided"}), 400

        # Prepare data for the pie chart
        labels = ['Positive', 'Neutral', 'Negative']
        sizes = [
            int(sentiment_counts.get('1', 0)),
            int(sentiment_counts.get('0', 0)),
            int(sentiment_counts.get('-1', 0))
        ]
        if sum(sizes) == 0:
            raise ValueError("Sentiment counts sum to zero")
        
        colors = ['#36A2EB', '#C9CBCF', '#FF6384']  # Blue, Gray, Red

        # Generate the pie chart
        plt.figure(figsize=(6, 6))
        plt.pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct='%1.1f%%',
            startangle=140,
            textprops={'color': 'w'}
        )
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

        # Save the chart to a BytesIO object
        img_io = io.BytesIO()
        plt.savefig(img_io, format='PNG', transparent=True)
        img_io.seek(0)
        plt.close()

        # Return the image as a response
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        app.logger.error(f"Error in /generate_chart: {e}")
        return jsonify({"error": f"Chart generation failed: {str(e)}"}), 500

@app.route('/generate_wordcloud', methods=['POST'])
def generate_wordcloud():
    try:
        data = request.get_json()
        comments = data.get('comments')

        if not comments:
            return jsonify({"error": "No comments provided"}), 400

        # Preprocess comments
        preprocessed_comments = [preprocess_comment(comment) for comment in comments]

        # Combine all comments into a single string
        text = ' '.join(preprocessed_comments)

        # Generate the word cloud
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='black',
            colormap='Blues',
            stopwords=set(stopwords.words('english')),
            collocations=False
        ).generate(text)

        # Save the word cloud to a BytesIO object
        img_io = io.BytesIO()
        wordcloud.to_image().save(img_io, format='PNG')
        img_io.seek(0)

        # Return the image as a response
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        app.logger.error(f"Error in /generate_wordcloud: {e}")
        return jsonify({"error": f"Word cloud generation failed: {str(e)}"}), 500

@app.route('/generate_trend_graph', methods=['POST'])
def generate_trend_graph():
    try:
        data = request.get_json()
        sentiment_data = data.get('sentiment_data')

        if not sentiment_data:
            return jsonify({"error": "No sentiment data provided"}), 400

        # Convert sentiment_data to DataFrame
        df = pd.DataFrame(sentiment_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Set the timestamp as the index
        df.set_index('timestamp', inplace=True)

        # Ensure the 'sentiment' column is numeric
        df['sentiment'] = df['sentiment'].astype(int)

        # Map sentiment values to labels
        sentiment_labels = {-1: 'Negative', 0: 'Neutral', 1: 'Positive'}

        # Resample the data over monthly intervals and count sentiments
        monthly_counts = df.resample('M')['sentiment'].value_counts().unstack(fill_value=0)

        # Calculate total counts per month
        monthly_totals = monthly_counts.sum(axis=1)

        # Calculate percentages
        monthly_percentages = (monthly_counts.T / monthly_totals).T * 100

        # Ensure all sentiment columns are present
        for sentiment_value in [-1, 0, 1]:
            if sentiment_value not in monthly_percentages.columns:
                monthly_percentages[sentiment_value] = 0

        # Sort columns by sentiment value
        monthly_percentages = monthly_percentages[[-1, 0, 1]]

        # Plotting
        plt.figure(figsize=(12, 6))

        colors = {
            -1: 'red',     # Negative sentiment
            0: 'gray',     # Neutral sentiment
            1: 'green'     # Positive sentiment
        }

        for sentiment_value in [-1, 0, 1]:
            plt.plot(
                monthly_percentages.index,
                monthly_percentages[sentiment_value],
                marker='o',
                linestyle='-',
                label=sentiment_labels[sentiment_value],
                color=colors[sentiment_value]
            )

        plt.title('Monthly Sentiment Percentage Over Time')
        plt.xlabel('Month')
        plt.ylabel('Percentage of Comments (%)')
        plt.grid(True)
        plt.xticks(rotation=45)

        # Format the x-axis dates
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=12))

        plt.legend()
        plt.tight_layout()

        # Save the trend graph to a BytesIO object
        img_io = io.BytesIO()
        plt.savefig(img_io, format='PNG')
        img_io.seek(0)
        plt.close()

        # Return the image as a response
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        app.logger.error(f"Error in /generate_trend_graph: {e}")
        return jsonify({"error": f"Trend graph generation failed: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug = True)