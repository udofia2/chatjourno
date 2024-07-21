import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
from transformers import BartForConditionalGeneration, BartTokenizer
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the dataset
dataset = pd.read_csv('story_data.csv')

# Load Sentence-BERT model for semantic search
sbert_model = SentenceTransformer('all-mpnet-base-v2')
sbert_model.to(device)

# Load BART summarization model and tokenizer
summarization_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
summarization_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
summarization_model.to(device)

# Load precomputed story embeddings
story_embeddings = torch.load('story_embeddings.pt')

# Function to summarize text using BART
def summarize_text(text):
    inputs = summarization_tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
    inputs = inputs.to(device)
    summary_ids = summarization_model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = summarization_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Function to process query and retrieve similar stories with summaries
def retrieve_and_summarize_stories(query, top_n=5):
    query_embedding = sbert_model.encode(query, convert_to_tensor=True, device=device)
    
    similarities = util.pytorch_cos_sim(query_embedding, story_embeddings)[0]
    top_indices = torch.topk(similarities, k=top_n).indices
    
    results = []
    for index in top_indices:
        index = index.item()  # Convert tensor to integer
        story = dataset.iloc[index]
        summary = summarize_text(story['CONTENTS'])
        result = {
            "title": story['NEWS TITLE'],
            "source": story['SOURCE'],
            "url": story['NEWS LINK'],
            "summary": summary
        }
        results.append(result)
    
    return results

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    user_query = request.form['query']
    retrieved_stories = retrieve_and_summarize_stories(user_query)
    
    return jsonify(retrieved_stories)

if __name__ == '__main__':
    app.run(debug=True)

