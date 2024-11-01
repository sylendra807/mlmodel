from flask import Flask, jsonify, request
from pymongo import MongoClient
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# MongoDB connection
MONGO_DB_CONNECT_URL = os.getenv('MONGO_DB_CONNECT_URL')
client = MongoClient(MONGO_DB_CONNECT_URL)
db = client['Bounty_basket']
collection = db['orders']

# Step 1: Fetch data and prepare it for the recommendation model
def prepare_data():
    data = pd.DataFrame(list(collection.find({}, {'_id': 0, 'userId': 1, 'name': 1})))
    data['itemID'] = data['name']
    data['rating'] = 1  # Assume each interaction has a rating of 1
    data = data.rename(columns={'userId': 'userID'})
    
    # Create a pivot table
    pivot_table = data.pivot_table(index='userID', columns='itemID', values='rating', fill_value=0)
    return pivot_table

# Step 2: Calculate cosine similarity
def calculate_similarity(pivot_table):
    cosine_sim = cosine_similarity(pivot_table)
    sim_df = pd.DataFrame(cosine_sim, index=pivot_table.index, columns=pivot_table.index)
    return sim_df

# Route to fetch recommendations
@app.route('/recommend', methods=['GET'])
def recommend():
    user_id = request.args.get('user_id', type=str)  # Get user_id from query parameters
    pivot_table = prepare_data()
    sim_df = calculate_similarity(pivot_table)
    
    if user_id in sim_df.index:
        similar_users = sim_df[user_id].sort_values(ascending=False).index[1:11]  # Get top 10 similar users
        recommended_items = pivot_table.loc[similar_users].sum().sort_values(ascending=False).index[:10]
        return jsonify({'recommended_items': recommended_items.tolist()})
    else:
        return jsonify({'error': 'Invalid user_id'}), 400

# Home route
@app.route('/')
def home():
    return "Welcome to the Recommendation System API!"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)
