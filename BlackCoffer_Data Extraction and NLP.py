#!/usr/bin/env python
# coding: utf-8

# # Importing The Dataset "Input.xlsx"

# In[1]:


import pandas as pd 
import numpy as np 


# In[2]:


data=pd.read_excel(r"C:\Users\HP\Dropbox\Input.xlsx")
data


# # 2 Data Extraction

# In[3]:


from bs4 import BeautifulSoup
import requests


# In[4]:


# Function to extract and print article text from URL
def extract_and_print_article_text(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        # Extract article title and text
        article_title = soup.find('title').get_text()
        article_text = " ".join([p.get_text() for p in soup.find_all('p')])
        # Print article text
        print(f"Article Title: {article_title}\n")
        print(f"Article Text:\n{article_text}\n")
    except Exception as e:
        print(f"Error extracting article from URL {url}: {e}")

# Iterate over each row in input data and extract article text
for index, row in data.iterrows():
    url = row['URL']
    print(f"Extracting article from URL: {url}\n")
    extract_and_print_article_text(url)
    print("="*50)  # Add separator between articles


# # 3	Data Analysis

# ## Importing the required library

# In[5]:


import nltk
import string
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.probability import FreqDist
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import cmudict


# In[6]:


# Download necessary resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('cmudict')


# ## Cleaning using Stop Words Lists

# In[7]:


import os

# Function to remove stop words using multiple stop words lists
def remove_stopwords(text):
    # Load stop words from all files in the "StopWords" folder
    stop_words = set()
    stop_words_folder = r"C:\Users\HP\Dropbox\StopWords-20240316T053348Z-001\StopWords"
    for filename in os.listdir(stop_words_folder):
        with open(os.path.join(stop_words_folder, filename), 'r') as file:
            stop_words.update([line.strip() for line in file.readlines()])
    
    # Tokenize the text
    tokens = word_tokenize(text)
    # Remove punctuation
    table = str.maketrans('', '', string.punctuation)
    tokens = [word.translate(table) for word in tokens]
    # Remove stopwords
    tokens = [word for word in tokens if word.lower() not in stop_words]
    return ' '.join(tokens)


# In[8]:


def count_syllables(word):
    # Load CMU Pronouncing Dictionary
    d = cmudict.dict()
    # Check if word is in dictionary
    if word.lower() in d:
        return [len(list(y for y in x if y[-1].isdigit())) for x in d[word.lower()]][0]
    else:
        return 0


# ## 	Analysis of Readability

# In[9]:


def analyze_text(text):
    # Remove stopwords and clean text
    cleaned_text = remove_stopwords(text)
    
    # Tokenize sentences
    sentences = sent_tokenize(cleaned_text)
    num_sentences = len(sentences)
    
    # Tokenize words
    words = word_tokenize(cleaned_text)
    num_words = len(words)
    
    # Compute complexity variables
    avg_sentence_length = num_words / num_sentences
    
    complex_word_count = 0
    for word in words:
        if count_syllables(word) > 2:
            complex_word_count += 1
    
    percentage_complex_words = complex_word_count / num_words * 100
    
    fog_index = 0.4 * (avg_sentence_length + percentage_complex_words)
    
    avg_words_per_sentence = num_words / num_sentences
    
    # Compute personal pronouns
    personal_pronouns = re.findall(r'\b(I|we|my|ours|us)\b', text, flags=re.IGNORECASE)
    
    # Compute average word length
    total_word_length = sum(len(word) for word in words)
    avg_word_length = total_word_length / num_words
    
    return {
        "Average Sentence Length": avg_sentence_length,
        "Percentage of Complex Words": percentage_complex_words,
        "FOG Index": fog_index,
        "Average Number of Words Per Sentence": avg_words_per_sentence,
        "Complex Word Count": complex_word_count,
        "Word Count": num_words,
        "Personal Pronouns": len(personal_pronouns),
        "Average Word Length": avg_word_length
    }


# ##  Creating a dictionary of Positive and Negative words

# In[10]:


# Function to calculate sentiment scores
def calculate_sentiment_scores(text, positive_words, negative_words):
    positive_score = sum(1 for word in text.split() if word.lower() in positive_words)
    negative_score = sum(1 for word in text.split() if word.lower() in negative_words)
    polarity_score = (positive_score - negative_score) / ((positive_score + negative_score) + 0.000001)
    subjectivity_score = (positive_score + negative_score) / (len(text.split()) + 0.000001)
    return {
        "Positive Score": positive_score,
        "Negative Score": negative_score,
        "Polarity Score": polarity_score,
        "Subjectivity Score": subjectivity_score
    }


# ## Adding the Master Dictionary

# In[12]:


# Function to load positive and negative words
def load_sentiment_words():
    positive_words = set()
    negative_words = set()
    # Load positive words
    with open(r"C:\Users\HP\Dropbox\MasterDictionary-20240316T053031Z-001\MasterDictionary\positive-words.txt", 'r') as file:
        for word in file:
            positive_words.add(word.strip())
    # Load negative words
    with open(r"C:\Users\HP\Dropbox\MasterDictionary-20240316T053031Z-001\MasterDictionary\negative-words.txt", 'r') as file:
        for word in file:
            negative_words.add(word.strip())
    return positive_words, negative_words


# ## Analyzing the texts

# In[13]:


import requests
from bs4 import BeautifulSoup

def extract_article_text(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        # Extract article text
        article_text = ''
        for paragraph in soup.find_all('p'):
            article_text += paragraph.text.strip() + ' '
        return article_text
    except Exception as e:
        print(f"Error extracting article from URL {url}: {e}")
        return None


# ## Analyzeing The Text and Getting The Output

# In[14]:


# Main function to analyze text and print results
def main():
    # Load sentiment words
    positive_words, negative_words = load_sentiment_words()
    
    # Read URLs from input.xlsx
    df = pd.read_excel(r"C:\Users\HP\Dropbox\Input.xlsx")
    
    # Create an empty DataFrame to store results
    results_df = pd.DataFrame(columns=[
        "URL_ID",
        "Positive Score",
        "Negative Score",
        "Polarity Score",
        "Subjectivity Score",
        "Average Sentence Length",
        "Percentage of Complex Words",
        "FOG Index",
        "Average Number of Words Per Sentence",
        "Complex Word Count",
        "Word Count",
        "Personal Pronouns",
        "Average Word Length"
    ])
    
    # Iterate over each row in input data and extract article text
    for index, row in df.iterrows():
        url_id = row['URL_ID']
        url = row['URL']
        article_text = extract_article_text(url)
        
        if article_text:
            # Analyze text
            text_analysis = analyze_text(article_text)
            sentiment_scores = calculate_sentiment_scores(article_text, positive_words, negative_words)
            
            # Combine results
            result_row = {
                "URL_ID": url_id,
                **sentiment_scores,
                **text_analysis
            }
            results_df = results_df.append(result_row, ignore_index=True)
        else:
            print(f"Failed to extract article for URL_ID: {url_id}")
    
    # Save results to output file
    results_df.to_excel("output.xlsx", index=False)

if __name__ == "__main__":
    main()


# In[ ]:





# In[ ]:





# In[ ]:




