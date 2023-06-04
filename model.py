import pandas as pd
import spacy
import numpy as np
import numpy
from sklearn.cluster import KMeans
import gensim
from gensim import corpora
import nltk
import re
nltk.download('punkt')
nltk.download('omw-1.4')
from gensim import corpora
import gensim.downloader as api
w2v_model_pre = api.load('glove-twitter-200')
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.datasets import make_blobs
import seaborn as sns
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import DBSCAN
import random
from tqdm import tqdm
tqdm.pandas()
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import collections
from sklearn.metrics import silhouette_score
import itertools
from sklearn.metrics.pairwise import cosine_similarity
from gensim.corpora import Dictionary
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import contractions
from sklearn.cluster import KMeans
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
from nltk.corpus import wordnet
from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer
from scipy.special import softmax
import csv
import torch
import urllib.request

def apply_tfidf_filter(series, stopwords=[], max_features=None, min_df=1, max_df=1.0,val=1):
    # Create a TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer(
        stop_words=stopwords,
        max_features=max_features,
        min_df=min_df,
        max_df=max_df
    )
    
    # Fit the vectorizer on the series data and transform the series into TF-IDF vectors
    tfidf_vectors = tfidf_vectorizer.fit_transform(series)
    
    # Get the feature names (words) from the vectorizer's vocabulary
    feature_names = tfidf_vectorizer.get_feature_names_out()
    
    # Create a DataFrame from the TF-IDF vectors
    tfidf_df = pd.DataFrame(tfidf_vectors.toarray(), columns=feature_names)
    
    # Identify the low TF-IDF words
    #display(tfidf_df.mean().to_string())
    #print(tfidf_df.columns)
    if val==1:
      low_tfidf_words = tfidf_df.columns[tfidf_df.mean() < 0.04]  # Adjust the threshold as per your requirements
    else:
      low_tfidf_words = tfidf_df.columns[tfidf_df.mean() > 0.01]
    
    # Remove the low TF-IDF words from the series
    filtered_series = series.apply(lambda x: ' '.join([word for word in x.split() if word in low_tfidf_words]))
    
    return filtered_series


def lemmatize_word(word, pos_tag='VBG'):
    lemmatizer = WordNetLemmatizer()
    if pos_tag.startswith('V'):  # Verb
        return lemmatizer.lemmatize(word, wordnet.VERB)
    elif pos_tag.startswith('N'):  # Noun
        return lemmatizer.lemmatize(word, wordnet.NOUN)
    elif pos_tag.startswith('J'):  # Adjective
        return lemmatizer.lemmatize(word, wordnet.ADJ)
    elif pos_tag.startswith('R'):  # Adverb
        return lemmatizer.lemmatize(word, wordnet.ADV)
    else:
        return lemmatizer.lemmatize(word)

def clean_text(text):
    text=re.sub('\w*\d\w*','', text)
    text=re.sub('\n',' ',text)
    text=re.sub(r"http\S+", "", text)
    text=re.sub('[^a-z]',' ',text)
    return text

def call_all(XYZ,custom_stop_words=["student", "students",'makes','make']):
  combined_data=XYZ
  custom_stop_words=[x.lower() for x in custom_stop_words]
  print(custom_stop_words)
  combined_data=combined_data.apply(lambda x:x.lower())
  combined_data=combined_data.apply(lambda x:contractions.fix(x))
  combined_data=combined_data.apply(lambda x: clean_text(x))
  pos_tag = 'VBG'
  combined_data = combined_data.apply(lambda x: ' '.join([lemmatize_word(word) for word in x.split()]))
  combined_data=combined_data.apply(lambda x: re.sub(' +',' ',x))
  
  nlp = spacy.load('en_core_web_sm')
  nlp.max_length=5000000
  combined_training=combined_data.progress_apply(lambda x: ' '.join([token.lemma_ for token in list(nlp(x)) if (token.is_stop==False and token.text not in custom_stop_words)]))
  #combined_training=combined_data.progress_apply(lambda x: ' '.join([token.lemma_ for token in list(nlp(x)) if (token.is_stop==False)]))

  # Assuming you have a pandas Series named "comments"
  #display(combined_training.to_string())
  combined_training = apply_tfidf_filter(combined_training, stopwords=['the', 'and'], max_features=1000, min_df=2, max_df=0.8)
  #display(combined_training.to_string())
  return combined_training


def split_comments_into_sentences(df):
    new_rows = []
    for index, row in df.iterrows():
        comments = row['Data']
        sentences = nltk.sent_tokenize(comments)

        for sentence in sentences:
            new_row = row.copy()
            new_row['Data'] = sentence.strip()
            new_row['OriginalC'] = comments.strip()
            new_rows.append(new_row)

    new_df = pd.DataFrame(new_rows)
    new_df.reset_index(drop=True, inplace=True)
    return new_df

def get_embedding_w2v_pre(doc_tokens):
    embeddings = []
    if len(doc_tokens) < 1:
        return np.zeros(200)
    else:
        for tok in doc_tokens:
            if tok in w2v_model_pre.key_to_index:
                embeddings.append(w2v_model_pre.get_vector(tok))
            else:
                embeddings.append(np.random.rand(200))
        return np.mean(embeddings, axis=0)

def get_vectors(cleaned):
  vectors_pre=cleaned.apply(lambda x :get_embedding_w2v_pre(x.split()))
  return vectors_pre

def kmeans_clusters(X, k):
    """Generate clusters and print Silhouette metrics using KMeans

    Args:
        X: Matrix of features.
        k: Number of clusters.

    Returns:
        Trained clustering model and labels based on X.
    """
    km = KMeans(n_clusters=k,n_init=50).fit(X)
    return km, km.labels_, km.inertia_

def evaluate_clustering(X, k, data, dictionary,activate=True,km_up=None,labels_up=None):
    # Train clustering model and get cluster labels
    if activate:
      km, labels, _ = kmeans_clusters(X, k)
    else:
      km,labels=km_up,labels_up

    # Convert the labels to a list
    labels = list(labels)

    # Calculate pairwise cosine similarity and APCS for each cluster
    cluster_centroids = []
    cluster_similarities = []
    try :
      for i in range(k):
          # get all vectors in the cluster
          cluster_vectors = [X[j] for j in range(len(X)) if labels[j] == i]
          # calculate mean of vectors in the cluster
          centroid = sum(cluster_vectors) / len(cluster_vectors)
          cluster_centroids.append(centroid)

          # calculate pairwise cosine similarity between each document and cluster centroid
          similarities = cosine_similarity(cluster_vectors, [centroid])
          # flatten the similarity matrix and take the average to get APCS
          apcs = similarities.flatten().mean()
          cluster_similarities.append(apcs)

      # calculate overall coherence
      coherence_score = sum(cluster_similarities) / len(cluster_similarities)
    except :
      coherence_score=0.0

    # Compute silhouette score for the clustering solution
    silhouette_sco = silhouette_score(X, labels)

    return coherence_score, silhouette_sco, km, labels
    
def find_best_pram(vectors_scaled,data,up,sub=False):
  # Create a dictionary from the text data
  dictionary = Dictionary([doc.split() for doc in data])
  k_range = [i for i in range(5, up)]
  if sub:
    k_range = [i for i in range(2, up)]

  # Initialize lists to store the results
  coherence_scores = []
  silhouette_scores = []
  all_kms=[]
  all_labels=[]
  # Loop over all parameter combinations in the grid
  for k in k_range:
      # Evaluate clustering for the current parameter combination
      coherence_score, sil_score, km, labels = evaluate_clustering(vectors_scaled, k, data, dictionary)
      #print("k={}, mb={}, coherence_score={}, silhouette_score={}".format(k, mb, coherence_score, sil_score))
      # Add the results to the lists
      coherence_scores.append(coherence_score)
      silhouette_scores.append(sil_score)
      all_kms.append(km)
      all_labels.append(labels)

  score_sum = np.array(coherence_scores) + np.array(silhouette_scores)

  # Find the parameter combination that gives the highest sum of coherence and silhouette scores
  best_index = np.argmax(score_sum)
  best_k = k_range[best_index]
  best_coherence_score = coherence_scores[best_index]
  best_silhouette_score = silhouette_scores[best_index]
  best_cluster=all_kms[best_index]
  best_labels=all_labels[best_index]

  print("Best number of clusters: ", best_k)
  print("Best coherence score: ", best_coherence_score)
  print("Best silhouette score: ", best_silhouette_score)
  return best_k,best_cluster,best_labels


def find_top_words(series,flag=False):
    # Concatenate all strings in the series
    combined_text = ' '.join(series.astype(str))
    
    # Split the text into individual words
    words = combined_text.split()
    if flag:
      words=[x for x in words if len(x)>3]
    # Create a pandas Series with word frequencies
    word_counts = pd.Series(words).value_counts()
    # Get the top 3 words with highest frequency
    top_words = word_counts.head(3)
    return list(top_words.index)

def naming_Clusters(df,vectors_scaled,ncl,clustering):
  cluster_centers = clustering.cluster_centers_
  top_words = []
  nlp=spacy.load('en_core_web_sm')
  clean_com=df['Cleaned']
  for i in range(ncl):
      k=10
      center = cluster_centers[i]
      fil_df=df[df['cluster'] == i]
      if fil_df.shape[0]<k:
        k=fil_df.shape[0]
      dists = np.linalg.norm(vectors_scaled - center, axis=1)
      closest_docs = np.argsort(dists)[:k]  # top 10 closest documents to the center
      words = []
      for doc_idx in closest_docs:
          try:
            words += [token.text for token in nlp(clean_com[doc_idx]) if token.is_alpha]
          except:
            pass
      if len(words)<1:
        words=find_top_words(fil_df['Cleaned'])
        if len(words)<1:
          words=find_top_words(fil_df['Data'],True)
      top_words.append(words)

  # Assign trigram names to each cluster
  trigram_names = []
  for words in top_words:
      counter = collections.Counter(words)
      top_words = counter.most_common(3)
      trigram_name = "_".join([word for word, _ in top_words])
      trigram_names.append(trigram_name)
  return trigram_names

def test_score(vectors_scaled,df,k,cluster,labels):
  data=df['Cleaned']
  dictionary = Dictionary([doc.split() for doc in data])
  coherence_score, sil_score,_,_ = evaluate_clustering(vectors_scaled, k, data, dictionary,False,cluster,labels)
  print("k={}, coherence_score={}, silhouette_score={}".format(k, coherence_score, sil_score))

def nos_to_cluster(i,lab_names):
  return lab_names[i]

def Topic_clustering(df,Stop_data):
  datafr1=df[df['Data'].notna()]
  df=split_comments_into_sentences(datafr1)
  df['Cleaned']=call_all(df['Data'],Stop_data[0])
  vec=get_vectors(df['Cleaned'])
  #vectors_scaled = StandardScaler().fit_transform([x for x in vec])
  vectors_scaled = [x for x in vec]
  #pca = PCA(n_components=70)
  #vectors_scaled = pca.fit_transform(vectors_scaled)
  #df['Vectors']=vec
  n_clus, clustering, cluster_labels=find_best_pram(vectors_scaled,df['Cleaned'],25)
  print("Estimated number of clusters: %d" % n_clus)
  test_score(vectors_scaled,df,n_clus,clustering, cluster_labels)
  df["cluster"]=cluster_labels
  names=naming_Clusters(df,vectors_scaled,n_clus,clustering)
  df['Topic_names']=df["cluster"].apply(lambda x: names[x])
  for i, name in enumerate(names):
    print(f"Cluster {i}: {name}")
  print(df.shape)
  df.head()
  return df

def sub_naming_Clusters(df,vectors_scaled,ncl,clustering):
  cluster_centers = clustering.cluster_centers_
  top_words = []
  nlp=spacy.load('en_core_web_sm')
  #print(len(clean_com),type(clean_com))
  #print(len(vectors_scaled))
  print("-------------------")
  for i in range(ncl):
      fil_df=df[df['sub_cluster'] == i]
      wordsub=find_top_words(fil_df['sub_cleaned'])
      top_words.append(wordsub)

  # Assign trigram names to each cluster
  trigram_names = []
  for words in top_words:
      trigram_name = "_".join([word for word in words])
      trigram_names.append(trigram_name)
  return trigram_names

def sub_Topic_clustering(df):
  #pca = PCA(n_components=70)
  #vectors_scaled = pca.fit_transform(vectors_scaled)
  print(f"Group size: {len(df)}")
  try:
    df['sub_cleaned'] = apply_tfidf_filter(df['Cleaned'], stopwords=['the', 'and'], max_features=1000, min_df=1, max_df=1.0,val=0)
  except:
    df['sub_cleaned'] = df['Cleaned']
  if len(df) < 4 or df['sub_cleaned'].nunique()==1:
    # Assign "None" values to subcluster and subtopic names
    df["sub_cluster"] = -1
    df["sub_Topic_names"] = "None"
    return df
  vec=get_vectors(df['sub_cleaned'])
  vectors_scaled = [x for x in vec]
  max_up=len(vectors_scaled)-1
  if max_up>15:
    max_up=15
  print(max_up)
  n_clus, clustering, cluster_labels=find_best_pram(vectors_scaled,df['sub_cleaned'],max_up,sub=True)
  print("Estimated number of clusters: %d" % n_clus)
  test_score(vectors_scaled,df,n_clus,clustering, cluster_labels)
  df["sub_cluster"] = cluster_labels
  print(len(cluster_labels))
  print(len(df))
  #topic_counts=get_docs_vs_topics(df,"sub_cluster")
  names=sub_naming_Clusters(df,vectors_scaled,n_clus,clustering)
  df['sub_Topic_names']=df["sub_cluster"].apply(lambda x: names[x])
  for i, name in enumerate(names):
    print(f"Cluster {i}: {name}")
  print(df.shape)
  #df.head()
  return df

def sub_call(df):
  subsets = [subset for _, subset in df.groupby('cluster')]
  results=[]
  for subset in subsets:
    subset_result = sub_Topic_clustering(subset)
    results.append(subset_result)

  # Concatenate the results into a single dataframe
  output_df = pd.concat(results)
  #output_df=output_df[['Data','Cleaned','sub_cleaned','cluster','Topic_names','sub_cluster','sub_Topic_names']]

  return output_df

def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

task='sentiment'
MODEL = f"cardiffnlp/twitter-roberta-base-{task}"

tokenizer = AutoTokenizer.from_pretrained(MODEL)

# download label mapping
labels=[]
mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/{task}/mapping.txt"
with urllib.request.urlopen(mapping_link) as f:
    html = f.read().decode('utf-8').split("\n")
    csvreader = csv.reader(html, delimiter='\t')
labels = [row[1] for row in csvreader if len(row) > 1]

# PT
model_sen = AutoModelForSequenceClassification.from_pretrained(MODEL)
model_sen.save_pretrained(MODEL)
tokenizer.save_pretrained(MODEL)


def find_sentiment(text):
  text = preprocess(text)
  encoded_input = tokenizer(text, return_tensors='pt')
  output = model_sen(**encoded_input)
  scores = output[0][0].detach().numpy()
  scores = softmax(scores)

  ranking = np.argsort(scores)
  jk=np.argmax(scores)
  #print(jk,labels[jk])
  return labels[jk]

def call_everything(datafr,Stop_data):
  datafr=Topic_clustering(datafr,Stop_data)
  datafr=sub_call(datafr)
  datafr.drop(['Cleaned', 'sub_cleaned','cluster','sub_cluster'], axis=1,inplace=True)
  datafr.rename({'Data': 'Comments', 'Topic_names': 'Topics','sub_Topic_names':'SubTopics'}, axis=1, inplace=True)
  datafr['Sentiment']=datafr['Comments'].apply(lambda x: find_sentiment(x))
  return datafr


