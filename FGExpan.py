import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from gensim.models import FastText, KeyedVectors
import matplotlib.pyplot as plt
import seaborn as sns 
import numpy as np

class FGExpan:
    def __init__(self, data_path):

        with open(data_path, 'r') as file:
            self.data = json.load(file)['root']
        self.abstracts = [item['abstract'] for item in self.data]
        self.processed_abstracts = [self.preprocess_text(abstract) for abstract in self.abstracts]
        self.vectorizer = TfidfVectorizer()
        self.X = self.vectorizer.fit_transform(self.processed_abstracts)
        self.model = None

    def filter_data(self, condition_func):

        self.data = [item for item in self.data if condition_func(item)]
        self.abstracts = [item['abstract'] for item in self.data]
        self.processed_abstracts = [self.preprocess_text(abstract) for abstract in self.abstracts]
        self.X = self.vectorizer.fit_transform(self.processed_abstracts)

    def save_processed_data(self, path):

        with open(path, 'w') as file:
            json.dump(self.processed_abstracts, file)

    def load_processed_data(self, path):

        with open(path, 'r') as file:
            self.processed_abstracts = json.load(file)
        self.X = self.vectorizer.fit_transform(self.processed_abstracts)

    def get_cluster_centroids(self):

        cluster_labels = self.cluster_entities()
        centroids = []
        for label in set(cluster_labels):
            members = self.X[cluster_labels == label]
            centroid = members.mean(axis=0)
            centroids.append(centroid)
        return centroids

    def get_cluster_keywords(self, n_keywords=5):

        cluster_labels = self.cluster_entities()
        keywords = {}
        for label in set(cluster_labels):
            members = self.X[cluster_labels == label]
            tfidf_sorting = np.argsort(members.mean(axis=0)).flatten()[::-1]
            top_n_words = [self.vectorizer.get_feature_names()[i] for i in tfidf_sorting[:n_keywords]]
            keywords[label] = top_n_words
        return keywords
    
    def preprocess_text(self, text):

        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word not in stopwords.words('english')]
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        return tokens

    def cluster_entities(self, n_clusters=5):

        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(self.X)
        return kmeans.labels_

    def extract_summary(self, abstract, n_sentences=2):

        sentences = [sent for sent in abstract.split('.') if sent]
        if len(sentences) <= n_sentences:
            return abstract
        
        tfidf = TfidfVectorizer().fit_transform(sentences)
        ranked_sentences = sorted(((tfidf[i, :], i) for i in range(tfidf.shape[0])), reverse=True)
        
        top_sentence_indices = [ranked_sentences[i][1] for i in range(n_sentences)]
        top_sentence_indices.sort()
        
        return '. '.join([sentences[i] for i in top_sentence_indices])
    
    def create_taxonomy(self, cluster_labels):

        taxonomy = {}
        for index, label in enumerate(cluster_labels):
            if label not in taxonomy:
                taxonomy[label] = []
            taxonomy[label].append(self.abstracts[index])
        return taxonomy

    def load_word_embeddings(self, path):
 
        self.model = KeyedVectors.load_word2vec_format(path, binary=True)

    def extended_search(self, query):

        if not self.model:
            raise ValueError("Word Embeddings model is not loaded yet.")
        similar_words = self.model.most_similar(positive=[query], topn=10)
        return similar_words

    def visualize_clusters(self, cluster_labels):

        tsne = TSNE(n_components=2, random_state=0)
        X_2d = tsne.fit_transform(self.X.toarray())
        plt.figure(figsize=(10, 6))
        sns.scatterplot(X_2d[:, 0], X_2d[:, 1], hue=cluster_labels, palette="deep")
        plt.title("Visualization of Clusters")
        plt.show()

    def run(self, n_clusters=5, embeddings_path=None):
   
        cluster_labels = self.cluster_entities(n_clusters)
        taxonomy = self.create_taxonomy(cluster_labels)
        if embeddings_path:
            self.load_word_embeddings(embeddings_path)
        return taxonomy
