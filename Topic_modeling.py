import os
import tarfile
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk
import re

# Download NLTK data files (only for the first time)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

data_path = "C:\\Users\\Akash\\Downloads\\twenty+newsgroups\\20_newsgroups.tar.gz"
extracted_path = "C:\\Users\\Akash\\Downloads\\twenty+newsgroups\\20_newsgroups"

if not os.path.exists(extracted_path):
    with tarfile.open(data_path, 'r:gz') as tar:
        tar.extractall(path=os.path.dirname(data_path))

# Load the data from the specific category
category_path = os.path.join(extracted_path, 'misc.forsale')
documents = []

for file_name in os.listdir(category_path):
    file_path = os.path.join(category_path, file_name)
    with open(file_path, 'r', encoding='latin1') as file:
        documents.append(file.read())

# Preprocess the text data
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    # Remove non-alphabetic characters
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I|re.A)
    text = text.lower()
    text = text.strip()
    
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Remove stop words and lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

documents = [preprocess(doc) for doc in documents]

# Convert the text data into numerical form using TF-IDF
vectorizer = TfidfVectorizer(max_df=0.9, min_df=2, stop_words='english')
X = vectorizer.fit_transform(documents)

# Apply LDA for topic modeling
lda = LatentDirichletAllocation(n_components=10, random_state=42)
lda.fit(X)

# Display the topics found by LDA
def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print(f"Topic {topic_idx}:")
        print(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))

no_top_words = 10
display_topics(lda, vectorizer.get_feature_names_out(), no_top_words)

# Apply K-means for document clustering
num_clusters = 10
km = KMeans(n_clusters=num_clusters, random_state=42)
km.fit(X)

# Attach the cluster labels to the documents
df = pd.DataFrame({'Document': documents, 'Cluster': km.labels_})

# Display the clustering results
print(df.head())

# Save the clustering results to a CSV file
df.to_csv('document_clusters.csv', index=False)
