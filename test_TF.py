from sklearn.feature_extraction.text import TfidfVectorizer

# Charger les données prétraitées
with open('preprocessed_data.pkl', 'rb') as f:
    documents = pickle.load(f)

# Préparer les données pour TF-IDF
corpus = [" ".join(doc['tokens']) for doc in documents]

# Initialiser le vectoriseur TF-IDF
vectorizer = TfidfVectorizer()

# Calculer les vecteurs TF-IDF
X = vectorizer.fit_transform(corpus)

# X est une matrice où chaque ligne représente un document et chaque colonne représente un terme (mot)
print(f"Shape de la matrice TF-IDF: {X.shape}")
