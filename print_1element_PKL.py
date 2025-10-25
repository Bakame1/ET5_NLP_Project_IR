import pickle

# Charger les données prétraitées depuis le fichier pickle
with open('preprocessed_data.pkl', 'rb') as f:
    documents = pickle.load(f)

# Afficher le premier document
print("Premier document prétraité :")
print(documents[1])
