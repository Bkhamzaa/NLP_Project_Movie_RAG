# NLP_Project_Movie_RAG
Projet de NLP (Master DSSD) visant à créer un assistant de recommandation de films utilisant une architecture **RAG (Retrieval-Augmented Generation)**.

## 📌 Fonctionnalités
- **Moteur de recherche sémantique :** Trouve des films basés sur le sens de la phrase et non juste des mots-clés.
- **Architecture RAG :** Utilise `ChromaDB` pour stocker les vecteurs et `Ollama` pour la génération de texte.
- **Interface Web :** Interface utilisateur interactive réalisée avec `Streamlit`.
- **Mode Local :** Fonctionne entièrement sur CPU sans dépendance Cloud (confidentialité respectée).

## 🛠️ Stack Technique
- **Langage :** Python 3.9+
- **UI :** Streamlit
- **Vector Store :** ChromaDB
- **Embedding :** sentence-transformers (`all-MiniLM-L6-v2`)
- **LLM :** Ollama (Gemma, Mistral, ou Llama3)

## 🚀 Installation

### 1. Prérequis
- Python installé.
- [Ollama](https://ollama.com/) installé et lancé.

### 2. Cloner le projet


### 3. Installer les dépendances
- pip install -r requirements.txt

### 4. Télécharger le modèle Ollama
Assurez-vous qu'Ollama tourne, puis téléchargez un modèle léger pour le test :
-ollama pull gemma3:4b  /ou
-ollama pull gemma3:270m

## ▶️ Exécution

***Lancez le serveur Ollama dans un terminal :
- ollama serve

***Lancez l'application Streamlit dans un autre terminal :
- streamlit run app.py
