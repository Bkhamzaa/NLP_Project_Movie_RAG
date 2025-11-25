import streamlit as st
import pandas as pd
import ast
import chromadb
from sentence_transformers import SentenceTransformer
import requests
import os

# ==========================================
# 1. CONFIGURATION DE LA PAGE
# ==========================================
st.set_page_config(
    page_title="Ciné-Assistant RAG",
    page_icon="🎬",
    layout="wide"
)

# ==========================================
# 2. CHARGEMENT ET CACHE DES DONNÉES
# ==========================================
@st.cache_data
def load_and_process_data():
    if not os.path.exists('tmdb_5000_movies.csv') or not os.path.exists('tmdb_5000_credits.csv'):
        return None

    # Chargement
    movies = pd.read_csv('tmdb_5000_movies.csv')
    credits = pd.read_csv('tmdb_5000_credits.csv')
    
    # Fusion
    movies = movies.merge(credits, left_on='id', right_on='movie_id')
    movies = movies.rename(columns={'title_x': 'title'})

    # Nettoyage
    def convert(text):
        try: return [i['name'] for i in ast.literal_eval(text)]
        except: return []

    def convert3(text):
        try: return [i['name'] for i in ast.literal_eval(text)][:3]
        except: return []

    def fetch_director(text):
        try:
            for i in ast.literal_eval(text):
                if i['job'] == 'Director': return i['name']
        except: return None

    movies['genres'] = movies['genres'].apply(convert)
    movies['keywords'] = movies['keywords'].apply(convert)
    movies['cast'] = movies['cast'].apply(convert3)
    movies['director'] = movies['crew'].apply(fetch_director)

    # Création de la "Soup"
    def create_soup(x):
        return " ".join([
            ' '.join(x['keywords']) if isinstance(x['keywords'], list) else '',
            ' '.join(x['cast']) if isinstance(x['cast'], list) else '',
            str(x['director']) if x['director'] else '',
            ' '.join(x['genres']) if isinstance(x['genres'], list) else '',
            str(x['overview']) if not pd.isna(x['overview']) else ''
        ])

    movies['soup'] = movies.apply(create_soup, axis=1)
    return movies[['id', 'title', 'overview', 'genres', 'director', 'vote_average', 'soup']]

# ==========================================
# 3. INITIALISATION DU RAG (CORRECTION ICI)
# ==========================================
@st.cache_resource
def init_rag_pipeline(df):
    # --- CORRECTION DU BUG "META TENSOR" ---
    # On force l'utilisation du CPU pour éviter le conflit avec accelerate/cuda
    model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
    
    chroma_client = chromadb.PersistentClient(path="./movie_rag_db")
    
    try:
        collection = chroma_client.get_collection(name="movies_app")
        if collection.count() == 0: raise Exception("Vide")
    except:
        collection = chroma_client.create_collection(name="movies_app")
        # On limite à 2000 films pour aller vite
        subset = df.head(2000)
        embeddings = model.encode(subset['soup'].tolist())
        
        metadatas = []
        for _, row in subset.iterrows():
            metadatas.append({
                "title": row['title'],
                "director": str(row['director']),
                "rating": str(row['vote_average']),
                "overview": str(row['overview'])
            })
            
        collection.add(
            ids=[str(i) for i in subset['id'].tolist()],
            embeddings=embeddings,
            metadatas=metadatas
        )
        
    return collection, model

# ==========================================
# 4. INTERFACE LLM (OLLAMA)
# ==========================================
def query_ollama(context, question, model_name):
    url = "http://localhost:11434/api/generate"
    prompt = f"""[ROLE] Expert Cinéma.
    [CONTEXTE] {context}
    [QUESTION] {question}
    [INSTRUCTION] Recommande 3 film du contexte ."""
    
    try:
        response = requests.post(url, json={
            "model": model_name, "prompt": prompt, "stream": False, "temperature": 0.3
        })
        if response.status_code == 200: return response.json()['response']
        return f"Erreur: {response.text}"
    except: return "Erreur de connexion à Ollama."

# ==========================================
# 5. L'APPLICATION PRINCIPALE
# ==========================================
def main():
    # BARRE LATÉRALE
    with st.sidebar:
        st.header("⚙️ Paramètres")
        # Assurez-vous que le nom du modèle ici correspond à ce que vous avez dans "ollama list"
        model_choice = st.selectbox("Modèle", ["gemma3:270m", "gemma3:4b"])
        st.info("Assurez-vous qu'Ollama tourne !")

    # CONTENU PRINCIPAL
    st.title("🎬 Assistant Cinéma RAG")
    st.write("Posez une question, je cherche dans la base et je demande au LLM.")

    # Chargement
    with st.spinner('Chargement du système...'):
        df = load_and_process_data()
        if df is None:
            st.error("CSV manquants !")
            st.stop()
        collection, model = init_rag_pipeline(df)
        st.success("Système prêt !")

    # Chat
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_query = st.chat_input("Ex: Je veux un film de science-fiction avec des aliens")

    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            status = st.status("Recherche RAG en cours...", expanded=True)
            
            # 1. Retrieval
            vec = model.encode([user_query])
            results = collection.query(query_embeddings=vec, n_results=3)
            
            context_txt = ""
            for i in range(len(results['ids'][0])):
                meta = results['metadatas'][0][i]
                context_txt += f"- {meta['title']} ({meta['director']}): {meta['overview']}\n"
                status.write(f"Trouvé : **{meta['title']}**")
            
            status.update(label="Terminé !", state="complete", expanded=False)
            
            # 2. Generation
            response_placeholder = st.empty()
            response_placeholder.markdown("🤖 *Le LLM réfléchit...*")
            full_response = query_ollama(context_txt, user_query, model_choice)
            response_placeholder.markdown(full_response)
            
            st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    main()