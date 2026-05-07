import os
import json
import numpy as np
import faiss
import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from prompt import SYSTEM_PROMPT, build_context

# Charger les variables d'environnement
load_dotenv()

# Configuration de la page
st.set_page_config(
    page_title="Assistant Code du Travail",
    page_icon="⚖️",
    layout="centered"
)

# Titre de l'application
st.title("Assistant Code du Travail")
st.caption("Posez vos questions sur le droit du travail français")
st.divider()

# Chargement de l'index et du modèle (une seule fois grâce au cache)
@st.cache_resource
def load_resources():
    index = faiss.read_index("index/faiss.index")
    with open("index/chunks_meta.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)
    model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return index, chunks, model, client

index, chunks, model, client = load_resources()

# Fonction de recherche
def search(question, k=4):
    question_vector = model.encode([question]).astype(np.float32)
    faiss.normalize_L2(question_vector)
    scores, indices = index.search(question_vector, k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        results.append({
            "content": chunks[idx]["content"],
            "metadata": chunks[idx]["metadata"],
            "score": float(score)
        })
    return results

# Fonction de génération
def generate_response(question, results):
    context = build_context(results)
    user_prompt = f"{context}\nQuestion : {question}"
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.1,
        max_tokens=1000
    )
    return response.choices[0].message.content

# Zone de saisie de la question
question = st.text_input("Votre question :", placeholder="Ex: Combien de jours de congés payés ai-je droit ?")

# Bouton de recherche
if st.button("Rechercher", type="primary"):
    if not question:
        st.warning("Veuillez entrer une question.")
    else:
        # Recherche
        with st.spinner("Recherche en cours..."):
            results = search(question)

        # Score de confiance
        best_score = results[0]["score"] if results else 0
        if best_score < 0.45:
            st.warning(f"Aucun article directement pertinent trouvé (score : {best_score:.2f}). La réponse peut être imprécise.")

        # Génération
        with st.spinner("Génération de la réponse..."):
            response = generate_response(question, results)

        # Affichage de la réponse
        st.subheader("Réponse")
        st.markdown(response)

        # Affichage des sources
        st.divider()
        st.subheader("Sources consultées")
        for i, result in enumerate(results, 1):
            meta = result["metadata"]
            with st.expander(f"[{i}] Article {meta['article']} — {meta['titre']} (score : {result['score']:.2f})"):
                st.write(result["content"])