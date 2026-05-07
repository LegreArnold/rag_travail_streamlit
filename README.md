# Assistant Code du Travail — RAG

Application de questions-réponses juridiques basée sur une architecture RAG (Retrieval-Augmented Generation) construite de zéro sans framework d'abstraction (ni LangChain ni LlamaIndex).

**Problème résolu** : Les LLM hallucinent sur des sujets juridiques spécialisés. Ce système contraint le modèle à répondre uniquement sur la base d'articles officiels du Code du travail français, en citant systématiquement ses sources.

**Démo live** : https://rag-code-travail.streamlit.app

---

## Stack technique

| Composant | Technologie | Rôle |
|---|---|---|
| Langage | Python 3.10+ | Développement complet |
| LLM | Groq — llama-3.3-70b | Génération des réponses |
| Embeddings | sentence-transformers | Transformation texte → vecteurs |
| Modèle embedding | paraphrase-multilingual-mpnet-base-v2 | Encodage multilingue (français) |
| Base vectorielle | FAISS (Meta) | Recherche par similarité cosinus |
| Interface | Streamlit | Application web interactive |
| Déploiement | Streamlit Cloud | Hébergement gratuit |
| Versioning | Git + GitHub | Gestion du code |
| Config | python-dotenv | Gestion des secrets |

---

## Architecture

Phase 1 — Indexation (une fois)
Légifrance → corpus JSON → chunk_text() → embed_chunks() → FAISS → index/
Phase 2 — Interrogation (temps réel)
Question → embedding → FAISS search → top 4 chunks → Groq LLM → réponse


---

## Fonctionnalités

- Recherche sémantique sur 22 articles officiels issus de Légifrance
- 5 thèmes couverts : durée du travail, congés payés, CDI/CDD, licenciement, rupture conventionnelle
- Score de confiance : avertissement si aucun article pertinent trouvé
- Sources affichées avec score de similarité
- Interface web déployée et accessible publiquement

---

## Installation

### 1. Cloner le repo
git clone https://github.com/LegreArnold/rag_travail_streamlit.git
cd rag_travail_streamlit

### 2. Créer et activer l'environnement virtuel
python -m venv venv
venv\Scripts\activate

### 3. Installer les dépendances
pip install -r requirements.txt

### 4. Configurer la clé API Groq
Créer un fichier .env

### 5. Lancer l'app
streamlit run app.py

---

## Choix techniques

**IndexFlatIP avec normalisation L2** → similarité cosinus pour la recherche sémantique.

**Temperature à 0.1** → réponses factuelles, zéro hallucination.

**Chunking avec overlap de 50 caractères** → pas de perte d'information aux frontières.

**Modèle multilingue** → corpus 100% français correctement encodé.

**Prompt système strict** → citation obligatoire des articles, disclaimer juridique systématique.