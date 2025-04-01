import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf

# Constants
MODEL_PATH = 'keras_model.h5'
LABELS_PATH = 'labels.txt'
SAMPLE_IMAGE_PATH = 'waste1.jpg'

# @st.cache_resource
def load_model(model_path):
    '''Charge le modèle avec les couches personnalisées'''
    class CustomDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
      @classmethod
      def from_config(cls, config):
          new_config = dict(config)  # Conversion en dict classique
          new_config.pop('groups', None)
          return super().from_config(new_config)

    
    return tf.keras.models.load_model(
        model_path,
        custom_objects={'DepthwiseConv2D': CustomDepthwiseConv2D},
        compile=False
    )

# @st.cache_data
def load_labels(labels_path):
    '''Charge et nettoie les labels'''
    with open(labels_path, 'r') as f:
        return [line.strip().split()[-1].lower() for line in f]

def preprocess_image(img):
    '''Prétraitement de l'image pour le modèle'''
    img = ImageOps.fit(img, (224, 224), Image.Resampling.LANCZOS)
    img_array = np.asarray(img, dtype=np.float32)
    return (img_array / 127.5) - 1  # Normalisation spécifique à Teachable Machine

def predict_image(model, img_array, class_names):
    '''Effectue la prédiction et retourne les résultats'''
    processed_img = img_array[np.newaxis, ...]
    prediction = model.predict(processed_img, verbose=0)
    predicted_class_idx = np.argmax(prediction)
    confidence = np.max(prediction)
    return class_names[predicted_class_idx], confidence

# Configuration de l'interface
st.set_page_config(page_title="Classification des Déchets", layout="wide")
st.title('♻️ Classification des Déchets Organiques/Recyclables')
st.markdown("""
**Cette application permet de classifier vos déchets en deux catégories :**
- 🍃 Déchet Organique
- ♻️ Matériau Recyclable
""")

# Chargement des éléments une seule fois
model = load_model(MODEL_PATH)
class_names = load_labels(LABELS_PATH)

# Sidebar
page = st.sidebar.radio("Navigation", ["Exemple d'Image", "Uploader une Image"])

if page == "Exemple d'Image":
    st.header("Prédiction sur l'exemple fourni")
    
    if st.button('Afficher l\'exemple'):
        try:
            img = Image.open(SAMPLE_IMAGE_PATH)
            st.image(img, caption='Image Exemple', use_column_width=True)
            
            if st.button('Lancer la prédiction'):
                with st.spinner('Analyse en cours...'):
                    img_array = preprocess_image(img)
                    class_name, confidence = predict_image(model, img_array, class_names)
                    
                    st.subheader("Résultats")
                    st.success(f"**Prédiction : {class_name.upper()}**")
                    st.metric("Confiance", f"{confidence:.2%}")
        except Exception as e:
            st.error(f"Erreur lors du chargement de l'exemple : {str(e)}")

elif page == "Uploader une Image":
    st.header("Prédiction sur votre propre image")
    uploaded_file = st.file_uploader("Choisissez une image (JPG/PNG)", type=["jpg", "png"])
    
    if uploaded_file is not None:
        try:
            img = Image.open(uploaded_file)
            st.image(img, caption='Image Uploadée', use_column_width=True)
            
            if st.button('Analyser l\'image'):
                with st.spinner('Traitement en cours...'):
                    img_array = preprocess_image(img)
                    class_name, confidence = predict_image(model, img_array, class_names)
                    
                    st.subheader("Résultats")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.success(f"**Catégorie :** {class_name.upper()}")
                    with col2:
                        st.metric("Niveau de Confiance", f"{confidence:.2%}")
                    
                    # Affichage complémentaire
                    expander = st.expander("Détails techniques")
                    with expander:
                        st.write("Distribution des probabilités :")
                        probs = model.predict(img_array[np.newaxis, ...], verbose=0)[0]
                        for name, prob in zip(class_names, probs):
                            st.write(f"- {name.upper()}: {prob:.2%}")
        except Exception as e:
            st.error(f"Erreur de traitement : {str(e)}")

# Section d'information
st.sidebar.markdown("""
---
**Instructions d'utilisation :**
1. Choisissez une page dans le menu
2. Pour l'analyse personnalisée :
   - Uploader une image claire
   - Cliquez sur 'Analyser l'image'
3. Les résultats apparaîtront automatiquement
""")