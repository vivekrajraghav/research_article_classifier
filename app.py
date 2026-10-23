import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import nltk
from nltk.corpus import wordnet, stopwords
from nltk.stem import WordNetLemmatizer
import io
import os
@st.cache_resource
def download_nltk_data():
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('averaged_perceptron_tagger_eng', quiet=True)

download_nltk_data()

lemmatizer = WordNetLemmatizer()
stop_word = set(stopwords.words('english'))
additional_noise_words = [
    'social', 'science', 'business', 'economics', 'paper', 'study', 'research', 
    'article', 'results', 'show', 'abstract', 'journal', 'author', 'conclusion', 
    'report', 'citation', 'evaluation', 'domain', 'concept', 'factor', 'public', 
    'scientific', 'scientist', 'data', 'information', 'publication', 'cite', 
    'analyze', 'come', 'people', 'model', 'requirement', 'implementation', 
    'discovery', 'argue', 'scholar', 'environment','association', 'publish', 
    'western', 'ib', 'library','sociology', 'sociologist', 'sociological', 'society'
]
stop_word.update(additional_noise_words)

def clean_text(text):
    text = str(text)
    text = re.sub(r"'s\b", "", text)
    text = text.replace("-", " ")
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.lower().split()
    tagged_word = nltk.pos_tag(words)
    cleaned_words = []
    for word, tag in tagged_word:
        pos = tag[0].upper()
        tag_dict = {
            "J": wordnet.ADJ,
            "N": wordnet.NOUN,
            "V": wordnet.VERB,
            "R": wordnet.ADV
        }
        wordnet_pos = tag_dict.get(pos, wordnet.NOUN)
        lemma_word = lemmatizer.lemmatize(word, wordnet_pos)
        if lemma_word not in stop_word:
            cleaned_words.append(lemma_word)
    return ' '.join(cleaned_words)

@st.cache_resource
def load_model():
    return joblib.load("abstract_classifier.joblib")

pipeline = load_model()

st.title("Research Article Classifier")
st.markdown("Categorize research paper abstracts into **Science**, **Business & Economics**, or **Social Science** domains.")

app_mode = st.radio(
    "Choose Prediction Mode:",
    ("Single Abstract", "Batch Prediction"),
    horizontal=True
)
st.divider()

if app_mode == "Single Abstract":
    st.header("Single Abstract Prediction")
    
    # Use a form to automatically clear the text area on submit
    with st.form(key="single_predict_form", clear_on_submit=True):
        abstract_input = st.text_area("Paste the abstract here:", height=200)
        submit_button = st.form_submit_button("Predict Domain")

    if submit_button:
        if abstract_input.strip():
            with st.spinner("Analyzing text..."):
                cleaned_input = clean_text(abstract_input)
                
                # Check if cleaning removed everything
                if not cleaned_input.strip():
                    st.error("The abstract contains only noise words or is empty after cleaning.")
                else:
                    prediction = pipeline.predict([cleaned_input])[0]
                    st.success(f"Predicted Research Area: **{prediction}**")
                    st.info(f"**Original Text Analyzed:**\n\n{abstract_input}") # Optional: shows what was submitted since the box clears
        else:
            st.warning("Please enter an abstract to classify.")

elif app_mode == "Batch Prediction":
    st.header("Batch Prediction via Excel")
    st.write("Upload an Excel file containing a column named `Abstract`.")
    uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx", "xls"])
    if uploaded_file is not None:
        try:
            base_name = os.path.splitext(uploaded_file.name)[0]
            df = pd.read_excel(uploaded_file)
            if "Abstract" not in df.columns:
                st.error("The uploaded Excel file must contain an 'Abstract' column.")
            else:
                with st.spinner(f"Processing {len(df)} rows..."):
                    # Preprocess text
                    df["cleaned_abstract"] = df["Abstract"].apply(clean_text)
                    
                    # Prevent crashes by replacing empty strings with NaN and dropping them
                    df["cleaned_abstract"] = df["cleaned_abstract"].replace(r'^\s*$', np.nan, regex=True)
                    df = df.dropna(subset=["cleaned_abstract"], ignore_index=True)
                    
                    if df.empty:
                        st.error("No valid text found in the abstracts after cleaning. All rows were dropped.")
                    else:
                        # Predict
                        df["Prediction"] = pipeline.predict(df["cleaned_abstract"])
                        
                        # Drop the temporary column to keep the output clean
                        df_output = df.drop(columns=["cleaned_abstract"])
                        
                        st.success(f"Batch prediction completed! {len(df_output)} valid abstracts classified.")
                        st.dataframe(df_output.head())
                        
                        # Convert DataFrame to an Excel file in memory
                        output = io.BytesIO()
                        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                            df_output.to_excel(writer, index=False, sheet_name='Predictions')
                        processed_data = output.getvalue()
                        
                        st.download_button(
                            label="Download Predictions as Excel",
                            data=processed_data,
                            file_name=f"{base_name}_predictions.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
        except Exception as e:
            st.error(f"An error occurred while processing the file: {e}")