import pandas as pd
from datasets import load_dataset
from transformers import AutoModel, AlbertTokenizer
import torch
from googletrans import Translator
from scipy.spatial.distance import cosine

def get_sentence_embedding(text, tokenizer, model):
    """
    Generates sentence embedding using a transformer model.
    """
    encoded_input = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        word_embeddings = model(**encoded_input).last_hidden_state
    sentence_embedding = word_embeddings.mean(dim=1)
    return sentence_embedding

def translate_to_hindi(text):
    """
    Translates English text to Hindi.
    """
    translator = Translator()
    try:
        translated_text = translator.translate(text, src='en', dest='hi')
        return translated_text.text
    except Exception as e:
        print(f"Error during translation: {e}")
        return text

def main():
    """
    Main function to run the semantic similarity pipeline.
    """
    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset("stsb_multi_mt", name="en", split="train")
    df = dataset.to_pandas()
    df = df.head(10) # Using a small subset for demonstration

    # Translate sentences to Hindi
    print("Translating sentences to Hindi...")
    df['sentence1_hi'] = df['sentence1'].apply(translate_to_hindi)
    df['sentence2_hi'] = df['sentence2'].apply(translate_to_hindi)

    # Load Indic-BERT model and tokenizer
    print("Loading Indic-BERT model...")
    tokenizer = AlbertTokenizer.from_pretrained('ai4bharat/indic-bert')
    model = AutoModel.from_pretrained('ai4bharat/indic-bert')

    # Get sentence embeddings
    print("Generating sentence embeddings...")
    df['embedding1'] = df['sentence1_hi'].apply(lambda x: get_sentence_embedding(x, tokenizer, model))
    df['embedding2'] = df['sentence2_hi'].apply(lambda x: get_sentence_embedding(x, tokenizer, model))

    # Calculate cosine similarity
    print("Calculating cosine similarity...")
    df['cosine_similarity'] = df.apply(lambda row: 1 - cosine(row['embedding1'].squeeze(), row['embedding2'].squeeze()), axis=1)

    # Save results
    output_df = df[['sentence1', 'sentence2', 'sentence1_hi', 'sentence2_hi', 'similarity_score', 'cosine_similarity']]
    output_df.to_csv("semantic_similarity_results.csv", index=False)
    print("Results saved to semantic_similarity_results.csv")

if __name__ == "__main__":
    main()