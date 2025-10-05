# Semantic Similarity for Hindi

This project analyzes the semantic similarity between pairs of English sentences by first translating them to Hindi and then comparing their sentence embeddings.

## Features

-   Loads the `stsb_multi_mt` dataset.
-   Translates English sentences to Hindi using `googletrans`.
-   Generates sentence embeddings using the `ai4bharat/indic-bert` model.
-   Calculates the cosine similarity between the sentence embeddings.
-   Saves the results to a CSV file.

## Installation

1.  Clone the repository:
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To run the semantic similarity analysis, execute the `main.py` script:

```bash
python main.py
```

The script will perform the following steps:
1.  Download the necessary dataset and models.
2.  Translate the sentences.
3.  Generate embeddings.
4.  Calculate similarity scores.
5.  Save the output to `semantic_similarity_results.csv`.

## Output

The results are saved in `semantic_similarity_results.csv` with the following columns:

-   `sentence1`: The original first English sentence.
-   `sentence2`: The original second English sentence.
-   `sentence1_hi`: The Hindi translation of the first sentence.
-   `sentence2_hi`: The Hindi translation of the second sentence.
-   `similarity_score`: The original similarity score from the dataset.
-   `cosine_similarity`: The calculated cosine similarity of the Hindi sentence embeddings.