import pandas as pd
import os

def prepare_dataset():
    print("Loading downloaded Kaggle dataset...")
    # Pointing exactly to the file you downloaded
    input_path = os.path.join('dataset', 'phishing_site_urls.csv')
    
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Could not find {input_path}.")
        return

    print("Columns found in dataset:", df.columns.tolist())
    
    # The exact column names from your screenshot
    url_column = 'URL'     
    label_column = 'Label' 
    
    clean_df = pd.DataFrame()
    clean_df['url'] = df[url_column]
    
    # Translating Kaggle's 'bad'/'good' to our 'phishing'/'legitimate'
    label_mapping = {
        'bad': 'phishing', 
        'good': 'legitimate',
        'phishing': 'phishing',
        'legitimate': 'legitimate',
        1: 'phishing',
        0: 'legitimate'
    }
    
    clean_df['label'] = df[label_column].map(label_mapping)
    clean_df = clean_df.dropna()

    output_path = os.path.join('dataset', 'urls.csv')
    clean_df.to_csv(output_path, index=False)
    
    print(f"Success! Formatted {len(clean_df)} URLs and saved to {output_path}")

if __name__ == "__main__":
    prepare_dataset()