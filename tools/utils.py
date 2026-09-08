from collections import defaultdict
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import scipy.spatial as sp
import pandas as pd
import numpy as np
import re


# Registry of supported embedding models. Each entry gives the HF repo id,
# whether trust_remote_code is needed to load it, and the text prefix this
# model expects for a symmetric clustering/similarity task (models that are
# trained with task-instruction prefixes need this to get on-task embeddings).
EMBEDDING_MODELS = {
    "nomic-embed-text-v1.5": {
        "repo_id": "nomic-ai/nomic-embed-text-v1.5",
        "trust_remote_code": True,
        "prefix": "clustering:  ",
    },
    "qwen3-embedding-0.6b": {
        "repo_id": "Qwen/Qwen3-Embedding-0.6B",
        "trust_remote_code": False,
        "prefix": "",  # Qwen3-Embedding needs no prefix for document/corpus-side text
    },
    "embeddinggemma": {
        "repo_id": "google/embeddinggemma-300m",
        "trust_remote_code": False,
        "prefix": "task: clustering | query: ",
    },
    "bge-large-en-v1.5": {
        "repo_id": "BAAI/bge-large-en-v1.5",
        "trust_remote_code": False,
        "prefix": "",  # BGE needs no prefix for symmetric tasks (clustering/similarity); only retrieval queries take an instruction prefix
    },
}


def load_embedding_model(name: str, device: str = "cpu"):
    """Load a sentence-transformers embedding model by short name.

    Arguments:
        name: a key in EMBEDDING_MODELS, or a raw HF repo id for a model
            not in the registry (loaded with trust_remote_code=False and
            no text prefix)
        device: torch device to move the model to

    Returns:
        tuple: (model, prefix) - the loaded SentenceTransformer and the
        text prefix to prepend before encoding, per that model's convention
    """
    config = EMBEDDING_MODELS.get(name, {"repo_id": name, "trust_remote_code": False, "prefix": ""})
    model = SentenceTransformer(config["repo_id"], trust_remote_code=config["trust_remote_code"])
    model.to(device)
    return model, config["prefix"]



def generate_response(prompt,client,model='gpt-4o', max_tokens=100,temperature=.0):
    # Generate a response using OpenAI ChatGPT
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
        n=1,
        stop=None,
        timeout=10
    )

    # Extract the generated response from the API response
    #generated_response = response.choices[0].text.strip()

    return response.choices[0].message.content



def parse_filename(fn: str) -> tuple:
    """function for processing the filename of the input text files
    
    Arguments:
        n: str, filename
    
    Returns:
        tuple: platform, date as integer
    """
    el = fn.split('_')
    platform = el[0]

    # make sure each file has month and day
    month, day = '01','01'
    
    if len(el) == 4:
        d = el[1]+el[2]+day  # we ignore the day always replace with 01
    
    elif len(el) == 3:
        d = el[1]+el[2]+day
    
    elif len(el) == 2:
        d = el[1]+month+day
    
    return platform,int(d)

def replace_named_entities(doc) -> str:
    """function for replacing named entities with [MASK] token
    Arguments:
        text: str, input text
        nlp: spacy model
    """
   
    replaced_text = ""
    for token in doc:
        if token.ent_type_:
            replaced_text += "[mask] "
        else:
            replaced_text += token.text + " "
    return replaced_text.strip().lower()

def remove_urls(text):
    """function for removing urls from the text
    Arguments:
        text: str, input text
    Returns:
        str : cleaned text
    """
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    cleaned_text = re.sub(url_pattern, '', text)
    
    return cleaned_text

def process_data(data : str, nlp, model, prefix: str = "clustering:  ") -> tuple:
    """function for processing the input text data
    give a folder name it will read all the text files and the folder names
    for each text file it will read the text and split it into sentences
    then it will replace named entities with [mask] token and remove the platform name and urls
    finally it will return the embeddings of the sentences and the metadata as a dataframe
    with columns: platform, year, sentence

    Arguments:
        data: str, folder name
        nlp: spacy model
        model: sentence transformer model
        prefix: text prefix to prepend before encoding, matching the
            embedding model's expected task-instruction convention (see
            EMBEDDING_MODELS / load_embedding_model)
    Returns:
        tuple: embeddings, metadata

    """

    data = Path(data)
    txt_paths = list(data.glob('**/*.txt'))
    print(f'{len(txt_paths)} contracts in the corpus.')
    embeddings, metadata = [], []
    for f in tqdm(txt_paths):    
        platform, year = parse_filename(f.stem)
    
        with open(f) as in_txt:
            text = in_txt.read().strip()
        
        doc = nlp(text)
        
        for sentence  in doc.sents:
            if len(str(sentence)) < 10: continue
            
            sentence_processed = replace_named_entities(sentence) # remove named entities
            #sentence_processed = sentence_processed.lower().replace(platform.lower(),'[mask]') # make double sure the platform name is masked    
            sentence_processed = re.sub(rf"\b{platform.lower()}\b", '[mask]', sentence_processed) # make double sure the platform name is masked using regex
            sentence_processed = remove_urls(sentence_processed) # remove urls
            metadata.append([platform,year,sentence,sentence_processed])
            sentence = remove_urls(str(sentence)) # remove urls
            embeddings.append(model.encode(prefix + sentence.lower()))

    df = pd.DataFrame(metadata, columns=['platform','year','sentence','sentence_processed'])
    print(f'Embedded {len(df)} sentences...')
    return embeddings, df



def get_timeline(metadata, embeddings, platform, threshold):
    resultdict = defaultdict(dict)
    dates = sorted(metadata[metadata.platform==platform].year.unique())
    for i in tqdm(range(len(dates)-1)):
        year_1, year_2 = dates[i],dates[i+1]

        year_1_dt = datetime.strptime(str(year_1), "%Y%m%d").date()
        year_2_dt = datetime.strptime(str(year_2), "%Y%m%d").date()
        idx_1 = list(metadata[(metadata.platform==platform) & (metadata.year ==year_1)].index)
        idx_2 = list(metadata[(metadata.platform==platform) & (metadata.year ==year_2)].index)
        mult = 1 - sp.distance.cdist(embeddings[idx_1,:], embeddings[idx_2,:], 'cosine')
        y = np.apply_along_axis(np.max,0,mult)
        x = np.apply_along_axis(np.max,1,mult)
        
        resultdict[year_2_dt]['copied'] = len(np.where(y >= threshold)[0])
        resultdict[year_2_dt]['deletions'] = len(np.where(x < threshold)[0])
        resultdict[year_2_dt]['additions'] = len(np.where(y < threshold)[0])
        resultdict[year_2_dt]['length'] = len(idx_2)
        resultdict[year_2_dt]['length_t_min_1'] = len(idx_1)
        resultdict[year_2_dt]['length_diff'] = len(idx_2) - len(idx_1)
        resultdict[year_2_dt]['future_projection'] = x
        resultdict[year_2_dt]['past_projection'] = y
        resultdict[year_2_dt]['matrix'] = mult
    
    result_df = pd.DataFrame.from_dict(resultdict).T
    result_df['platform'] = platform
    return result_df

def compare_timelines(metadata, embeddings,platforms, threshold):
    result_df = pd.concat([get_timeline(metadata, embeddings, p, threshold) for p in platforms])
    result_df['rel_copied'] = result_df['copied'] / result_df['length']
    result_df['rel_added'] = result_df['length_diff'] / result_df['length']
    result_df['rel_additions'] = result_df['additions'] / result_df['length']
    result_df['rel_deletions'] = result_df['deletions'] / result_df['length_t_min_1']
    return result_df

def negative_closest_to_zero(lst):
    # Filter the list to keep only negative values
    negative_values = [(x,y) for x,y in lst if y < 0]

    # If there are no negative values, return None
    if not negative_values:
        return None, None

    # Find the negative value closest to zero
    closest_negative = min(negative_values, key=lambda x: abs(x[1]))

    return closest_negative

def convergence(metadata, embeddings, platform_t,platform_c, threshold=.9):
    """
    """
    dates_t = sorted(metadata[metadata.platform==platform_t].date.unique())
    dates_c = sorted(metadata[metadata.platform==platform_c].date.unique())
    resultdict = defaultdict(dict)
    for i,d in enumerate(dates_t):
        days_delta = [(dc,(dc - d).days) for dc in dates_c]
        dc, days = negative_closest_to_zero(days_delta)
        if dc:

            metadata[(metadata.date == dc) & (metadata.platform==platform_c)]

            idx_1 = list(metadata[(metadata.platform==platform_t) & (metadata.date ==d)].index)
            idx_2 = list(metadata[(metadata.platform==platform_c) & (metadata.date ==dc)].index)
            #print(i,len(idx_1),len(idx_2),d,dc, days)
            mult = 1 - sp.distance.cdist(embeddings[idx_1,:], embeddings[idx_2,:], 'cosine')
            x = np.apply_along_axis(np.max,0,mult)
            resultdict[d]['date'] = d
            resultdict[d]['date_gap'] = abs(days)
            resultdict[d]['mean'] = np.mean(x)
            resultdict[d]['similar'] = len(x[x > threshold]) / len(x)
            resultdict[d]['matrix'] = mult

    return pd.DataFrame(resultdict).T


def replace_minus_ones_with_prev(X, axis=1, inplace=False):
    """
    Replace -1 entries in a matrix/array with the nearest preceding 0 or 1 along the given axis.
    If there is no preceding non -1 value, the -1 is left unchanged.

    Parameters:
    - X: array-like (numpy array, list of lists, or pandas DataFrame)
    - axis: 1 to replace along rows (left-to-right), 0 to replace along columns (top-to-bottom)
    - inplace: if True and X is a numpy array or DataFrame, modify it in place; otherwise return a new array

    Returns:
    - numpy.ndarray or pandas.DataFrame with replacements applied (unless inplace=True modifies input)
    """


    is_df = pd is not None and isinstance(X, pd.DataFrame)
    if is_df:
        arr = X.values
    else:
        arr = X if isinstance(X, (np.ndarray,)) else np.array(X)

    if not inplace:
        arr = arr.copy()

    if axis not in (0, 1):
        raise ValueError("axis must be 0 or 1")

    # iterate over the chosen axis and carry forward the last seen non -1 value
    if axis == 1:
        # rows
        for r in range(arr.shape[0]):
            last = None
            for c in range(arr.shape[1]):
                val = arr[r, c]
                if val != -1:
                    last = val
                elif last is not None:
                    arr[r, c] = last
    else:
        # columns
        for c in range(arr.shape[1]):
            last = None
            for r in range(arr.shape[0]):
                val = arr[r, c]
                if val != -1:
                    last = val
                elif last is not None:
                    arr[r, c] = last

    if is_df:
        if inplace:
            X.iloc[:, :] = arr
            return X
        else:
            return pd.DataFrame(arr, index=X.index, columns=X.columns)
    else:
        return arr
