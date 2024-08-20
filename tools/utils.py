
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import re

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

def process_data(data : str, nlp, model) -> tuple:
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
    Returns:
        tuple: embeddings, metadata

    """

    data = Path('data')
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

            sentence = replace_named_entities(sentence) # remove named entities
            sentence = sentence.lower().replace(platform.lower(),'[mask]') # make double sure the platform name is masked    
            sentence = remove_urls(sentence) # remove urls
            metadata.append([platform,year,sentence])
            embeddings.append(model.encode("clustering:  " + sentence))

    df = pd.DataFrame(metadata, columns=['platform','year','sentence'])    
    print(f'Embedded {len(df)} sentences...')
   
    return embeddings, df