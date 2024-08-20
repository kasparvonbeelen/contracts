from collections import defaultdict
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import scipy.spatial as sp
import pandas as pd
import numpy as np
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
        x = np.apply_along_axis(np.max,0,mult)
        y = np.apply_along_axis(np.max,1,mult)
        
        resultdict[year_2_dt]['copied'] = len(np.where(y >= threshold)[0])
        resultdict[year_2_dt]['deletions'] = len(np.where(x < threshold)[0])
        resultdict[year_2_dt]['additions'] = len(np.where(y < threshold)[0])
        resultdict[year_2_dt]['length'] = len(y)
        resultdict[year_2_dt]['length_t_min_1'] = len(x)
        resultdict[year_2_dt]['future_projection'] = x
        resultdict[year_2_dt]['past_projection'] = y
        resultdict[year_2_dt]['matrix'] = mult
    
    result_df = pd.DataFrame.from_dict(resultdict).T
    result_df['platform'] = platform
    return result_df

def compare_timelines(metadata, embeddings,platforms, threshold):
    result_df = pd.concat([get_timeline(metadata, embeddings, p, threshold) for p in platforms])
    result_df['rel_copied'] = result_df['copied'] / result_df['length']
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

def convergence(metadata, embeddings, platform_t,platform_c):
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
            resultdict[d]['mean'] = np.mean(x)
            resultdict[d]['similar'] = len(x[x > .9]) / len(x)
            resultdict[d]['matrix'] = mult

    return pd.DataFrame(resultdict).T