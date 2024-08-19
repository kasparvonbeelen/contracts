

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

def replace_named_entities(text: str, nlp) -> str:
    """function for replacing named entities with [MASK] token
    Arguments:
        text: str, input text
        nlp: spacy model
    """
    doc = nlp(text)
    replaced_text = ""
    for token in doc:
        if token.ent_type_:
            replaced_text += "[mask] "
        else:
            replaced_text += token.text + " "
    return replaced_text.strip().lower()