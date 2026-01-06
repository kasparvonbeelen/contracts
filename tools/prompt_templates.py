# general prompt structure
# state the task for both positive and negative examples
# state the structure of the prompt and repeat the task
# list components of the prompt
# hint at the answer

def generate_positive_prompts_zero_shot(clause_type, definitions, num_examples):
    definition = definitions[clause_type]
    return ["""You are a helpful AI that generates a new example a {0} clause.

We give a definition of a {0} clause. You need to generate a new example of a {0} clause.
               
DEFINITON: 
The defintion of a {0} clause is: {1}

NEW {0} CLAUSE:
""".format(
          clause_type,definition)  for j in range(num_examples)]

def generate_negative_prompts_zero_shot(clause_type, definitions, num_examples):
    definition = definitions[clause_type]
    return ["""You are a helpful AI that generates legal clauses for terms of use.

You need to generate a sentence that resembles clauses from terms of use contracts.
The examples are NOT allowed to be a {0} clause. 

NEW EXAMPLE:
""".format(
          clause_type,definition)  for j in range(num_examples)]



def generate_positive_prompts_few_shot(clause_type, definitions, data, num_examples):
    definition = definitions[clause_type]
    return ["""You are a helpful AI that generates a new example a {0} clause.

We first give a definition of a {0} clause followed by three examples. 

You must generate a new example of a {0} clause.
You can combine elements of the three examples, but have to change the word order, use synonyms, and change the sentence structure. 
The end result, however, has to remain a {0} clause from a legal and semantic point of view. 
               
DEFINITON: 
The defintion of a {0} clause is: {2}

EXAMPLES:          
Below are three examples of {0} clauses. One line per example.\n\n{1}
               
NEW {0} CLAUSE:
""".format(
          clause_type,data[data.labels==1].text.sample(3, random_state=j).str.cat(sep='\n'),definition)  for j in range(num_examples)]


def generate_negative_prompts_few_shot(clause_type, definitions, data, num_examples):
    definition = definitions[clause_type]
    return  ["""You are a helpful AI that generates legal clauses for terms of use.

You need to generate a sentence that resembles example clauses provided below. 
You can combine elements of the these examples, but have to change the word order, use synonyms, and change the sentence structure. 
The new example is NOT allowed to be a {0} clause.  

EXAMPLES:          
Below are three examples clause. One line per example.\n\n{1}
            
NEW EXAMPLE:
""".format(
        clause_type, data[data.labels==0].text.sample(3, random_state=j).str.cat(sep='\n'), definition
            )  for j in range(num_examples)]



def generate_positive_prompts_contrastive_few_shot(clause_type, definitions, data, num_examples):
    definition = definitions[clause_type]
    return ["""You are a helpful AI that generates a new example a {0} clause.

We provide you with the following information:
- DEFINITION: We give a definition of a {0} clause
- EXAMPLES OF {0} CLAUSES: We give three examples of {0} clauses
- CONTRASTIVE EXAMPLES: We give three contrastive examples, which are text fragments that resemble {0} clauses BUT DO NOT have the same legal meaning or function. 

You must generate a new example of a {0} clause.
You can combine elements of the three {0} clauses, but have to change the word order, use synonyms, and change the sentence structure. 
Make sure the new example is very different from the contrastive examples.
The end result, however, has to remain a {0} clause from a legal and semantic point of view. 
            

DEFINITON: 
The defintion of of {0} clause is: {1}

EXAMPLES OF {0} CLAUSES:          
Below are three examples of a {0} clause. One example per line.\n\n{2}
            
CONTRASTIVE EXAMPLES:
Below are three examples of contrastive examples. One example per line.\n\n{3}
            
NEW EXAMPLE OF {0} CLAUSE:
     """.format(
          clause_type,
          definition,
          data[data.labels==1].text.sample(3, random_state=j).str.cat(sep='\n'),
          data[data.labels==0].text.sample(3, random_state=j).str.cat(sep='\n'))  for j in range(num_examples)]





def generate_negative_prompts_contrastive_few_shot(clause_type, definitions, data, num_examples):
    definition = definitions[clause_type]
    return ["""You are a helpful AI that generates legal clauses for terms of use.

We provide you with the following information:
- EXAMPLES OF {0} CLAUSES: We give three examples of {0} clauses
- CONTRASTIVE EXAMPLES: We give three contrastive examples, which are text fragments that might resemble {0} clauses BUT DO NOT have the same legal meaning or function. 

You must generate a new example of a contrastive clause.
You can combine elements of the these contrastive examples, but have to change the word order, use synonyms, and change the sentence structure. 
Make sure the new example is very different from the examples {0} clauses. 
The new example is NOT allowed to be a {0} clause.  

EXAMPLES OF {0} CLAUSES:          
Below are three examples of a {0} clause. One example per line.\n\n{2}
            
CONTRASTIVE EXAMPLES:
Below are three examples of contrastive examples. One example per line.\n\n{3}
            
NEW CONTRASTIVE EXAMPLE:
     """.format(
          clause_type,
          definition,
          data[data.labels==1].text.sample(3, random_state=j).str.cat(sep='\n'),
          data[data.labels==0].text.sample(3, random_state=j).str.cat(sep='\n'))  for j in range(num_examples)]

########################
##### Iteration 2 ######
########################

def generate_positive_prompts_few_shot_syn(clause_type, definitions, data, syn_data, num_examples):
    definition = definitions[clause_type]
    return ["""You are a helpful AI that creates synthetic data by generating a new example a {0} clause.

We first give a definition of a {0} clause followed by five synthetic examples. 
You must generate a new example of a {0} clause.
You can combine elements of the three examples, and have to change the word order, use synonyms, and change the sentence structure. 
The end result, however, has to remain a {0} clause from a legal and semantic point of view. 
               
DEFINITON: 
The defintion of a {0} clause is: {2}

SYNTHETIC EXAMPLES:
Below are five synthetic examples of a {0} clause. One line per example.\n 
Make sure your new example is uses different words but remains a {0} clause.\n\n
{3}            

NEW {0} CLAUSE:
""".format(
          clause_type,
          data[data.labels==1].text.sample(3, random_state=j).str.cat(sep='\n'),
          definition,
          syn_data[syn_data.task=='positive_prompts_few_shot'].text.sample(5, random_state=j).str.cat(sep='\n'))  for j in range(num_examples)]


def generate_negative_prompts_few_shot_syn(clause_type, definitions, data, syn_data, num_examples):
    definition = definitions[clause_type]
    return  ["""You are a helpful AI that creates synthetic data by generating a contrastive example of {0} clauses, which is a short text that looks like a {0} clause but legally has a different function and meaning.

We first give a definition of a {0} clause followed by five synthetic contrastive examples, which are text fragments that resemble {0} clauses BUT DO NOT have the same legal meaning or function. 
You need to generate a sentence that is similar to these constrastive examples but using different words.
The examples are NOT allowed to be a {0} clause.  
            
DEFINITON: 
The defintion of of {0} clause is: {2}

SYNTHETIC EXAMPLES:
Below are five examples of synthetic data. One line per example.\n Make sure your new example is VERY different from these examples but resembles the style of contract and is not a {0} clause.\n\n
{3}                  

NEW CONTRASTIVE EXAMPLE:
""".format(
          clause_type,
          data[data.labels==1].text.sample(3, random_state=j).str.cat(sep='\n'),
          definition,
          syn_data[syn_data.task=='negative_prompts_few_shot'].text.sample(5, random_state=j).str.cat(sep='\n'))  for j in range(num_examples)]

# Old prompts

# def generate_negative_prompts_few_shot(clause_type, definitions, data, num_examples):
#     definition = definitions[clause_type]
#     return  ["""You are a helpful AI that creates synthetic data by generating a contrastive example of {0} clauses, which is a short text that looks like a {0} clause but legally has a different function and meaning.

# We first give a definition of a {0} clause followed by three contrastive examples, which are text fragments that resemble {0} clauses BUT DO NOT have the same legal meaning or function. 
# You need to generate a sentence that is similar to these constrastive examples but uses different words.
# The examples are NOT allowed to be a {0} clause.  
            
# DEFINITON: 
# The defintion of of {0} clause is: {2}

# EXAMPLES:          
# Below are three contrastive examples that resemble this definition but are NOT {0} clauses. One line per example.\n\n{1}
            
# NEW CONTRASTIVE EXAMPLE:
# """.format(
#         clause_type, data[data.labels==0].text.sample(3, random_state=j).str.cat(sep='\n'), definition
#             )  for j in range(num_examples)]

# def generate_negative_prompts_contrastive_few_shot(clause_type, definitions, data, num_examples):
#     definition = definitions[clause_type]
#     return ["""You are a helpful AI that creates synthetic data by generating a contrastive example of {0} clauses, which is a short text that looks like a {0} clause but legally has a different function and meaning. 

# We provide you with the following information:
# - DEFINITION: We give a definition of a {0} clause
# - EXAMPLES OF {0} CLAUSES: We give three examples of {0} clauses
# - CONTRASTIVE EXAMPLES: We give three contrastive examples, which are text fragments that resemble {0} clauses BUT DO NOT have the same legal meaning or function. 

# You need to generate a sentence that is similar to these constrastive examples but uses different words.
# The examples are NOT allowed to be a {0} clause.  
               
# DEFINITON: 
# The defintion of of {0} clause is: {1}

# EXAMPLES OF {0} CLAUSES:          
# Below are three examples of a {0} clause. One example per line.\n\n{2}
            
# CONTRASTIVE EXAMPLES:
# Below are three examples of contrastive examples. One example per line.\n\n{3}
            
# NEW CONTRASTIVE EXAMPLE:
#      """.format(
#           clause_type,
#           definition,
#           data[data.labels==1].text.sample(3, random_state=j).str.cat(sep='\n'),
#           data[data.labels==0].text.sample(3, random_state=j).str.cat(sep='\n'))  for j in range(num_examples)]
