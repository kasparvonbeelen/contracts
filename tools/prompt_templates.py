def generate_positive_prompts(clause_type, definitions, data, num_examples):
    definition = definitions[clause_type]
    return ["""You are a helpful AI that creates synthetic data by generating new examples a {0} clause. We replaced the name of the company with a "[company]" token and lowercased all characters.

     We first provide a definition followed by five examples of {0} clauses. You need to generate a new sentence similar but NOT identical to these examples and has the same legal function as {0} clause. 
               
     DEFINITON: 
     The defintion of of {0} clause is: {2}

     EXAMPLES:          
     Below are three examples of {0} clauses.\n\n{1}
               
     NEW EXAMPLE:
     """.format(
          clause_type,data[data.labels==1].text.sample(3, random_state=j).str.cat(sep='\n'),definition)  for j in range(num_examples)]

def generate_negative_prompts(clause_type, definitions, data, num_examples):
    definition = definitions[clause_type]
    return  [""" You are a helpful AI that creates synthetic data by generating new examples that look like a {0} clause but have a different meaning. 
    We replaced the name of the company with a "[company]" token and lowercased all characters.

    We first provide three examples of sentences that resemble {0} clauses BUT have a different meaning. 
    You need to generate a new sentence similar but not identical to these examples. They are NOT allowed to be a {0}. 
            
    DEFINITON: 
    The defintion of of {0} clause is: {2}

    EXAMPLES:          
    Below are five examples of {0} clauses.\n\n{1}
            
    NEW EXAMPLE:
    """.format(
        clause_type, data[data.labels==0].text.sample(3, random_state=j).str.cat(sep='\n'), definition
            )  for j in range(num_examples)]