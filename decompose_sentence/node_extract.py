import spacy
from typing import List, Dict

# Load the spaCy English model
nlp = spacy.load("en_core_web_sm")

def clause_decomposition(sentence: str) -> List[str]:
    """
    Decomposes the input sentence into a list of clauses.
    
    The approach here is a simple heuristic: we search for subordinating 
    conjunction markers (like "if", "because", "when", etc.) that are labeled
    as a marker (dep == "mark"). If found, we treat the text before the first 
    marker as one clause (typically the main clause) and the text from the marker 
    onward as the subordinate clause. 
    
    For a sentence without such markers, the entire sentence is returned as a single clause.
    """
    doc = nlp(sentence)
    subordinate_markers = {"if", "because", "while", "although", "since", "when"}
    marker_index = None
    for token in doc:
        if token.text.lower() in subordinate_markers and token.dep_ == "mark":
            marker_index = token.i
            break

    if marker_index is None:
        return [sentence.strip()]

    # The main clause is everything before the first subordinate marker.
    main_clause = doc[:marker_index].text.strip()
    # The subordinate clause is from the marker until the end.
    subordinate_clause = doc[marker_index:].text.strip()
    
    return [main_clause, subordinate_clause]

def extract_components(clause: str) -> Dict[str, str]:
    """
    Applies a subject–predicate–complement (S-P-C) extraction on the given clause.
    
    - The **subject** is identified as the first token with a dependency label in {"nsubj", "nsubjpass"}.
    - The **predicate** is based on the ROOT token of the clause and includes auxiliaries and negations.
    - The **complement** is searched for among children of the ROOT with dependency labels like "attr", "dobj", 
      or found within a prepositional phrase (e.g., "of a target group").
    
    Returns a dictionary with keys "subject", "predicate", and "complement".
    """
    doc = nlp(clause)
    subject = None
    predicate = None
    complement = None

    # 1. Find the subject: typically a token with the dependency "nsubj" or "nsubjpass"
    for token in doc:
        if token.dep_ in {"nsubj", "nsubjpass"}:
            # Expand to cover the full subtree of the subject for a more complete phrase.
            subject = doc[token.left_edge.i: token.right_edge.i + 1].text
            break

    # 2. Identify the predicate: we use the token that is the sentence ROOT,
    #    including attached auxiliaries and negations.
    root = None
    for token in doc:
        if token.dep_ == "ROOT":
            root = token
            break
    if root:
        aux_parts = []
        for child in root.children:
            if child.dep_ in {"aux", "auxpass", "neg"}:
                aux_parts.append(child.text)
        # Combine the auxiliaries/negations with the main verb.
        predicate = " ".join(aux_parts + [root.text])

    # 3. Extract a complement: look for an attribute or direct object.
    if root:
        for child in root.children:
            if child.dep_ in {"attr", "dobj"}:
                complement = doc[child.left_edge.i: child.right_edge.i + 1].text
                break
            # Alternatively, if the child is a preposition (like "of"),
            # look for its object.
            if child.dep_ == "prep":
                for subchild in child.children:
                    if subchild.dep_ == "pobj":
                        complement = doc[child.left_edge.i: subchild.right_edge.i + 1].text
                        break
                if complement:
                    break

    return {
        "subject": subject,
        "predicate": predicate,
        "complement": complement
    }

def decompose_and_extract(sentence: str) -> List[Dict[str, str]]:
    """
    Combines clause decomposition with S-P-C extraction.
    
    First, the sentence is split into clauses (for example, a main clause and a conditional clause).
    Then, each clause is processed to extract its subject, predicate, and complement. 
    The original clause text is also included in the results.
    
    Returns a list of dictionaries, one per clause.
    """
    clauses = clause_decomposition(sentence)
    components_list = []
    for clause in clauses:
        components = extract_components(clause)
        components["clause"] = clause  # add the original clause text for reference
        components_list.append(components)
    return components_list

# Test the functions with the provided sentence
if __name__ == "__main__":
    test_sentence = (
        "stereotype is a false generalizations of a target group if the generalizations does not "
        "apply to some individuals from the target group"
    )
    
    print("Decomposed Clauses and Extracted Components:")
    results = decompose_and_extract(test_sentence)
    
    for i, comp in enumerate(results, 1):
        print(f"\nClause {i}: {comp['clause']}")
        print(f"  Subject   : {comp['subject']}")
        print(f"  Predicate : {comp['predicate']}")
        print(f"  Complement: {comp['complement']}")
