import spacy

# Load English language model
nlp = spacy.load("en_core_web_sm")

def split_clauses_recursive(text):
    """Recursively splits a sentence into clauses using dependency parsing."""
    doc = nlp(text)
    clauses = []
    current_clause = []
    split_found = False

    for token in doc:
        # Check for clause boundaries: coordinating conjunctions (CCONJ) or subordinating markers
        if token.dep_ in ("cc", "mark") or token.pos_ == "CCONJ":
            # Ensure the split point is valid (e.g., both sides have a verb)
            left_clause = doc[token.left_edge.i : token.i]
            right_clause = doc[token.i + 1 : token.right_edge.i + 1]
            if _has_verb(left_clause) and _has_verb(right_clause):
                clauses.append(" ".join(current_clause))
                clauses.extend(split_clauses_recursive(" ".join([t.text for t in right_clause])))
                split_found = True
                break
            else:
                current_clause.append(token.text)
        else:
            current_clause.append(token.text)

    if not split_found:
        clauses.append(" ".join(current_clause))
    else:
        return clauses

    # Recursively split subclauses
    final_clauses = []
    for clause in clauses:
        sub_clauses = split_clauses_recursive(clause)
        if len(sub_clauses) > 1:
            final_clauses.extend(sub_clauses)
        else:
            final_clauses.append(clause)
    return final_clauses

def _has_verb(span):
    """Checks if a span contains at least one verb."""
    return any(token.pos_ == "VERB" for token in span)

sentence = "The cat sleeps when the dog barks, and the bird sings."
result = split_clauses_recursive(sentence)
print(result)