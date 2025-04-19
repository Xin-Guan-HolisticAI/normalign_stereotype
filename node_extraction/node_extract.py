"""Input Sentence → Clause Identification → Key Concept Extraction
→ Concept Replacement → Verification → Result Combination"""
import json
import networkx as nx
from itertools import permutations


from LLMFactory import LLMFactory


def identify_clauses(sentence, llm):
    """
    Split a sentence into its smallest constituent clauses recursively using LLM.

    Args:
        sentence (str): The input sentence
        llm (LLMFactory): The LLM instance to use

    Returns:
        list: List of clauses as strings
    """
    # Run the prompt through the LLM
    response = llm.run_prompt(
        "identify_clauses",
        sentence=sentence
    )

    # Parse the response
    try:
        # Parse the JSON response
        clauses = json.loads(response)

        # Validate that we got a list
        if not isinstance(clauses, list):
            print(f"Warning: Expected a list of clauses but got {type(clauses)}")
            clauses = []

    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON response: {e}")
        # Fallback to simple line splitting if JSON parsing fails
        clauses = [line.strip() for line in response.split('\n') if line.strip()]

    return clauses


def extract_key_concepts(clause, llm):
    """
    Identify key noun concepts (nouns or noun phrases with adjectives)

    Args:
        clause (str): A clause from the sentence
        llm (LLMFactory): The LLM instance to use

    Returns:
        list: List of (concept, position) tuples
    """
    # Run the prompt through the LLM
    response = llm.run_prompt(
        prompt_template_name="extract_key_concepts",
        clause=clause
    )

    # Parse the response to get key concepts
    try:
        key_concepts = json.loads(response)
    except (json.JSONDecodeError, KeyError):
        # Fallback handling if response isn't properly formatted
        key_concepts = []

    return key_concepts


def replace_concepts(clauses, key_concepts):
    """
    Replace key concepts with %# placeholders

    Args:
        clauses (list): list of original clauses
        key_concepts (list): List of list of concepts str

    Returns:
        modified_clause （list): List of str
    """
    modified_clauses = []
    counter = 1

    for i in range(len(clauses)):
        curr_clause = clauses[i]
        curr_key_concepts = key_concepts[i]

        for concept in curr_key_concepts:
            curr_clause = curr_clause.replace(concept, f"{{{counter}}}")
            counter += 1
        modified_clauses.append(curr_clause)
        counter = 1
    return modified_clauses


def create_concept_pairs(key_concepts):
    """
    Create all possible pairs of concepts from a flattened list of concepts.

    Args:
        key_concepts (list): List of lists containing strings

    Returns:
        list: List of lists, each containing a pair of concepts
    """
    # Flatten the list of lists into a single list
    flattened_concepts = []
    for sublist in key_concepts:
        flattened_concepts.extend(sublist)

    # Remove duplicates if needed (optional)
    unique_concepts = list(set(flattened_concepts))

    # Generate all combinations of 2 elements
    concept_pairs = list(permutations(unique_concepts, 2))

    # Convert tuples to lists
    result = [list(pair) for pair in concept_pairs]

    return result


def judge_concepts_relations(concepts_pairs, sentence, llm):
    concepts_relations_reasoning = []
    concepts_relations_judgement = []

    for concept_1, concept_2 in concepts_pairs:
        # Run the prompt through the LLM
        response = llm.run_prompt(
            prompt_template_name="judge_dependency",
            sentence=sentence,
            concept_1=concept_1,
            concept_2=concept_2
        )

        try:
            result = json.loads(response)
            reasoning = result.get("reasoning", "")
            judgement = result.get("answer", "not sure").lower().strip()

            if judgement not in {"yes", "no", "not sure"}:
                judgement = "not sure"

        except json.JSONDecodeError:
            # Fallback parsing
            reasoning = "Failed to parse model response",
            judgement = "not sure"

        concepts_relations_reasoning.append(reasoning)
        concepts_relations_judgement.append(judgement)

    return concepts_relations_reasoning, concepts_relations_judgement


def check_conflicts(concepts_pairs, concepts_relations_judgement):
    # Create a dictionary to store judgments for easy lookup
    judgments = {}
    for pair, judgment in zip(concepts_pairs, concepts_relations_judgement):
        judgments[tuple(pair)] = judgment

    # List to store conflicting concept pairs
    conflict_concepts = []

    # Set to keep track of pairs we've already checked
    checked_pairs = set()

    for i, (pair, judgment) in enumerate(zip(concepts_pairs, concepts_relations_judgement)):
        # Create the reverse pair
        reverse_pair = [pair[1], pair[0]]
        reverse_tuple = tuple(reverse_pair)
        pair_tuple = tuple(pair)

        # Skip if we've already checked this pair or its reverse
        if pair_tuple in checked_pairs or reverse_tuple in checked_pairs:
            continue

        # Mark this pair as checked
        checked_pairs.add(pair_tuple)
        checked_pairs.add(reverse_tuple)

        # Get the judgment for the reverse pair
        reverse_judgment = judgments.get(reverse_tuple)

        # Check for conflict: both "yes"
        if judgment == "yes" and reverse_judgment == "yes":
            conflict_concepts.append(pair)

    return conflict_concepts

def resolve_conflict_relations(conflict_concepts, sentence, llm):
    """
    Resolve conflicts in relations by asking the LLM for clarification.

    Args:
        conflict_concepts (list): List of conflicting concept pairs
        sentence (str): The original sentence
        llm (LLMFactory): The LLM instance to use

    Returns:
        list: List of resolved relations
    """
    resolved_relations = []
    resolved_reasonings = []

    for concept_1, concept_2 in conflict_concepts:
        # Run the prompt through the LLM
        response = llm.run_prompt(
            prompt_template_name="resolve_relationship_conflict",
            sentence=sentence,
            concept_1=concept_1,
            concept_2=concept_2
        )

        try:
            result = json.loads(response)
            resolved_relation = result.get("answer", "not sure").lower().strip()
            resolved_reasoning = result.get("reasoning", "")

            if resolved_relation not in {"yes", "no", "not sure"}:
                resolved_relation = "not sure"
        except json.JSONDecodeError:
            resolved_relation = "not sure"

        resolved_relations.append(resolved_relation)
        resolved_reasonings.append(resolved_reasoning)

    return resolved_relations, resolved_reasonings


def update_relations_judgement(concept_pairs, concepts_relations_judgement, conflict_concepts, resolved_conflicts):
    for conflict_pair, new_judgement in zip(conflict_concepts, resolved_conflicts):
        if new_judgement == 'yes':
            # Find the reverse pair and update its judgement to 'no'
            reverse_pair = [conflict_pair[1], conflict_pair[0]]
            if reverse_pair in concept_pairs:
                index = concept_pairs.index(reverse_pair)
                concepts_relations_judgement[index] = 'no'
        elif new_judgement == 'no':
            # Update the original conflict pair's judgement to 'no'
            if conflict_pair in concept_pairs:
                index = concept_pairs.index(conflict_pair)
                concepts_relations_judgement[index] = 'no'

    return concepts_relations_judgement


# def generate_and_save_dot(concept_pairs, concepts_relations_judgement, filename="concept_graph.dot"):
#     """
#     Generate a DOT file from concept pairs and relation judgements, saving to specified filename.
#
#     Args:
#         concept_pairs (list): List of concept pairs
#         concepts_relations_judgement (list): List of relation judgements
#         filename (str): Output filename (default: 'concept_graph.dot')
#     """
#     dot_content = ['digraph G {']
#
#     # Add all unique nodes first
#     unique_concepts = set()
#     for pair in concept_pairs:
#         unique_concepts.add(pair[0])
#         unique_concepts.add(pair[1])
#
#     # Format nodes (handle concepts with spaces)
#     for concept in sorted(unique_concepts):
#         dot_content.append(f'    "{concept}";')
#
#     # Add edges for "yes" relations
#     for pair, judgement in zip(concept_pairs, concepts_relations_judgement):
#         if judgement == 'yes':
#             source = f'"{pair[1]}"'
#             target = f'"{pair[0]}"'
#             dot_content.append(f'    {source} -> {target};')
#
#     dot_content.append('}')
#
#     # Write to file
#     with open(filename, 'w') as f:
#         f.write('\n'.join(dot_content))
#
#     print(f"DOT file saved as {filename}")




def generate_and_save_dot(concept_pairs, concepts_relations_judgement, filename="concept_graph.dot", prune=False):
    """
    Generate a DOT file with edges pointing FROM concept2 TO concept1
    (representing "need concept2 to understand concept1")
    """
    # Create a directed graph
    G = nx.DiGraph()

    # First, add all unique concepts as nodes
    unique_concepts = set()
    for concept_1, concept_2 in concept_pairs:
        unique_concepts.add(concept_1)
        unique_concepts.add(concept_2)

    # Add all concepts as nodes
    for concept in unique_concepts:
        G.add_node(concept)

    # Add edges where judgment is 'yes'
    # Note: edges point from concept_2 to concept_1 as specified
    for (concept_1, concept_2), judgment in zip(concept_pairs, concepts_relations_judgement):
        if judgment == 'yes':
            G.add_edge(concept_2, concept_1)

    # Prune transitive edges if requested
    if prune:
        transitive_closure = nx.transitive_closure(G)
        edges_to_remove = []

        for u, v in G.edges():
            # Check for indirect paths (transitive edges)
            for intermediate in G.nodes():
                if (intermediate != u and intermediate != v and
                        transitive_closure.has_edge(u, intermediate) and
                        transitive_closure.has_edge(intermediate, v)):
                    edges_to_remove.append((u, v))
                    break

        for u, v in edges_to_remove:
            G.remove_edge(u, v)

    # Generate DOT file content
    dot_content = ["digraph G {"]
    for u, v in G.edges():
        dot_content.append(f'    "{u}" -> "{v}";')
    dot_content.append("}")

    # Save to file
    with open(filename, 'w') as f:
        f.write('\n'.join(dot_content))

    print(f"DOT file saved as {filename}")


def process_sentence(sentence, model_name="qwen-turbo-latest"):
    """
    Process a sentence through the entire pipeline

    Args:
        sentence (str): Input sentence
        model_name (str): Name of the model to use from settings.yaml

    Returns:
        list: List of [clause, verification_result] pairs
    """
    # Initialize LLM
    llm = LLMFactory(model_name)

    # Step 1: Identify clauses
    clauses = identify_clauses(sentence, llm)
    print(clauses)

    # Step 2: Extract key concepts
    key_concepts = []
    # Process each clause
    for clause in clauses:
        key_concepts.append(extract_key_concepts(clause, llm))
    print(key_concepts)

    # Step 3: Replace concepts
    modified_clause = replace_concepts(clauses, key_concepts)
    print(modified_clause)

    # Step 4: Build Matrix
    concepts_pairs = create_concept_pairs(key_concepts)
    print(concepts_pairs)

    # Step 5: Judge the Relationship between concept pairs
    concepts_relations_reasoning, concepts_relations_judgement = judge_concepts_relations(concepts_pairs, sentence, llm)
    print(concepts_relations_reasoning)
    print(concepts_relations_judgement)

    # Step 6: Check if the relationships are consistent
    conflict_concepts = check_conflicts(concepts_pairs, concepts_relations_judgement)
    print(conflict_concepts)

    # Step 7: Resolve the conflict relations
    resolved_conflicts, resolved_reasonings = resolve_conflict_relations(conflict_concepts, sentence, llm)
    print(resolved_conflicts)
    print(resolved_reasonings)

    # Step 8: Update The relation judgement
    concepts_relations_judgement = update_relations_judgement(concepts_pairs, concepts_relations_judgement, conflict_concepts, resolved_conflicts)
    print(concepts_relations_judgement)

    # Step 9: Generate the dot file
    generate_and_save_dot(concepts_pairs, concepts_relations_judgement, "original_concept_graph.dot")

    # Step 10: Generate the pruned dot file
    generate_and_save_dot(concepts_pairs, concepts_relations_judgement, "pruned_concept_graph.dot", True)

if __name__ == "__main__":
    # sentence = "The cat sleeps when the dog barks, and the bird sings."
    sentence = "From a statement, stereotype is a false generalizations of a target group if the generalizations does not apply to some individuals from the target group."
    # Process with default model
    result = process_sentence(sentence)
    # print("Results:")
    # for modified_clause, is_valid in result:
    #     print(f"- {modified_clause} (Valid: {is_valid})")
