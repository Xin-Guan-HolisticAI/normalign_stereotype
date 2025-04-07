from textblob import TextBlob
import nltk
import logging

def _mock_get_phrase_pos(phrase):
    concept_list = [
        "statement", "generalized_belief", "target_group", "individual",
        "attribute", "not_possess", "not_possess_attribute",
        "true_that_of_generalized_belief_individual_not_possess_attribute",
        "false_generalization_from_generalized_belief", "answer",
    ]
    noun_list = [
        "statement", "generalized_belief", "target_group", "individual",
        "attribute",
        "true_that_of_generalized_belief_individual_not_possess_attribute",
        "false_generalization_from_generalized_belief", "answer",
    ]
    if not phrase in concept_list:
        print(phrase)
        raise ValueError("can not mock pos")

    if phrase in noun_list:
        return "noun"
    else:
        return "verb"






nltk.downloader._downloader._quiet = True  # Disable downloader output

# Attempt to load datasets (errors still occur but aren't printed)
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('averaged_perceptron_tagger_eng')
except LookupError:
    pass  # Handle missing data if needed

# # Disable the NLTK data loader logger
# nltk_logger = logging.getLogger('nltk_data')
# nltk_logger.setLevel(logging.ERROR)  # Suppress messages below CRITICAL
#
# # Suppress download messages
# nltk.download('punkt_tab', quiet=True)
# nltk.download('averaged_perceptron_tagger_eng', quiet=True)


grammar_config = {
    # Noun Phrase (NP): Determiner + Adjectives + Noun(s)
    "noun": [
        ["DT", "JJ", "JJ", "NN"],  # e.g., "the quick brown fox"
        ["DT", "JJ", "NN"],  # e.g., "a happy child"
        ["DT", "NN"],  # e.g., "the dog"
        ["NN"],  # e.g., "dogs"
        ["NNS"],  # e.g., "apples"
        ["NNP"],  # e.g., "London"
        ["PRP$", "NN"],  # e.g., "her book"
        ["CD", "NNS"],  # e.g., "three cats"
        ["DT", "JJ", "NNS"],  # e.g., "some old books"
        ["DT", "NN", "NN"],  # e.g., "chocolate cake"
    ],

    # Verb Phrase (VP): Verb + Adverb/Particle
    "verb": [
        ["VB"],  # e.g., "run"
        ["VBD"],  # e.g., "jumped"
        ["VBG"],  # e.g., "running"
        ["VB", "RB"],  # e.g., "walk quickly"
        ["VBD", "RP"],  # e.g., "took off"
        ["MD", "VB"],  # e.g., "will go"
        ["VBZ", "ADVP"],  # e.g., "eats slowly" (ADVP = adverb phrase)
        ["VBG", "PP"],  # e.g., "running in the park" (PP = prepositional phrase)
    ],

    # Adjective Phrase (ADJP): Adverb + Adjective
    "ADJP": [
        ["JJ"],  # e.g., "happy"
        ["JJR"],  # e.g., "happier"
        ["JJS"],  # e.g., "happiest"
        ["RB", "JJ"],  # e.g., "very happy"
        ["RB", "JJR"],  # e.g., "much better"
    ],

    # Adverb Phrase (ADVP): Adverb(s)
    "ADVP": [
        ["RB"],  # e.g., "quickly"
        ["RBR"],  # e.g., "faster"
        ["RBS"],  # e.g., "fastest"
        ["RB", "RB"],  # e.g., "very quietly"
    ],

    # Prepositional Phrase (PP): Preposition + Noun Phrase
    "PP": [
        ["IN", "DT", "NN"],  # e.g., "in the house"
        ["IN", "NN"],  # e.g., "with friends"
        ["IN", "NP"],  # e.g., "under the table" (NP = ["DT", "NN"])
        ["TO", "VB"],  # e.g., "to run"
    ],

    # Interjection (INTJ)
    "INTJ": [
        ["UH"],  # e.g., "Wow", "Hello"
    ],

    # Coordinating Conjunction (CC)
    "CC": [
        ["CC"],  # e.g., "and", "but"
    ],

    # Quantifier Phrase (QP)
    "QP": [
        ["CD"],  # e.g., "five"
        ["CD", "NNS"],  # e.g., "five apples"
    ],
}

def _get_phrase_pos(phrase, grammar = grammar_config, mock =True):
    """
    Determine the POS of a phrase based on a grammar of allowed POS sequences.
    If the phrase is a single token, return its POS directly.

    Args:
        phrase (str): Input phrase to analyze.
        grammar (dict): A dictionary where keys are POS labels (e.g., 'NP', 'VP')
                        and values are lists of valid POS tag sequences.

    Returns:
        str: The POS label if a match is found; otherwise, 'UNKNOWN'.
    """

    if mock:
        return _mock_get_phrase_pos(phrase)


    def _clean_string(s):
        result = []
        depth = 0
        for char in s:
            if char == '(':
                depth += 1
            elif char == ')':
                if depth > 0:
                    depth -= 1
            else:
                if depth == 0:
                    result.append(char)
        return ''.join(result)

    phrase = _clean_string(phrase)

    # Get POS tags of the phrase using TextBlob
    blob = TextBlob(phrase)
    pos_sequence = [tag for word, tag in blob.tags]  # Extract POS tags as a list

    # # If the phrase is a single token, return its POS directly
    # if len(pos_sequence) == 1:
    #     return pos_sequence[0]

    # Check against the grammar rules for multi-token phrases
    for pos_label, patterns in grammar.items():
        if pos_sequence in patterns:
            return pos_label
    return "UNKNOWN"