import polars as pl
from spacy.symbols import ADJ, NOUN, AUX, VERB
from typing import List, NamedTuple
import re


class CollocationPair(NamedTuple):
    pair_type: str = ''      # "adj_noun" or "subj_verb"
    word1: str = ''          # adjective or subject
    word2: str = ''          # noun or verb
    negation: str = ''       # any negation associated with the pair

    def __str__(self):
        parts = [self.negation, self.word1, self.word2] if self.negation else [self.word1, self.word2]
        return ' '.join(filter(None, parts))


def get_negation(token):
    """Extract negation associated with a token"""
    for child in token.children:
        if child.dep_ == 'neg':
            return child.text
    return ''


def clean_word(text, clean_text=True):
    """Clean and normalize a word"""
    if not text:
        return ''
    if clean_text:
        return re.sub("[^A-Za-z0-9 ]+", "", text).strip().lower()
    return text


def extract_adj_noun_pairs(doc, lemmatize=False, clean_text=True) -> List[CollocationPair]:
    """
    Extract adjective-noun pairs from a spaCy doc object

    Args:
        doc: spaCy doc object
        lemmatize: Whether to use lemmatized forms
        clean_text: Whether to clean text (remove special characters)

    Returns:
        List of CollocationPair named tuples
    """
    pairs = []

    for token in doc:
        if token.pos == ADJ and token.head.pos == NOUN:
            adjective = token
            noun = token.head

            # Get negation (check noun and its verb head)
            negation = get_negation(noun)
            if not negation and (noun.head.pos == AUX or noun.head.pos == VERB):
                negation = get_negation(noun.head)

            # Extract text or lemma
            if lemmatize:
                adj_text = adjective.lemma_
                noun_text = noun.lemma_
            else:
                adj_text = adjective.text
                noun_text = noun.text

            # Clean text
            adj_text = clean_word(adj_text, clean_text)
            noun_text = clean_word(noun_text, clean_text)
            negation = clean_word(negation, clean_text)

            # Only add if both words have content after cleaning
            if adj_text and noun_text:
                pair = CollocationPair(
                    pair_type="adj_noun",
                    word1=adj_text,
                    word2=noun_text,
                    negation=negation
                )
                pairs.append(pair)

    return pairs


def extract_subj_verb_pairs(doc, lemmatize=False, clean_text=True) -> List[CollocationPair]:
    """
    Extract subject-verb pairs from a spaCy doc object

    Args:
        doc: spaCy doc object
        lemmatize: Whether to use lemmatized forms
        clean_text: Whether to clean text (remove special characters)

    Returns:
        List of CollocationPair named tuples
    """
    pairs = []

    for token in doc:
        # Look for nominal subjects (nsubj) or passive subjects (nsubjpass)
        if token.dep_ in ('nsubj', 'nsubjpass'):
            subject = token
            verb = token.head

            # Make sure the head is actually a verb
            if verb.pos not in (VERB, AUX):
                continue

            # Get negation from the verb
            negation = get_negation(verb)

            # Extract text or lemma
            if lemmatize:
                subj_text = subject.lemma_
                verb_text = verb.lemma_
            else:
                subj_text = subject.text
                verb_text = verb.text

            # Clean text
            subj_text = clean_word(subj_text, clean_text)
            verb_text = clean_word(verb_text, clean_text)
            negation = clean_word(negation, clean_text)

            # Only add if both words have content after cleaning
            if subj_text and verb_text:
                pair = CollocationPair(
                    pair_type="subj_verb",
                    word1=subj_text,
                    word2=verb_text,
                    negation=negation
                )
                pairs.append(pair)

    return pairs


def extract_all_collocations(doc, lemmatize=False, clean_text=True) -> List[CollocationPair]:
    """
    Extract all collocation types from a spaCy doc object

    Args:
        doc: spaCy doc object
        lemmatize: Whether to use lemmatized forms
        clean_text: Whether to clean text (remove special characters)

    Returns:
        List of CollocationPair named tuples (both adj_noun and subj_verb)
    """
    pairs = []
    pairs.extend(extract_adj_noun_pairs(doc, lemmatize, clean_text))
    pairs.extend(extract_subj_verb_pairs(doc, lemmatize, clean_text))
    return pairs


def process_collocation_row(row, nlp_model, lemmatize=False, clean_text=True):
    """
    Process a single row to extract all collocation types

    Args:
        row: Dictionary containing text data
        nlp_model: spaCy model
        lemmatize: Whether to lemmatize
        clean_text: Whether to clean text

    Returns:
        Polars DataFrame with extracted pairs
    """
    text = str(row["text"])
    doc = nlp_model(text)

    pairs = extract_all_collocations(doc, lemmatize=lemmatize, clean_text=clean_text)

    if not pairs:
        # Return empty DataFrame with correct schema
        return pl.DataFrame(schema={
            "record_id": pl.Utf8,
            "col": pl.Utf8,
            "pair_type": pl.Utf8,
            "word1": pl.Utf8,
            "word2": pl.Utf8,
            "negation": pl.Utf8,
            "pair_text": pl.Utf8
        })

    # Create DataFrame from pairs
    df_data = []
    for pair in pairs:
        df_data.append({
            "record_id": row["record_id"],
            "col": row["col"],
            "pair_type": pair.pair_type,
            "word1": pair.word1,
            "word2": pair.word2,
            "negation": pair.negation,
            "pair_text": str(pair)
        })

    return pl.DataFrame(df_data)


# Keep old function name for backwards compatibility
def process_adj_noun_row(row, nlp_model, lemmatize=False, clean_text=True):
    """
    Backwards compatible function - now extracts all collocation types
    """
    return process_collocation_row(row, nlp_model, lemmatize, clean_text)
