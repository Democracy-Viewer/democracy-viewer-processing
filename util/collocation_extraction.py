import polars as pl
from spacy.symbols import ADJ, NOUN, AUX, VERB
from typing import List, NamedTuple, Optional
import re


class AdjNounPair(NamedTuple):
    verb_neg: str = ''
    neg_det: str = ''
    adjective: str = ''
    noun: str = ''

    def __str__(self):
        return ' '.join(filter(None, self))


def get_subject_neg(token):
    """Extract negation associated with the subject/noun"""
    for child in token.children:
        if child.dep_ == 'neg':
            return child
    return None


def get_verb_neg(verb_token):
    """Extract negation associated with the verb"""
    for child in verb_token.children:
        if child.dep_ == 'neg':
            return child, True
    return None, False


def extract_adj_noun_pairs(doc, lemmatize=False, clean_text=True) -> List[AdjNounPair]:
    """
    Extract adjective-noun pairs from a spaCy doc object
    
    Args:
        doc: spaCy doc object
        lemmatize: Whether to use lemmatized forms
        clean_text: Whether to clean text (remove special characters)
    
    Returns:
        List of AdjNounPair named tuples
    """
    pairs = []
    
    for adjective in doc:
        if adjective.pos == ADJ and adjective.head.pos == NOUN:
            noun = adjective.head

            # Get negation for the noun
            neg_det = get_subject_neg(noun)
            neg_det = neg_det.text if neg_det else ''

            # Get verb negation if noun is connected to a verb
            verb_neg = ''
            if noun.head.pos == AUX or noun.head.pos == VERB:
                verb_neg_token, _ = get_verb_neg(noun.head)
                verb_neg = verb_neg_token.text if verb_neg_token else ''

            # Extract text or lemma
            if lemmatize:
                adj_text = adjective.lemma_
                noun_text = noun.lemma_
            else:
                adj_text = adjective.text
                noun_text = noun.text

            # Clean text if requested
            if clean_text:
                adj_text = re.sub("[^A-Za-z0-9 ]+", "", adj_text).strip().lower()
                noun_text = re.sub("[^A-Za-z0-9 ]+", "", noun_text).strip().lower()
                verb_neg = re.sub("[^A-Za-z0-9 ]+", "", verb_neg).strip().lower() if verb_neg else ''
                neg_det = re.sub("[^A-Za-z0-9 ]+", "", neg_det).strip().lower() if neg_det else ''

            # Only add if both adjective and noun have content after cleaning
            if adj_text and noun_text:
                pair = AdjNounPair(
                    verb_neg=verb_neg,
                    neg_det=neg_det,
                    adjective=adj_text,
                    noun=noun_text
                )
                pairs.append(pair)
    
    return pairs


def process_adj_noun_row(row, nlp_model, lemmatize=False, clean_text=True):
    """
    Process a single row to extract adjective-noun pairs
    
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
    
    pairs = extract_adj_noun_pairs(doc, lemmatize=lemmatize, clean_text=clean_text)
    
    if not pairs:
        # Return empty DataFrame with correct schema
        return pl.DataFrame(schema={
            "record_id": pl.Utf8,
            "col": pl.Utf8,
            "verb_neg": pl.Utf8,
            "neg_det": pl.Utf8,
            "adjective": pl.Utf8,
            "noun": pl.Utf8,
            "pair_text": pl.Utf8
        })
    
    # Create DataFrame from pairs
    df_data = []
    for pair in pairs:
        df_data.append({
            "record_id": row["record_id"],
            "col": row["col"],
            "verb_neg": pair.verb_neg,
            "neg_det": pair.neg_det,
            "adjective": pair.adjective,
            "noun": pair.noun,
            "pair_text": str(pair)
        })
    
    return pl.DataFrame(df_data)