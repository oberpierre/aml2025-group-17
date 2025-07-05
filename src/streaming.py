import asyncio
from typing import List, Tuple, Dict, AsyncGenerator, Optional

async def stream_sentence(sentence: Tuple[List[str], Optional[List[str]]], delay: float = 0.0) -> AsyncGenerator[Tuple[str, Optional[str]], None]:
    """
    Creates an async generator that yields (token, label) pairs one by one.
    
    Args:
        sentence: A tuple of (tokens, labels)
        delay: Optional delay between tokens
    
    Yields:
        (token, label) pairs one at a time
    """
    tokens, labels = sentence
    for i, token in enumerate(tokens):
        if delay > 0:
            await asyncio.sleep(delay)
        label = labels[i] if labels is not None else None
        yield (token, label)

ENTITY_MAP = {
    0: "O",
    1: "B-PERSON", 2: "I-PERSON",
    3: "B-NORP", 4: "I-NORP",
    5: "B-FAC", 6: "I-FAC",
    7: "B-ORG", 8: "I-ORG",
    9: "B-GPE", 10: "I-GPE",
    11: "B-LOC", 12: "I-LOC",
    13: "B-PRODUCT", 14: "I-PRODUCT",
    15: "B-DATE", 16: "I-DATE",
    17: "B-TIME", 18: "I-TIME",
    19: "B-PERCENT", 20: "I-PERCENT",
    21: "B-MONEY", 22: "I-MONEY",
    23: "B-QUANTITY", 24: "I-QUANTITY",
    25: "B-ORDINAL", 26: "I-ORDINAL",
    27: "B-CARDINAL", 28: "I-CARDINAL",
    29: "B-EVENT", 30: "I-EVENT",
    31: "B-WORK_OF_ART", 32: "I-WORK_OF_ART",
    33: "B-LAW", 34: "I-LAW",
    35: "B-LANGUAGE", 36: "I-LANGUAGE"
}

def process_ontonotes_example(example, include_labels=True):
    """
    Process an OntoNotes example to extract tokens and NER labels.
    
    Args:
        example: An example from the OntoNotes dataset
        include_labels: Whether to include ground truth labels
    
    Returns:
        List of (tokens, labels) pairs for each sentence in the document
    """
    result = []

    for sentence in example['sentences']:
        tokens = sentence['words']
        
        if include_labels and 'named_entities' in sentence:
            # Convert numeric labels to string format
            labels = [ENTITY_MAP.get(entity_id, "O") for entity_id in sentence['named_entities']]
            result.append((tokens, labels))
        else:
            result.append((tokens, None))
    
    return result