import asyncio
from typing import List, Tuple, Dict, AsyncGenerator, Optional
from utils import ENTITY_MAP

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