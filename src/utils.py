from typing import List, Dict

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

def convert_predictions(tokens: List[str], pipeline_output: List[Dict]) -> List[str]:
    """Convert pipeline output to token-level BIO tags."""
    predictions = ["O"] * len(tokens)
    
    offset = 0
    for entity in pipeline_output:
        entity_text = entity["word"]
        entity_type = entity["entity_group"]
        
        # Find the tokens that correspond to this entity
        entity_tokens = entity_text.split(" ")
        for i in range(offset, len(tokens) - len(entity_tokens) + 1):
            # print(f"Checking tokens: {tokens[i:i+len(entity_tokens)]} against entity tokens: {entity_tokens}")
            if tokens[i:i+len(entity_tokens)] == entity_tokens:
                # Mark the first token as B-TYPE
                predictions[i] = f"B-{entity_type}"
                
                # Mark subsequent tokens as I-TYPE
                for j in range(1, len(entity_tokens)):
                    predictions[i+j] = f"I-{entity_type}"
                
                offset = i + len(entity_tokens)  # Update offset to next token after the entity
                break
    
    return predictions

def convert_ids_to_bio(ids: List[int]) -> List[str]:
    """Convert a list of entity IDs to BIO tags."""
    return [ENTITY_MAP[id] for id in ids if id in ENTITY_MAP]