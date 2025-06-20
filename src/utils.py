from typing import List, Dict

def convert_predictions(tokens: List[str], pipeline_output: List[Dict]) -> List[str]:
    """Convert pipeline output to token-level BIO tags."""
    # replace any/multiple punctuation with empty string first using regex to avoid mismatches
    predictions = ["O"] * len(tokens)
    
    offset = 0
    for entity in pipeline_output:
        entity_text = entity["word"]
        entity_type = entity["entity_group"]
        
        # Find the tokens that correspond to this entity
        entity_tokens = entity_text.split(" ")
        for i in range(offset, offset + len(tokens) - len(entity_tokens) + 1):
            # print(f"Checking tokens: {tokens[i:i+len(entity_tokens)]} against entity tokens: {entity_tokens}")
            if tokens[i:i+len(entity_tokens)] == entity_tokens:
                # Mark the first token as B-TYPE
                predictions[i] = f"B-{entity_type}"
                offset = i + 1  # Move offset to next token
                
                # Mark subsequent tokens as I-TYPE
                for j in range(1, len(entity_tokens)):
                    predictions[i+j] = f"I-{entity_type}"
                    offset = i + j + 1  # Move offset to next token
                
                break
    
    return predictions

