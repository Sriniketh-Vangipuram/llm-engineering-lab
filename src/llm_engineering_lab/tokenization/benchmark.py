from transformers import AutoTokenizer

def load_tokenizer(model_name:str):
    
    return AutoTokenizer.from_pretrained(model_name)

def benchmark_text(
    tokenizer,
    text:str,
    language:str,
)->dict:
    
    tokens=tokenizer.encode(
        text,
        add_special_tokens=False
    )
    
    characters=len(text)
    
    words=len(text.split())
    
    token_count=len(tokens)
    
    return {
        "language":language,
        "characters":characters,
        "words":words,
        "tokens":token_count,
        "tokens_per_character":(
            token_count/characters if characters else 0
        ),
        "tokens_per_word":(
            token_count/words if words else 0
        ),
        
    }