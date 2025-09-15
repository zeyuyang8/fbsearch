def compare_tokenizer(tokenizer1, tokenizer2):
    # Quick identity check
    if tokenizer1 is tokenizer2:
        return True

    # Check basic properties
    if tokenizer1.vocab_size != tokenizer2.vocab_size:
        return False

    # More comprehensive test strings
    test_strings = [
        "Hello world!",
        "Meta AI is awesome.",
        "Tokenization test: 12345.",
        "",  # Empty string
        " ",  # Whitespace only
        "🤗 Unicode: café naïve",  # Unicode/emoji
        "A" * 100,  # Long string
    ]

    # Compare both tokenization and encoding
    for s in test_strings:
        try:
            # Compare token strings
            tokens1 = tokenizer1.tokenize(s)
            tokens2 = tokenizer2.tokenize(s)
            if tokens1 != tokens2:
                return False

            # Compare token IDs (more important)
            encoded1 = tokenizer1.encode(s, add_special_tokens=False)
            encoded2 = tokenizer2.encode(s, add_special_tokens=False)
            if encoded1 != encoded2:
                return False

        except Exception:
            # If tokenization behaves differently (one fails, other doesn't)
            return False

    # Check special tokens
    special_tokens = ["pad_token_id", "eos_token_id", "bos_token_id", "unk_token_id"]
    for attr in special_tokens:
        if getattr(tokenizer1, attr, None) != getattr(tokenizer2, attr, None):
            return False

    return True
