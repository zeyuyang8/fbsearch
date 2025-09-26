class PromptFormat:
    def __init__(self, before, after):
        self.before = before
        self.after = after

    def format(self, prompt):
        if isinstance(prompt, str):
            return self.before + prompt + self.after
        if isinstance(prompt, list):
            return [self.before + p + self.after for p in prompt]
        else:
            raise ValueError(f"Unsupported type {type(prompt)} for prompt")
