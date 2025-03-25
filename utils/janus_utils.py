def apply_template_mistral_instruct(system_message, content):
    prompt = f"{system_message}\n{content}".strip()
    return f"[INST] {prompt} [/INST] "

def extract_after_inst(text):
    """
    Extract everything after the closing [/INST] tag.
    """
    marker = '[/INST]'
    if marker in text:
        return text.split(marker, 1)[1].strip()
    else:
        print("No [/INST] marker found.")
        return ""