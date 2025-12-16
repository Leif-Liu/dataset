def format_prompt(template: str, prompt: str) -> str:
    return template.format(prompt=prompt)


def ensure_eos(text: str, eos_token: str | None) -> str:
    if not eos_token:
        return text
    if text.endswith(eos_token):
        return text
    return text + eos_token


