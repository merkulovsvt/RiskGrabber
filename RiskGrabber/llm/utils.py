def safe_prompt_str(s, default: str = "none") -> str:
    if s is None:
        return default
    t = str(s).strip()
    return t if t else default
