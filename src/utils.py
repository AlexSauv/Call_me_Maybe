def find_all_strings(user_prompt: str) -> list[str]:
    strings = []
    for quote in ["'", '"']:
        if quote in user_prompt:
            parts = user_prompt.split(quote)
            for i in range(1, len(parts), 2):
                clean_str = parts[i].strip()
                if clean_str:
                    print(f"LEN DE PARTS {len(parts)} clean_str {clean_str}")
                    strings.append(clean_str)
    return strings


def find_all_numbers(user_prompt: str) -> list[str]:
    num = []
    words = user_prompt.split()
    for word in words:
        new_digit = "".join(n for n in word if n.isdigit() or n == ".")
        if new_digit:
            value = float(new_digit) if "." in word else int(new_digit)
            num.append(value)
    return num
