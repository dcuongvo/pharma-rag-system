import time


def safe_llm_call(llm, prompt: str, max_retries: int = 5, base_delay: float = 10.0):
    """
    Retry LLM calls with exponential backoff (starting at 10s).
    """
    delay = base_delay
    last_error = None

    for attempt in range(max_retries):
        try:
            return llm.complete(prompt)

        except Exception as e:
            last_error = e
            print(f"LLM call failed (attempt {attempt + 1}/{max_retries}): {e}")

            if attempt == max_retries - 1:
                break

            print(f"Retrying in {delay:.1f} seconds...", flush=True)
            time.sleep(delay)

            delay *= 1.5  # slower growth than 2x (10 → 15 → 22 → 33...)

    raise last_error


def safe_indexing_llm_call(
    llm,
    prompt: str,
    max_retries: int = 5,
    base_delay: float = 10.0,
):
    """
    Indexing-time LLM retry helper.
    More patient retry strategy for offline / preprocessing tasks like:
    - doc type classification
    - continuation detection

    Minimum wait starts at 10 seconds.
    """
    delay = base_delay
    last_error = None

    for attempt in range(max_retries):
        try:
            return llm.complete(prompt)

        except Exception as e:
            last_error = e
            print(f"Indexing LLM call failed (attempt {attempt + 1}/{max_retries}): {e}")

            if attempt == max_retries - 1:
                break

            print(f"Retrying indexing call in {delay:.1f} seconds...", flush=True)
            time.sleep(delay)

            # keep it simple: fixed minimum 10s between retries
            delay = max(delay, 10.0)

    raise last_error