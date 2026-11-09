
import random
import time

import mock_llm

BASE_DELAY = 0.05
MAX_RETRIES = 3


def call_with_retry(ticket_id, messages):

    retry_count = 0

    for retry_number in range(MAX_RETRIES + 1):

        try:
            response = mock_llm.chat(messages)

            return response, retry_count

        except mock_llm.TransientLLMError:

            # No more retries after three retry attempts
            if retry_number >= MAX_RETRIES:
                raise

            # Exponential backoff with jitter
            wait = (
                BASE_DELAY
                * (2 ** retry_number)
                * random.uniform(0.9, 1.15)
            )

            print(
                f"RETRY ticket_id={ticket_id} "
                f"retry={retry_number} "
                f"wait={wait:.4f}s"
            )

            time.sleep(wait)

            retry_count += 1

    raise RuntimeError(
        f"Retry loop ended unexpectedly for {ticket_id}"
    )
