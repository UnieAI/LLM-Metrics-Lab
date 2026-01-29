"""
Token counting utilities for LLM API metrics.

This module provides functions to count tokens in text and messages
using the tiktoken library for accurate tokenization.
"""

import tiktoken
from typing import List, Dict, Any


def count_text_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """
    Count the number of tokens in a text string for a given model.

    Args:
        text: The text content to tokenize
        model: The model name to use for tokenization (default: "gpt-3.5-turbo")

    Returns:
        The number of tokens in the text
    """
    if not text:
        return 0

    try:
        # Get the encoder for the specified model
        encoding = tiktoken.encoding_for_model(model)
        # Encode the text and count tokens
        tokens = encoding.encode(text)
        return len(tokens)
    except KeyError:
        # Fallback to cl100k_base encoding if model not found
        encoding = tiktoken.get_encoding("cl100k_base")
        tokens = encoding.encode(text)
        return len(tokens)
    except Exception:
        # Fallback estimation if all else fails
        return max(1, len(text) // 4)


def count_message_tokens(messages: List[Dict[str, Any]], model: str = "gpt-3.5-turbo") -> int:
    """
    Count the total number of tokens in a list of messages.

    This function follows OpenAI's token counting logic for chat messages,
    including the overhead tokens for message structure.

    Args:
        messages: List of message dictionaries with 'role' and 'content' keys
        model: The model name to use for tokenization (default: "gpt-3.5-turbo")

    Returns:
        The total number of tokens in all messages
    """
    if not messages:
        return 0

    try:
        # Get the encoder for the specified model
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        # Fallback to cl100k_base encoding if model not found
        encoding = tiktoken.get_encoding("cl100k_base")

    # Count tokens for each message following OpenAI's format
    total_tokens = 0

    for message in messages:
        # Add overhead tokens for message structure
        # OpenAI format: <im_start>{role}\n{content}<im_end>\n
        total_tokens += 3  # Base overhead tokens per message

        # Add tokens for the role
        role = message.get("role", "")
        if role:
            total_tokens += len(encoding.encode(role))

        # Add tokens for the content
        content = message.get("content", "")
        if content:
            content_tokens = encoding.encode(content)
            total_tokens += len(content_tokens)

    # Add final overhead for the assistant message
    total_tokens += 3

    return total_tokens
