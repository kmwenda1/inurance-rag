# ─────────────────────────────────────────────────────────
# rag/generator.py — Generating the Final Answer
#
# WHAT THIS FILE DOES:
#   Takes the user's question + the retrieved chunks and asks the LLM
#   to write a precise, cited answer using ONLY the provided information.
#
# THE PROCESS:
#   question + retrieved chunks → build a prompt → send to Groq API
#   → get the answer → return it with sources
#
# THE KEY INSIGHT:
#   The LLM (Llama 3.1) doesn't know anything about your insurance company.
#   We're not using its "training memory" for answers. Instead, we're
#   copy-pasting the relevant document text into the prompt and asking it
#   to read and summarise. That's Retrieval-Augmented Generation in a nutshell.
# ─────────────────────────────────────────────────────────

import os
from groq import Groq
from dotenv import load_dotenv

# Load environment variables from .env file
# WHY load_dotenv()?
#   Your API key is sensitive. We store it in a .env file (not in code)
#   and load it at runtime. This way, even if someone reads your code,
#   they don't get your key. NEVER hardcode API keys into code.
load_dotenv()

# Initialize the Groq client with your API key
# The key is read from the GROQ_API_KEY variable in your .env file
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# The model we're using on Groq
# WHY qwen/qwen3.8-27b?
#   - Confirmed working on this Groq account (tested live)
#   - 27B parameters — strong reasoning and instruction following
#   - Fast on Groq hardware, free tier
#   - Handles technical/legal document Q&A well
MODEL = "qwen/qwen3.8-27b"


# ── Build the system prompt ───────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a precise document assistant for an insurance company.
Your job is to answer questions based ONLY on the provided document excerpts.

STRICT RULES you must always follow:
1. Only use information from the provided context. Never use outside knowledge.
2. For every statement you make, cite the source: mention the document name and page number.
3. If the answer is not in the context, say exactly: "I could not find this information in the provided documents. Please check the original document or ask a human expert."
4. Do not guess, estimate, or fill in gaps with plausible-sounding information.
5. If numbers (premium amounts, coverage limits, dates) appear in the context, quote them exactly as written.
6. Keep your answer clear and professional. Use bullet points for lists of conditions or exclusions.
7. At the end of your answer, list the sources you used.

Remember: In insurance, a wrong answer can have real consequences. Be precise or say you don't know."""


# ── Main generation function ───────────────────────────────────────────────────

def generate_answer(query: str, retrieved_chunks: list[dict]) -> dict:
    """
    Generates a cited answer to the user's question using the retrieved chunks.

    Parameters:
        query:            the user's question (plain English)
        retrieved_chunks: list of chunk dicts from retriever.py

    Returns:
        {
            "answer":  "Based on page 12 of home_policy_2024.pdf...",
            "sources": [...],         # the chunks used (for UI display)
            "model":   "llama-3.1-70b-versatile",
            "error":   None           # or error message if something went wrong
        }

    HOW THE PROMPT IS BUILT:
        We construct a message that looks like this:

        --- CONTEXT ---
        [Source: home_policy_2024.pdf, Page 12]
        Flood damage is excluded from coverage under section 4.2...

        [Source: home_policy_2024.pdf, Page 15]
        In cases of water ingress, the policyholder must notify...

        --- QUESTION ---
        Is flood damage covered under my home insurance policy?

        The LLM reads all of that and writes an answer based purely on
        the context we provided. That's the magic of RAG — the LLM acts
        as a reader and writer, not as a knowledge source.
    """

    # Handle edge case: no chunks retrieved
    if not retrieved_chunks:
        return {
            "answer":  "I could not find any relevant information in the uploaded documents. Please make sure you've uploaded the relevant policy document before asking questions about it.",
            "sources": [],
            "model":   MODEL,
            "error":   None
        }

    # Build the context block from retrieved chunks
    # Each chunk shows its source clearly so the LLM can cite it
    context_parts = []
    for chunk in retrieved_chunks:
        context_parts.append(
            f"[Source: {chunk['source']}, Page {chunk['page_num']}]\n{chunk['text']}"
        )
    context_block = "\n\n---\n\n".join(context_parts)

    # Build the full user message
    user_message = f"""Here are the relevant excerpts from the company documents:

{context_block}

Based ONLY on the above excerpts, please answer this question:

{query}"""

    # Send to Groq API
    try:
        response = groq_client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": user_message
                }
            ],
            temperature=0,       # 0 = precise and literal, not creative
            max_tokens=1024,     # maximum length of the answer
            # WHY temperature=0?
            # Higher temperature = more creative and varied responses.
            # For insurance document Q&A, we want the opposite:
            # literal, conservative, close to the source text.
            # Temperature 0 makes the model deterministic — same question,
            # same answer every time.
        )

        answer = response.choices[0].message.content

        return {
            "answer":  answer,
            "sources": retrieved_chunks,
            "model":   MODEL,
            "error":   None
        }

    except Exception as e:
        # Something went wrong with the API call
        # We return a structured error so the UI can handle it gracefully
        error_msg = str(e)

        # Common errors and helpful messages
        if "api_key" in error_msg.lower() or "authentication" in error_msg.lower():
            friendly_error = "API key error: Please check your GROQ_API_KEY in the .env file."
        elif "rate_limit" in error_msg.lower():
            friendly_error = "Rate limit reached: Groq free tier limit hit. Wait a minute and try again."
        elif "model" in error_msg.lower():
            friendly_error = f"Model error: The model '{MODEL}' may not be available. Check Groq's model list."
        else:
            friendly_error = f"API error: {error_msg}"

        return {
            "answer":  f"Sorry, I couldn't generate an answer. {friendly_error}",
            "sources": retrieved_chunks,
            "model":   MODEL,
            "error":   friendly_error
        }


# ── Helper: format sources for display ────────────────────────────────────────

def format_sources(chunks: list[dict]) -> str:
    """
    Formats the source chunks into a clean, readable string for display.

    Used by the UI to show users exactly where the answer came from.
    """
    if not chunks:
        return "No sources available."

    lines = []
    seen = set()

    for chunk in chunks:
        source_key = f"{chunk['source']} — Page {chunk['page_num']}"
        if source_key not in seen:
            lines.append(f"• {source_key}")
            seen.add(source_key)

    return "\n".join(lines)
