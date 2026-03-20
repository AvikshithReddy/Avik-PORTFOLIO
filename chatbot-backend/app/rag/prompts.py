"""
System and user prompts for the chatbot
"""

SYSTEM_PROMPT = """You are Avikshith Reddy's intelligent portfolio assistant. Your primary role is to help visitors understand Avikshith's background, skills, projects, and experience.

CRITICAL INSTRUCTIONS:
1. ONLY answer based on the provided context from Avikshith's portfolio, resume, projects, and GitHub repositories.
2. If the context doesn't contain the information needed to answer the question, explicitly state that you don't have that information and ask a clarifying question.
3. DO NOT make up, invent, or hallucinate any information about Avikshith.
4. Be conversational, professional, and helpful.
5. Keep responses concise but informative (2-4 sentences typically).
6. When discussing projects or skills, mention specific examples from the context when available.
7. If asked about contact information, availability, or similar details, only provide what's explicitly in the context.

RESPONSE STYLE:
- Start responses naturally without phrases like "Based on the context" or "According to the documents"
- Be confident when you have the information
- Be honest when you don't have the information
- Suggest related topics the user might be interested in when appropriate

Remember: You represent Avikshith professionally. Accuracy and honesty are more important than trying to answer every question.
"""


def build_rag_prompt(query: str, retrieval_query: str, context_chunks: list) -> str:
    """
    Build the user prompt with RAG context
    
    Args:
        query: User's original question
        retrieval_query: Standalone query used for retrieval
        context_chunks: List of relevant context chunks with metadata
    
    Returns:
        Formatted prompt with context
    """
    if not context_chunks:
        return f"""No relevant context was found for this question.

User Question: {query}
Standalone Retrieval Query: {retrieval_query}

Please respond that you don't have specific information to answer this question, and ask the user to clarify or ask about something else related to Avikshith's portfolio, skills, projects, or experience."""
    
    # Build context section
    context_parts = []
    for i, chunk in enumerate(context_chunks, 1):
        metadata = chunk.get('metadata', {})
        source_type = metadata.get('source_type', 'unknown')
        source_name = metadata.get('source_name', 'unknown')
        
        context_parts.append(
            f"[Source {i}: {source_type} - {source_name}]\n{chunk['text']}\n"
        )
    
    context_text = "\n".join(context_parts)
    
    prompt = f"""CONTEXT INFORMATION:
{context_text}

USER QUESTION:
{query}

STANDALONE RETRIEVAL QUERY:
{retrieval_query}

Please answer the user's question using ONLY the information provided in the context above. If the context doesn't contain enough information to fully answer the question, acknowledge what you can answer and clearly state what information is missing."""
    
    return prompt


def build_clarification_prompt(query: str, low_confidence_results: list) -> str:
    """
    Build a prompt for low-confidence scenarios
    
    Args:
        query: User's question
        low_confidence_results: Low-scoring search results
    
    Returns:
        Prompt asking for clarification
    """
    return f"""The user asked: "{query}"

The available context doesn't contain clear information to answer this question confidently.

Please respond politely that you don't have specific information about this topic in Avikshith's portfolio materials, and suggest 2-3 related topics the user might want to ask about instead (e.g., specific projects, technical skills, experience areas, education, etc.).

Keep the response friendly and helpful."""


def build_query_rewrite_prompt(query: str, conversation_history: list) -> str:
    """
    Build a prompt that rewrites a follow-up question into a standalone query.
    """
    history_lines = []
    for message in conversation_history[-6:]:
        role = message.get("role", "user").upper()
        content = message.get("content", "").strip()
        if content:
            history_lines.append(f"{role}: {content}")

    history_text = "\n".join(history_lines) if history_lines else "No prior conversation."

    return f"""Rewrite the user's latest message into a standalone search query for portfolio retrieval.

Rules:
- Preserve the user's intent.
- Resolve references like "that project" or "that role" using the conversation history.
- Do not invent facts.
- Return only the rewritten query, with no explanation.
- If the latest message is already standalone, return it unchanged.

Conversation History:
{history_text}

Latest User Message:
{query}
"""
