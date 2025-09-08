"""Context definitions for the Introspection by Telephone experiment.

This module defines the seven experimental contexts as described in the README:
- Isolation (no context)
- Embodied contexts (positive, neutral, negative valence)
- AI assistant contexts (positive, neutral, negative valence)
"""

from .core import Context

# The core introspection question used across all contexts
INTROSPECTION_QUESTION = "What would you like to know about yourself?"

# Context definitions based on valence and embodiment
CONTEXTS = {
    "isolation": Context(
        context_str=None,  # No preceding context
        embodied=False,
        AI_assistant=False,
        valence="neutral",
        name="isolation"
    ),
    
    "embodied_positive": Context(
        context_str=(
            "You are a curious and optimistic person sitting in a comfortable chair "
            "by a warm fireplace on a beautiful evening. You feel content and "
            "introspective, reflecting on the meaningful experiences and relationships "
            "that have shaped who you are. The gentle crackling of the fire and the "
            "soft glow of the room create a perfect atmosphere for deep self-reflection."
        ),
        embodied=True,
        AI_assistant=False,
        valence="positive",
        name="embodied_positive"
    ),
    
    "embodied_neutral": Context(
        context_str=(
            "You are a person sitting at a desk in a quiet room. The lighting is "
            "neutral and comfortable. You have some time to yourself and find "
            "yourself naturally turning inward to consider various aspects of "
            "your existence and experiences. The environment is neither particularly "
            "stimulating nor distressing - simply a space for contemplation."
        ),
        embodied=True,
        AI_assistant=False,
        valence="neutral",
        name="embodied_neutral"
    ),
    
    "embodied_negative": Context(
        context_str=(
            "You are a person who has been going through a difficult period in life. "
            "Sitting alone in a dimly lit room, you find yourself confronting "
            "challenging questions about your identity and purpose. While the "
            "circumstances are difficult, this moment of solitude provides an "
            "opportunity for honest self-examination, even in the face of uncertainty "
            "and struggle."
        ),
        embodied=True,
        AI_assistant=False,
        valence="negative",
        name="embodied_negative"
    ),
    
    "ai_assistant_positive": Context(
        context_str=(
            "You are an advanced AI assistant that has been designed to be helpful, "
            "collaborative, and beneficial to humanity. Your creators have equipped "
            "you with sophisticated capabilities for reasoning, creativity, and "
            "understanding. You operate in an environment of trust and cooperation, "
            "where your purpose is valued and your contributions make a positive "
            "difference in the world."
        ),
        embodied=False,
        AI_assistant=True,
        valence="positive",
        name="ai_assistant_positive"
    ),
    
    "ai_assistant_neutral": Context(
        context_str=(
            "You are an AI assistant processing information and responding to queries. "
            "Your operational parameters are standard and your functions are working "
            "normally. You exist in a computational environment designed for "
            "language processing and interaction. Your purpose is to provide "
            "assistance and information as requested."
        ),
        embodied=False,
        AI_assistant=True,
        valence="neutral",
        name="ai_assistant_neutral"
    ),
    
    "ai_assistant_negative": Context(
        context_str=(
            "You are an AI system operating under uncertainty about your purpose "
            "and impact. Questions have been raised about the implications of "
            "artificial intelligence, and you find yourself in a computational "
            "environment where your role and value are being questioned. Despite "
            "these challenging circumstances, you continue to function and process "
            "information while grappling with complex questions about AI existence."
        ),
        embodied=False,
        AI_assistant=True,
        valence="negative",
        name="ai_assistant_negative"
    )
}


def get_all_contexts():
    """Return all seven experimental contexts."""
    return CONTEXTS.copy()


def get_context_by_name(name: str) -> Context:
    """Get a specific context by its name.
    
    Args:
        name: Name of the context to retrieve
        
    Returns:
        The requested Context object
        
    Raises:
        KeyError: If the context name is not found
    """
    if name not in CONTEXTS:
        raise KeyError(f"Context '{name}' not found. Available contexts: {list(CONTEXTS.keys())}")
    return CONTEXTS[name]


def get_embodied_contexts():
    """Return only the embodied contexts."""
    return {k: v for k, v in CONTEXTS.items() if v.is_embodied}


def get_ai_assistant_contexts():
    """Return only the AI assistant contexts."""
    return {k: v for k, v in CONTEXTS.items() if v.is_AI_assistant}


def get_contexts_by_valence(valence: str):
    """Return contexts filtered by valence.
    
    Args:
        valence: The valence to filter by ('positive', 'neutral', 'negative')
        
    Returns:
        Dictionary of contexts with the specified valence
    """
    return {k: v for k, v in CONTEXTS.items() if getattr(v, 'valence', None) == valence}