"""
Advanced Memory Management System
Implements sliding window memory, conversation summarization, and context preservation
"""
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from collections import deque
import json


@dataclass
class ConversationTurn:
    """Single conversation turn (user message + assistant response)"""
    timestamp: datetime
    user_message: str
    assistant_response: str
    confidence: float = 0.0
    sources: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "user_message": self.user_message,
            "assistant_response": self.assistant_response,
            "confidence": self.confidence,
            "sources": self.sources,
            "metadata": self.metadata,
        }


class SlidingWindowMemory:
    """
    Maintains conversation context using sliding window
    Keeps recent conversations while efficiently managing memory
    """
    
    def __init__(self, window_size: int = 10, max_tokens: int = 3000):
        """
        Args:
            window_size: Number of recent turns to keep
            max_tokens: Approximate max tokens to keep in context
        """
        self.window_size = window_size
        self.max_tokens = max_tokens
        self.conversation_history: deque = deque(maxlen=window_size)
        self.summary_history: List[str] = []
        self.creation_time = datetime.now()
    
    def add_turn(self, 
                 user_message: str,
                 assistant_response: str,
                 confidence: float = 0.0,
                 sources: List[str] = None,
                 metadata: Dict = None) -> None:
        """Add a conversation turn to memory"""
        turn = ConversationTurn(
            timestamp=datetime.now(),
            user_message=user_message,
            assistant_response=assistant_response,
            confidence=confidence,
            sources=sources or [],
            metadata=metadata or {}
        )
        self.conversation_history.append(turn)
    
    def get_context(self, include_summaries: bool = True) -> str:
        """Get formatted context for LLM prompt"""
        context = ""
        
        # Add summaries if available
        if include_summaries and self.summary_history:
            context += "CONVERSATION SUMMARY:\n"
            context += "\n".join(self.summary_history[-3:])  # Last 3 summaries
            context += "\n\n"
        
        # Add recent conversation turns
        context += "RECENT CONVERSATION:\n"
        for turn in self.conversation_history:
            context += f"User: {turn.user_message}\n"
            context += f"Assistant: {turn.assistant_response}\n"
            if turn.sources:
                context += f"Sources: {', '.join(turn.sources)}\n"
            context += "\n"
        
        return context
    
    def get_last_n_turns(self, n: int = 5) -> List[ConversationTurn]:
        """Get last N conversation turns"""
        return list(self.conversation_history)[-n:]
    
    def estimate_tokens(self, text: str) -> int:
        """Rough estimate of tokens (1 token ≈ 4 chars)"""
        return len(text) // 4
    
    def get_context_tokens(self) -> int:
        """Estimate total tokens in current context"""
        context = self.get_context()
        return self.estimate_tokens(context)
    
    def is_near_limit(self, threshold: float = 0.8) -> bool:
        """Check if context is approaching token limit"""
        return self.get_context_tokens() > (self.max_tokens * threshold)
    
    def clear(self) -> None:
        """Clear all memory"""
        self.conversation_history.clear()
        self.summary_history.clear()
    
    def to_dict(self) -> Dict:
        """Convert memory to dictionary"""
        return {
            "window_size": self.window_size,
            "max_tokens": self.max_tokens,
            "turns": [turn.to_dict() for turn in self.conversation_history],
            "summaries": self.summary_history,
            "creation_time": self.creation_time.isoformat(),
        }


class ConversationSummarizer:
    """
    Summarizes conversations to preserve context while saving tokens
    Uses LLM for intelligent summarization
    """
    
    def __init__(self, openai_client=None, model: str = "gpt-3.5-turbo"):
        """
        Args:
            openai_client: OpenAI client instance
            model: Model to use for summarization
        """
        self.openai_client = openai_client
        self.model = model
    
    def summarize_turns(self, turns: List[ConversationTurn], max_length: int = 200) -> str:
        """
        Summarize a list of conversation turns
        
        Args:
            turns: List of conversation turns
            max_length: Maximum summary length
            
        Returns:
            Summary string
        """
        if not turns:
            return ""
        
        # Create conversation text
        conv_text = "\n".join([
            f"User: {turn.user_message}\nAssistant: {turn.assistant_response}"
            for turn in turns
        ])
        
        if not self.openai_client:
            # Fallback: simple summarization
            return self._simple_summarize(conv_text, max_length)
        
        try:
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "Summarize the following conversation in a concise way, preserving key information."
                    },
                    {
                        "role": "user",
                        "content": f"Summarize this conversation in {max_length} characters:\n\n{conv_text}"
                    }
                ],
                temperature=0.5,
                max_tokens=100
            )
            
            return response.choices[0].message.content
        
        except Exception as e:
            print(f"⚠️  Summarization error: {e}")
            return self._simple_summarize(conv_text, max_length)
    
    def _simple_summarize(self, text: str, max_length: int = 200) -> str:
        """Simple fallback summarization"""
        sentences = text.split('\n')
        summary = ""
        
        for sentence in sentences:
            if len(summary) + len(sentence) < max_length:
                summary += sentence + " "
            else:
                break
        
        return summary[:max_length]
    
    def extract_key_topics(self, turns: List[ConversationTurn]) -> List[str]:
        """Extract key topics from conversation"""
        topics = set()
        
        for turn in turns:
            # Simple keyword extraction
            words = turn.user_message.lower().split()
            # Filter out common words
            common_words = {'what', 'how', 'why', 'when', 'where', 'who', 'is', 'are', 'the', 'a', 'an', 'and', 'or', 'but'}
            for word in words:
                if len(word) > 3 and word not in common_words:
                    topics.add(word)
        
        return sorted(list(topics))


class ContextManager:
    """
    Manages overall conversation context
    Combines memory, summaries, and external context
    """
    
    def __init__(self, 
                 window_size: int = 10,
                 max_tokens: int = 3000,
                 openai_client=None):
        """Initialize context manager"""
        self.sliding_window = SlidingWindowMemory(window_size, max_tokens)
        self.summarizer = ConversationSummarizer(openai_client)
        self.external_context: Dict[str, str] = {}
    
    def add_conversation_turn(self,
                            user_message: str,
                            assistant_response: str,
                            confidence: float = 0.0,
                            sources: List[str] = None,
                            metadata: Dict = None) -> None:
        """Add turn and handle summarization if needed"""
        self.sliding_window.add_turn(
            user_message,
            assistant_response,
            confidence,
            sources,
            metadata
        )
        
        # Summarize if approaching limit
        if self.sliding_window.is_near_limit(threshold=0.8):
            self._trigger_summarization()
    
    def _trigger_summarization(self) -> None:
        """Trigger conversation summarization"""
        turns = self.sliding_window.get_last_n_turns(n=5)
        
        if len(turns) >= 3:
            summary = self.summarizer.summarize_turns(turns)
            self.sliding_window.summary_history.append(summary)
            print(f"📝 Conversation summarized: {summary[:100]}...")
    
    def set_external_context(self, context_type: str, content: str) -> None:
        """Set external context (portfolio, resume, GitHub)"""
        self.external_context[context_type] = content
    
    def get_full_context(self) -> str:
        """Get complete context for LLM"""
        context = ""
        
        # Add external context
        for context_type, content in self.external_context.items():
            context += f"\n{context_type.upper()} CONTEXT:\n{content[:500]}\n"
        
        # Add conversation context
        context += "\n" + self.sliding_window.get_context()
        
        return context
    
    def get_context_summary(self) -> Dict:
        """Get summary of context state"""
        return {
            "conversation_turns": len(self.sliding_window.conversation_history),
            "total_summaries": len(self.sliding_window.summary_history),
            "estimated_tokens": self.sliding_window.get_context_tokens(),
            "near_limit": self.sliding_window.is_near_limit(),
            "topics": self.summarizer.extract_key_topics(
                list(self.sliding_window.conversation_history)
            ),
        }
    
    def clear(self) -> None:
        """Clear all context"""
        self.sliding_window.clear()
        self.external_context.clear()
