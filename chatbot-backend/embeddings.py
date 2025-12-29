"""
Advanced Embeddings & Retrieval Module
Implements semantic similarity search using embeddings
"""
import numpy as np
from typing import List, Dict, Tuple
from sklearn.metrics.pairwise import cosine_similarity
import openai


class EmbeddingGenerator:
    """Generate and cache embeddings for portfolio documents"""
    
    def __init__(self, model: str = "text-embedding-ada-002"):
        self.model = model
        self.cache = {}
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding with caching"""
        if text in self.cache:
            return self.cache[text]
        
        try:
            response = openai.Embedding.create(
                input=text,
                model=self.model
            )
            embedding = np.array(response['data'][0]['embedding'])
            self.cache[text] = embedding
            return embedding
        except Exception as e:
            print(f"⚠️  Embedding error: {e}")
            return np.random.rand(1536)


class SemanticRetrievalEngine:
    """RAG retrieval engine with semantic similarity"""
    
    def __init__(self, embedding_generator: EmbeddingGenerator):
        self.embedding_generator = embedding_generator
        self.knowledge_base = []
        self.embeddings_matrix = None
    
    def add_document(self, doc: Dict):
        """Add document to retrieval engine"""
        doc_copy = doc.copy()
        text = doc_copy.get('text', '')
        
        if text:
            doc_copy['embedding'] = self.embedding_generator.get_embedding(text)
            self.knowledge_base.append(doc_copy)
    
    def build_index(self):
        """Build embedding matrix for fast retrieval"""
        if self.knowledge_base:
            embeddings = [doc['embedding'] for doc in self.knowledge_base]
            self.embeddings_matrix = np.array(embeddings)
    
    def retrieve(self, query: str, top_k: int = 5, threshold: float = 0.3) -> List[Dict]:
        """
        Retrieve top-k most relevant documents using cosine similarity
        
        Args:
            query: User query string
            top_k: Number of results to return
            threshold: Minimum similarity score
        
        Returns:
            List of relevant documents sorted by similarity
        """
        query_embedding = self.embedding_generator.get_embedding(query)
        
        if self.embeddings_matrix is None:
            self.build_index()
        
        if len(self.knowledge_base) == 0:
            return []
        
        # Calculate similarities
        similarities = cosine_similarity([query_embedding], self.embeddings_matrix)[0]
        
        # Get top-k with threshold
        scored_docs = [
            (self.knowledge_base[i], float(similarities[i]))
            for i in range(len(self.knowledge_base))
            if similarities[i] >= threshold
        ]
        
        # Sort by similarity
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Return top-k with scores
        return [
            {**doc, 'similarity_score': score}
            for doc, score in scored_docs[:top_k]
        ]
    
    def retrieve_by_type(self, query: str, doc_type: str, top_k: int = 3) -> List[Dict]:
        """Retrieve documents filtered by type"""
        results = self.retrieve(query, top_k=top_k*2)  # Get extra to filter
        filtered = [doc for doc in results if doc.get('type') == doc_type]
        return filtered[:top_k]


class ContextWindowManager:
    """Manage conversation context with smart relevance ranking"""
    
    def __init__(self, max_tokens: int = 4000, reserved_for_prompt: int = 1000):
        self.max_tokens = max_tokens
        self.reserved_for_prompt = reserved_for_prompt
        self.available_tokens = max_tokens - reserved_for_prompt
        self.conversation_history = []
    
    def estimate_tokens(self, text: str) -> int:
        """Rough token estimation (1 token ≈ 4 chars)"""
        return len(text) // 4
    
    def add_turn(self, role: str, content: str):
        """Add conversation turn"""
        self.conversation_history.append({
            'role': role,
            'content': content,
            'tokens': self.estimate_tokens(content)
        })
    
    def get_context_window(self, retrieved_docs_tokens: int = 500) -> List[Dict]:
        """
        Get optimal context window respecting token limits
        
        Priority:
        1. Recent user intent (last 2 user messages)
        2. Last successful assistant response
        3. Earlier relevant context
        """
        available = self.available_tokens - retrieved_docs_tokens
        context = []
        current_tokens = 0
        
        # Add recent messages first (LIFO)
        for turn in reversed(self.conversation_history):
            turn_tokens = turn['tokens']
            if current_tokens + turn_tokens <= available:
                context.insert(0, turn)
                current_tokens += turn_tokens
            else:
                break
        
        return context
    
    def get_user_intent_context(self) -> str:
        """Extract current user intent from recent history"""
        user_msgs = [t['content'] for t in self.conversation_history if t['role'] == 'user']
        if user_msgs:
            return ' '.join(user_msgs[-3:])  # Last 3 user messages
        return ""


class ResponseGrounder:
    """Verify and ground responses in source documents"""
    
    def __init__(self, knowledge_base: List[Dict]):
        self.knowledge_base = knowledge_base
    
    def extract_citations(self, response: str, retrieved_docs: List[Dict]) -> Dict:
        """
        Extract mentioned entities and map to sources
        
        Returns:
            {
                'citations': [(text, source_url), ...],
                'mentioned_projects': [...],
                'confidence_score': 0.0-1.0
            }
        """
        citations = []
        mentioned_projects = []
        
        # Extract project references
        for doc in retrieved_docs:
            title = doc.get('title', '')
            if title and title.lower() in response.lower():
                mentioned_projects.append(title)
                if doc.get('url'):
                    citations.append((title, doc['url']))
        
        confidence = len(mentioned_projects) / max(len(retrieved_docs), 1)
        
        return {
            'citations': citations,
            'mentioned_projects': mentioned_projects,
            'confidence_score': min(confidence, 1.0),
            'grounded': len(mentioned_projects) > 0
        }
    
    def verify_factual_consistency(self, response: str, retrieved_docs: List[Dict]) -> bool:
        """
        Check if response mentions projects/skills from retrieved docs
        Helps verify grounding
        """
        mentioned_count = 0
        
        for doc in retrieved_docs:
            # Check if doc content appears in response
            if 'content' in doc and doc['content'].lower() in response.lower():
                mentioned_count += 1
            
            # Check if title is mentioned
            if 'title' in doc and doc['title'].lower() in response.lower():
                mentioned_count += 1
        
        return mentioned_count > 0


class AdaptivePromptBuilder:
    """Build prompts adapted to conversation context and retrieved docs"""
    
    @staticmethod
    def build_project_focused_prompt(retrieved_projects: List[Dict]) -> str:
        """Build prompt optimized for project-related queries"""
        base = """Focus on specific technical achievements:
- Mention concrete metrics and results (e.g., "94% accuracy", "40% improvement")
- Explain the technologies stack used
- Highlight your role in implementation
- Connect to related projects when relevant
"""
        if retrieved_projects:
            base += "\nRelevant projects to reference:\n"
            for proj in retrieved_projects:
                base += f"• {proj.get('title', 'N/A')}: {proj.get('content', '')}\n"
        
        return base
    
    @staticmethod
    def build_skill_focused_prompt(retrieved_skills: List[Dict]) -> str:
        """Build prompt optimized for skill/capability queries"""
        base = """Demonstrate skill depth:
- Provide concrete examples from portfolio
- Show progression and growth
- Connect skills to real project outcomes
- Mention depth (years, complexity, scale)
"""
        if retrieved_skills:
            base += "\nSkill areas to cover:\n"
            for skill in retrieved_skills:
                base += f"• {skill.get('category', 'N/A')}: {skill.get('content', '')}\n"
        
        return base
    
    @staticmethod
    def build_experience_focused_prompt(retrieved_exp: List[Dict]) -> str:
        """Build prompt optimized for experience queries"""
        base = """Frame experience progression:
- Highlight career growth and evolution
- Show increasing responsibility
- Connect to portfolio projects
- Emphasize learning and adaptation
"""
        if retrieved_exp:
            base += "\nRelevant experience:\n"
            for exp in retrieved_exp:
                base += f"• {exp.get('title', 'N/A')}: {exp.get('content', '')}\n"
        
        return base


# Example usage
if __name__ == "__main__":
    # Initialize components
    embedding_gen = EmbeddingGenerator()
    retrieval_engine = SemanticRetrievalEngine(embedding_gen)
    context_manager = ContextWindowManager()
    
    # Sample documents
    sample_docs = [
        {
            'type': 'project',
            'title': 'NLP Classification Model',
            'text': 'Built NLP transformer model achieving 94% accuracy',
            'content': '94% accuracy on multi-label classification using BERT'
        },
        {
            'type': 'skill',
            'category': 'ML',
            'text': 'Machine learning and deep learning expertise',
            'content': 'PyTorch, TensorFlow, Scikit-learn'
        }
    ]
    
    # Add to engine
    for doc in sample_docs:
        retrieval_engine.add_document(doc)
    
    retrieval_engine.build_index()
    
    # Test retrieval
    results = retrieval_engine.retrieve("What NLP work have you done?", top_k=2)
    print(f"Retrieved {len(results)} documents")
    for r in results:
        print(f"  - {r.get('title')}: {r.get('similarity_score'):.2f}")
