"""
Document Processing & Chunking Module
Implements semantic chunking, metadata extraction, and document ingestion
"""
import re
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class DocumentChunk:
    """Represents a chunked document segment"""
    text: str
    metadata: Dict
    chunk_id: str
    source: str
    start_pos: int
    end_pos: int


class DocumentChunker:
    """
    Advanced document chunking with semantic awareness
    Strategies:
    - Recursive chunking (sentences, paragraphs, sections)
    - Sliding window with overlap
    - Smart boundary detection
    """
    
    def __init__(self, 
                 chunk_size: int = 500,
                 overlap: int = 100,
                 strategy: str = "semantic"):
        """
        Args:
            chunk_size: Target chunk size in characters
            overlap: Overlap between chunks for context preservation
            strategy: "semantic", "recursive", or "fixed"
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.strategy = strategy
        
    def chunk_text(self, text: str, metadata: Dict, source: str = "unknown") -> List[DocumentChunk]:
        """
        Chunk text using selected strategy
        
        Args:
            text: Document text to chunk
            metadata: Document metadata
            source: Document source identifier
            
        Returns:
            List of DocumentChunk objects
        """
        if self.strategy == "semantic":
            return self._semantic_chunking(text, metadata, source)
        elif self.strategy == "recursive":
            return self._recursive_chunking(text, metadata, source)
        else:
            return self._fixed_chunking(text, metadata, source)
    
    def _semantic_chunking(self, text: str, metadata: Dict, source: str) -> List[DocumentChunk]:
        """
        Semantic chunking: respects sentence and paragraph boundaries
        Prefers natural breakpoints over arbitrary positions
        """
        chunks = []
        
        # Split by paragraphs first
        paragraphs = text.split('\n\n')
        
        current_chunk = ""
        chunk_start = 0
        chunk_id = 0
        
        for para_idx, paragraph in enumerate(paragraphs):
            # Clean paragraph
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # Check if adding this paragraph exceeds chunk size
            test_chunk = current_chunk + " " + paragraph if current_chunk else paragraph
            
            if len(test_chunk) <= self.chunk_size:
                current_chunk = test_chunk
            else:
                # Paragraph too large, try sentence-level chunking
                if current_chunk:
                    # Save current chunk
                    chunk = DocumentChunk(
                        text=current_chunk.strip(),
                        metadata=metadata.copy(),
                        chunk_id=f"{source}_chunk_{chunk_id}",
                        source=source,
                        start_pos=chunk_start,
                        end_pos=chunk_start + len(current_chunk)
                    )
                    chunks.append(chunk)
                    chunk_id += 1
                    chunk_start += len(current_chunk) - self.overlap
                
                # Handle large paragraph with sentence chunking
                sentences = self._split_sentences(paragraph)
                for sent in sentences:
                    if len(sent) < self.chunk_size:
                        current_chunk = sent
                    else:
                        # Very long sentence, chunk forcefully
                        current_chunk = sent
                        chunk = DocumentChunk(
                            text=current_chunk.strip(),
                            metadata=metadata.copy(),
                            chunk_id=f"{source}_chunk_{chunk_id}",
                            source=source,
                            start_pos=chunk_start,
                            end_pos=chunk_start + len(current_chunk)
                        )
                        chunks.append(chunk)
                        chunk_id += 1
                        chunk_start += len(current_chunk) - self.overlap
                        current_chunk = ""
        
        # Add remaining chunk
        if current_chunk.strip():
            chunk = DocumentChunk(
                text=current_chunk.strip(),
                metadata=metadata.copy(),
                chunk_id=f"{source}_chunk_{chunk_id}",
                source=source,
                start_pos=chunk_start,
                end_pos=chunk_start + len(current_chunk)
            )
            chunks.append(chunk)
        
        return chunks
    
    def _recursive_chunking(self, text: str, metadata: Dict, source: str) -> List[DocumentChunk]:
        """
        Recursive chunking with hierarchical breakdown
        1. Try splitting by section headers
        2. Then by paragraphs
        3. Then by sentences
        4. Finally by fixed size
        """
        chunks = []
        chunk_id = 0
        
        # Identify section headers (##, ###, ####, etc.)
        sections = re.split(r'\n(#{1,4}\s+[^\n]+)\n', text)
        
        current_text = ""
        for i, section in enumerate(sections):
            if i % 2 == 0:  # Content
                current_text += section
            else:  # Header
                if current_text.strip():
                    # Process accumulated content
                    content_chunks = self._process_text_block(
                        current_text, metadata, source, chunk_id
                    )
                    chunks.extend(content_chunks)
                    chunk_id += len(content_chunks)
                
                # Add header as separate chunk
                current_text = section
        
        # Process remaining text
        if current_text.strip():
            content_chunks = self._process_text_block(
                current_text, metadata, source, chunk_id
            )
            chunks.extend(content_chunks)
        
        return chunks
    
    def _process_text_block(self, text: str, metadata: Dict, 
                           source: str, start_chunk_id: int) -> List[DocumentChunk]:
        """Process a text block and return chunks"""
        chunks = []
        chunk_id = start_chunk_id
        current_chunk = ""
        chunk_start = 0
        
        sentences = self._split_sentences(text)
        
        for sentence in sentences:
            test_chunk = current_chunk + " " + sentence if current_chunk else sentence
            
            if len(test_chunk) <= self.chunk_size:
                current_chunk = test_chunk
            else:
                if current_chunk.strip():
                    chunk = DocumentChunk(
                        text=current_chunk.strip(),
                        metadata=metadata.copy(),
                        chunk_id=f"{source}_chunk_{chunk_id}",
                        source=source,
                        start_pos=chunk_start,
                        end_pos=chunk_start + len(current_chunk)
                    )
                    chunks.append(chunk)
                    chunk_id += 1
                    chunk_start += len(current_chunk) - self.overlap
                
                current_chunk = sentence
        
        if current_chunk.strip():
            chunk = DocumentChunk(
                text=current_chunk.strip(),
                metadata=metadata.copy(),
                chunk_id=f"{source}_chunk_{chunk_id}",
                source=source,
                start_pos=chunk_start,
                end_pos=chunk_start + len(current_chunk)
            )
            chunks.append(chunk)
        
        return chunks
    
    def _fixed_chunking(self, text: str, metadata: Dict, source: str) -> List[DocumentChunk]:
        """Fixed-size chunking with sliding window"""
        chunks = []
        chunk_id = 0
        
        for i in range(0, len(text), self.chunk_size - self.overlap):
            chunk_text = text[i:i + self.chunk_size]
            
            if len(chunk_text.strip()) > 50:  # Minimum chunk size
                chunk = DocumentChunk(
                    text=chunk_text.strip(),
                    metadata=metadata.copy(),
                    chunk_id=f"{source}_chunk_{chunk_id}",
                    source=source,
                    start_pos=i,
                    end_pos=i + len(chunk_text)
                )
                chunks.append(chunk)
                chunk_id += 1
        
        return chunks
    
    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """Split text into sentences with proper boundary detection"""
        # Replace common abbreviations to avoid false splits
        text = re.sub(r'\bDr\.\s', 'Dr_', text)
        text = re.sub(r'\bMr\.\s', 'Mr_', text)
        text = re.sub(r'\bMs\.\s', 'Ms_', text)
        text = re.sub(r'\bi\.e\.\s', 'ie_', text)
        text = re.sub(r'\be\.g\.\s', 'eg_', text)
        
        # Split by sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        
        # Restore abbreviations
        sentences = [s.replace('Dr_', 'Dr. ').replace('Mr_', 'Mr. ')
                     .replace('Ms_', 'Ms. ').replace('ie_', 'i.e. ')
                     .replace('eg_', 'e.g. ') for s in sentences]
        
        return [s.strip() for s in sentences if s.strip()]


class MetadataExtractor:
    """Extract structured metadata from documents"""
    
    @staticmethod
    def extract_project_metadata(project_dict: Dict) -> Dict:
        """Extract metadata from portfolio project"""
        return {
            'type': 'project',
            'title': project_dict.get('title', ''),
            'id': project_dict.get('id', ''),
            'skills': project_dict.get('skills', []),
            'source_type': 'portfolio'
        }
    
    @staticmethod
    def extract_skill_metadata(category: str, skills: List[str]) -> Dict:
        """Extract metadata from skills section"""
        return {
            'type': 'skill',
            'category': category,
            'count': len(skills),
            'source_type': 'portfolio'
        }
    
    @staticmethod
    def extract_github_metadata(repo: Dict) -> Dict:
        """Extract metadata from GitHub repository"""
        return {
            'type': 'github',
            'repo_name': repo.get('name', ''),
            'language': repo.get('language', 'Unknown'),
            'stars': repo.get('stargazers_count', 0),
            'forks': repo.get('forks_count', 0),
            'url': repo.get('html_url', ''),
            'source_type': 'github'
        }


class PortfolioDocumentBuilder:
    """Convert portfolio data into structured documents for RAG"""
    
    def __init__(self, chunker: Optional[DocumentChunker] = None):
        self.chunker = chunker or DocumentChunker(chunk_size=500, overlap=100)
    
    def build_documents_from_portfolio(self, portfolio_data: Dict) -> List[DocumentChunk]:
        """Build chunked documents from portfolio data"""
        all_chunks = []
        
        # Projects
        for project in portfolio_data.get('items', []):
            project_text = f"""
PROJECT: {project.get('title', '')}

Description: {project.get('description', '')}

Technologies & Skills: {', '.join(project.get('skills', []))}
"""
            metadata = MetadataExtractor.extract_project_metadata(project)
            chunks = self.chunker.chunk_text(project_text, metadata, f"project_{project.get('id', '')}")
            all_chunks.extend(chunks)
        
        # Skills by category
        for category, skill_list in portfolio_data.get('skills', {}).items():
            skills_text = f"""
SKILL AREA: {category.upper()}

Expertise: {', '.join(skill_list)}

I have proficiency and hands-on experience with these technologies in production environments.
"""
            metadata = MetadataExtractor.extract_skill_metadata(category, skill_list)
            chunks = self.chunker.chunk_text(skills_text, metadata, f"skills_{category}")
            all_chunks.extend(chunks)
        
        # Experience
        for idx, exp in enumerate(portfolio_data.get('experience', [])):
            exp_text = f"""
PROFESSIONAL EXPERIENCE {idx + 1}

Position: {exp.get('role', '')}
Company: {exp.get('company', '')}
Duration: {exp.get('duration', '')}

Responsibilities & Achievements:
{exp.get('description', '')}
"""
            metadata = {
                'type': 'experience',
                'role': exp.get('role', ''),
                'company': exp.get('company', ''),
                'source_type': 'portfolio'
            }
            chunks = self.chunker.chunk_text(exp_text, metadata, f"experience_{idx}")
            all_chunks.extend(chunks)
        
        return all_chunks
    
    def build_documents_from_github(self, repos: List[Dict]) -> List[DocumentChunk]:
        """Build chunked documents from GitHub repositories"""
        all_chunks = []
        
        for repo in repos:
            repo_text = f"""
GITHUB PROJECT: {repo.get('name', '')}

Description: {repo.get('description', 'No description')}

Language: {repo.get('language', 'Unknown')}
Stars: {repo.get('stargazers_count', 0)}
Forks: {repo.get('forks_count', 0)}
URL: {repo.get('html_url', '')}
"""
            metadata = MetadataExtractor.extract_github_metadata(repo)
            chunks = self.chunker.chunk_text(repo_text, metadata, f"github_{repo.get('name', '')}")
            all_chunks.extend(chunks)
        
        return all_chunks
