"""
Document ingestion and RAG index building
"""

import numpy as np
from typing import List, Dict, Any
from pathlib import Path

from app.config import settings
from app.llm.openai_client import OpenAIClient
from app.rag.index import RAGIndex
from app.sources.local_files import (
    load_portfolio_json,
    load_resume_pdf,
    load_markdown_files
)
from app.sources.github import fetch_github_repos
from app.utils.text import chunk_text
from app.utils.logging import app_logger


class DocumentBuilder:
    """Builds and manages the RAG document index"""
    
    def __init__(self, openai_client: OpenAIClient, rag_index: RAGIndex):
        """
        Initialize document builder
        
        Args:
            openai_client: OpenAI client for embeddings
            rag_index: RAG index for storage
        """
        self.openai_client = openai_client
        self.rag_index = rag_index
    
    def build_index(
        self,
        sources: List[str] = None,
        force_rebuild: bool = False
    ) -> Dict[str, Any]:
        """
        Build the RAG index from all sources
        
        Args:
            sources: List of source types to include (default: all)
            force_rebuild: Force rebuild even if index exists
        
        Returns:
            Dictionary with build statistics
        """
        if sources is None:
            sources = ["portfolio", "resume", "markdown", "github"]
        
        # Check if index already exists
        if not force_rebuild and self.rag_index.loaded:
            stats = self.rag_index.get_stats()
            app_logger.info("Using existing RAG index")
            return {
                "status": "existing",
                "documents_processed": 0,
                "chunks_created": stats.get("total_chunks", 0),
                "embeddings_generated": stats.get("total_chunks", 0),
                "sources_ingested": sources
            }
        
        app_logger.info(f"Building RAG index from sources: {sources}")
        
        all_chunks = []
        errors = []
        docs_processed = 0
        
        # Load portfolio data
        if "portfolio" in sources:
            try:
                portfolio_chunks = self._process_portfolio()
                all_chunks.extend(portfolio_chunks)
                docs_processed += 1
                app_logger.info(f"Processed portfolio: {len(portfolio_chunks)} chunks")
            except Exception as e:
                error_msg = f"Portfolio processing error: {str(e)}"
                app_logger.error(error_msg)
                errors.append(error_msg)
        
        # Load resume
        if "resume" in sources:
            try:
                resume_chunks = self._process_resume()
                all_chunks.extend(resume_chunks)
                docs_processed += 1
                app_logger.info(f"Processed resume: {len(resume_chunks)} chunks")
            except Exception as e:
                error_msg = f"Resume processing error: {str(e)}"
                app_logger.error(error_msg)
                errors.append(error_msg)
        
        # Load markdown files
        if "markdown" in sources:
            try:
                md_chunks = self._process_markdown()
                all_chunks.extend(md_chunks)
                docs_processed += len(md_chunks) // 10 if md_chunks else 0  # Rough doc count
                app_logger.info(f"Processed markdown: {len(md_chunks)} chunks")
            except Exception as e:
                error_msg = f"Markdown processing error: {str(e)}"
                app_logger.error(error_msg)
                errors.append(error_msg)
        
        # Load GitHub repos
        if "github" in sources and settings.GITHUB_TOKEN:
            try:
                github_chunks = self._process_github()
                all_chunks.extend(github_chunks)
                docs_processed += len(github_chunks) // 5 if github_chunks else 0  # Rough doc count
                app_logger.info(f"Processed GitHub: {len(github_chunks)} chunks")
            except Exception as e:
                error_msg = f"GitHub processing error: {str(e)}"
                app_logger.error(error_msg)
                errors.append(error_msg)
        
        if not all_chunks:
            raise ValueError("No documents were successfully processed")
        
        # Generate embeddings
        app_logger.info(f"Generating embeddings for {len(all_chunks)} chunks...")
        texts = [chunk["text"] for chunk in all_chunks if chunk.get("text") and chunk["text"].strip()]
        
        if not texts:
            raise ValueError("No valid text chunks found for embedding generation")
        
        if len(texts) != len(all_chunks):
            app_logger.warning(f"Filtered out {len(all_chunks) - len(texts)} empty chunks")
        
        embeddings = self.openai_client.create_embeddings_batch(texts)
        
        # Convert to numpy array
        embeddings_array = np.array(embeddings)
        
        # Extract metadata
        metadata = [chunk["metadata"] for chunk in all_chunks]
        
        # Save to index
        self.rag_index.save_index(embeddings_array, metadata)
        
        return {
            "status": "success",
            "documents_processed": docs_processed,
            "chunks_created": len(all_chunks),
            "embeddings_generated": len(embeddings),
            "sources_ingested": sources,
            "errors": errors
        }
    
    def _process_portfolio(self) -> List[Dict[str, Any]]:
        """Process portfolio JSON file"""
        portfolio_data = load_portfolio_json(settings.PORTFOLIO_JSON_PATH)
        
        chunks = []
        
        # Process summary/about
        if "summary" in portfolio_data or "about" in portfolio_data:
            summary_text = portfolio_data.get("summary") or portfolio_data.get("about", "")
            if summary_text:
                chunks.extend(chunk_text(
                    summary_text,
                    chunk_size=settings.RAG_CHUNK_SIZE,
                    chunk_overlap=settings.RAG_CHUNK_OVERLAP,
                    metadata={
                        "source_type": "portfolio",
                        "source_name": "Professional Summary",
                        "locator": settings.PORTFOLIO_JSON_PATH
                    }
                ))
        
        # Process skills
        if "skills" in portfolio_data:
            skills_text = "Technical Skills:\n"
            skills = portfolio_data["skills"]
            if isinstance(skills, list):
                skills_text += ", ".join(skills)
            elif isinstance(skills, dict):
                for category, skill_list in skills.items():
                    skills_text += f"\n{category}: {', '.join(skill_list) if isinstance(skill_list, list) else skill_list}"
            
            chunks.extend(chunk_text(
                skills_text,
                chunk_size=settings.RAG_CHUNK_SIZE,
                chunk_overlap=settings.RAG_CHUNK_OVERLAP,
                metadata={
                    "source_type": "portfolio",
                    "source_name": "Technical Skills",
                    "locator": settings.PORTFOLIO_JSON_PATH
                }
            ))
        
        # Process projects
        if "projects" in portfolio_data:
            for project in portfolio_data["projects"]:
                project_text = f"Project: {project.get('name', 'Unnamed')}\n"
                if "description" in project:
                    project_text += f"{project['description']}\n"
                if "technologies" in project:
                    techs = project["technologies"]
                    tech_str = ", ".join(techs) if isinstance(techs, list) else str(techs)
                    project_text += f"Technologies: {tech_str}\n"
                if "achievements" in project:
                    project_text += f"Key achievements: {project['achievements']}"
                
                chunks.extend(chunk_text(
                    project_text,
                    chunk_size=settings.RAG_CHUNK_SIZE,
                    chunk_overlap=settings.RAG_CHUNK_OVERLAP,
                    metadata={
                        "source_type": "portfolio",
                        "source_name": f"Project: {project.get('name', 'Unnamed')}",
                        "locator": settings.PORTFOLIO_JSON_PATH
                    }
                ))
        
        # Process experience
        if "experience" in portfolio_data:
            for exp in portfolio_data["experience"]:
                exp_text = f"{exp.get('title', 'Position')} at {exp.get('company', 'Company')}\n"
                if "duration" in exp:
                    exp_text += f"Duration: {exp['duration']}\n"
                if "description" in exp:
                    exp_text += f"{exp['description']}\n"
                if "responsibilities" in exp:
                    resp = exp["responsibilities"]
                    if isinstance(resp, list):
                        exp_text += "Responsibilities:\n- " + "\n- ".join(resp)
                    else:
                        exp_text += f"Responsibilities: {resp}"
                
                chunks.extend(chunk_text(
                    exp_text,
                    chunk_size=settings.RAG_CHUNK_SIZE,
                    chunk_overlap=settings.RAG_CHUNK_OVERLAP,
                    metadata={
                        "source_type": "portfolio",
                        "source_name": f"Experience: {exp.get('company', 'Company')}",
                        "locator": settings.PORTFOLIO_JSON_PATH
                    }
                ))
        
        return chunks
    
    def _process_resume(self) -> List[Dict[str, Any]]:
        """Process resume PDF file"""
        resume_text = load_resume_pdf(settings.RESUME_PDF_PATH)
        
        return chunk_text(
            resume_text,
            chunk_size=settings.RAG_CHUNK_SIZE,
            chunk_overlap=settings.RAG_CHUNK_OVERLAP,
            metadata={
                "source_type": "resume",
                "source_name": "Resume PDF",
                "locator": settings.RESUME_PDF_PATH
            }
        )
    
    def _process_markdown(self) -> List[Dict[str, Any]]:
        """Process markdown files"""
        md_files = load_markdown_files(settings.PORTFOLIO_MD_GLOB)
        
        chunks = []
        for filepath, content in md_files.items():
            file_chunks = chunk_text(
                content,
                chunk_size=settings.RAG_CHUNK_SIZE,
                chunk_overlap=settings.RAG_CHUNK_OVERLAP,
                metadata={
                    "source_type": "markdown",
                    "source_name": Path(filepath).name,
                    "locator": filepath
                }
            )
            chunks.extend(file_chunks)
        
        return chunks
    
    def _process_github(self) -> List[Dict[str, Any]]:
        """Process GitHub repositories"""
        repos = fetch_github_repos(
            settings.GITHUB_USERNAME,
            settings.GITHUB_TOKEN,
            max_repos=settings.GITHUB_MAX_REPOS
        )
        
        chunks = []
        for repo in repos:
            repo_text = f"GitHub Repository: {repo['name']}\n"
            repo_text += f"Description: {repo.get('description', 'No description')}\n"
            
            if repo.get('languages'):
                repo_text += f"Languages: {', '.join(repo['languages'])}\n"
            
            if repo.get('topics'):
                repo_text += f"Topics: {', '.join(repo['topics'])}\n"
            
            if repo.get('readme'):
                repo_text += f"\nREADME:\n{repo['readme']}"
            
            repo_chunks = chunk_text(
                repo_text,
                chunk_size=settings.RAG_CHUNK_SIZE,
                chunk_overlap=settings.RAG_CHUNK_OVERLAP,
                metadata={
                    "source_type": "github",
                    "source_name": repo['name'],
                    "locator": repo.get('url', f"https://github.com/{settings.GITHUB_USERNAME}/{repo['name']}")
                }
            )
            chunks.extend(repo_chunks)
        
        return chunks
