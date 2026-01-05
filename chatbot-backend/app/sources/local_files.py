"""
Local file sources - portfolio JSON, resume PDF, markdown files
"""

import json
from pathlib import Path
from typing import Dict, Any
import pdfplumber
from app.utils.logging import app_logger


def load_portfolio_json(filepath: str) -> Dict[str, Any]:
    """
    Load portfolio data from JSON file
    
    Args:
        filepath: Path to portfolio JSON file
    
    Returns:
        Portfolio data dictionary
    """
    path = Path(filepath)
    
    if not path.exists():
        app_logger.warning(f"Portfolio JSON not found: {filepath}")
        return {}
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        app_logger.info(f"Loaded portfolio JSON from {filepath}")
        return data
    
    except Exception as e:
        app_logger.error(f"Error loading portfolio JSON: {str(e)}")
        return {}


def load_resume_pdf(filepath: str) -> str:
    """
    Extract text from resume PDF
    
    Args:
        filepath: Path to resume PDF file
    
    Returns:
        Extracted text content
    """
    path = Path(filepath)
    
    if not path.exists():
        app_logger.warning(f"Resume PDF not found: {filepath}")
        return ""
    
    try:
        text_content = []
        
        with pdfplumber.open(path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                text = page.extract_text()
                if text:
                    text_content.append(text)
                    app_logger.debug(f"Extracted page {page_num} from resume")
        
        full_text = "\n\n".join(text_content)
        app_logger.info(f"Loaded resume PDF from {filepath} ({len(full_text)} chars)")
        return full_text
    
    except Exception as e:
        app_logger.error(f"Error loading resume PDF: {str(e)}")
        return ""


def load_markdown_files(glob_pattern: str) -> Dict[str, str]:
    """
    Load all markdown files matching the glob pattern
    
    Args:
        glob_pattern: Glob pattern for markdown files (e.g., "./data/*.md")
    
    Returns:
        Dictionary mapping filepath to content
    """
    md_files = {}
    
    # Parse the glob pattern
    if "/" in glob_pattern:
        base_dir = "/".join(glob_pattern.split("/")[:-1])
        pattern = glob_pattern.split("/")[-1]
    else:
        base_dir = "."
        pattern = glob_pattern
    
    base_path = Path(base_dir)
    
    if not base_path.exists():
        app_logger.warning(f"Markdown directory not found: {base_dir}")
        return md_files
    
    try:
        for filepath in base_path.glob(pattern):
            if filepath.is_file() and filepath.suffix.lower() in ['.md', '.markdown']:
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    md_files[str(filepath)] = content
                    app_logger.info(f"Loaded markdown file: {filepath.name}")
                
                except Exception as e:
                    app_logger.error(f"Error reading {filepath}: {str(e)}")
        
        app_logger.info(f"Loaded {len(md_files)} markdown files")
        return md_files
    
    except Exception as e:
        app_logger.error(f"Error loading markdown files: {str(e)}")
        return md_files
