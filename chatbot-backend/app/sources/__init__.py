"""
Sources package for data ingestion
"""

from .local_files import load_portfolio_json, load_resume_pdf, load_markdown_files
from .github import fetch_github_repos

__all__ = ['load_portfolio_json', 'load_resume_pdf', 'load_markdown_files', 'fetch_github_repos']
