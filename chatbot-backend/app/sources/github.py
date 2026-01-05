"""
GitHub repository ingestion
"""

import requests
from typing import List, Dict, Any
from app.utils.logging import app_logger


def fetch_github_repos(
    username: str,
    token: str,
    max_repos: int = 20
) -> List[Dict[str, Any]]:
    """
    Fetch GitHub repositories for a user
    
    Args:
        username: GitHub username
        token: GitHub personal access token
        max_repos: Maximum number of repos to fetch
    
    Returns:
        List of repository data dictionaries
    """
    if not token:
        app_logger.warning("GitHub token not provided, skipping GitHub ingestion")
        return []
    
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    repos_data = []
    
    try:
        # Fetch repositories
        url = f"https://api.github.com/users/{username}/repos"
        params = {
            "sort": "updated",
            "direction": "desc",
            "per_page": max_repos
        }
        
        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        
        repos = response.json()
        app_logger.info(f"Fetched {len(repos)} repositories for {username}")
        
        for repo in repos[:max_repos]:
            repo_info = {
                "name": repo["name"],
                "description": repo.get("description", ""),
                "url": repo["html_url"],
                "language": repo.get("language"),
                "topics": repo.get("topics", []),
                "stars": repo.get("stargazers_count", 0),
                "forks": repo.get("forks_count", 0),
                "languages": [],
                "readme": ""
            }
            
            # Fetch languages
            if repo.get("languages_url"):
                try:
                    lang_response = requests.get(repo["languages_url"], headers=headers, timeout=5)
                    if lang_response.status_code == 200:
                        languages = lang_response.json()
                        repo_info["languages"] = list(languages.keys())
                except Exception as e:
                    app_logger.debug(f"Could not fetch languages for {repo['name']}: {str(e)}")
            
            # Fetch README
            readme_url = f"https://api.github.com/repos/{username}/{repo['name']}/readme"
            try:
                readme_response = requests.get(readme_url, headers=headers, timeout=5)
                if readme_response.status_code == 200:
                    readme_data = readme_response.json()
                    
                    # Fetch raw README content
                    if "download_url" in readme_data:
                        content_response = requests.get(readme_data["download_url"], timeout=5)
                        if content_response.status_code == 200:
                            repo_info["readme"] = content_response.text[:5000]  # Limit README size
                            app_logger.debug(f"Fetched README for {repo['name']}")
            
            except Exception as e:
                app_logger.debug(f"Could not fetch README for {repo['name']}: {str(e)}")
            
            repos_data.append(repo_info)
        
        app_logger.info(f"Successfully processed {len(repos_data)} GitHub repositories")
        return repos_data
    
    except requests.RequestException as e:
        app_logger.error(f"GitHub API error: {str(e)}")
        return []
    
    except Exception as e:
        app_logger.error(f"Error fetching GitHub repos: {str(e)}")
        return []
