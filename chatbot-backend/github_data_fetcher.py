"""
GitHub Data Fetcher Module
Retrieves repositories, contributions, and profile data using GitHub API
"""
import requests
from typing import List, Dict, Optional
from datetime import datetime
import json


class GitHubDataFetcher:
    """Fetch and process GitHub user data"""
    
    def __init__(self, username: str, token: Optional[str] = None):
        """
        Initialize GitHub fetcher
        
        Args:
            username: GitHub username
            token: GitHub personal access token (for higher rate limits)
        """
        self.username = username
        self.token = token
        self.base_url = "https://api.github.com"
        self.headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "PortfolioChatbot"
        }
        
        if token:
            self.headers["Authorization"] = f"token {token}"
    
    def fetch_user_profile(self) -> Dict:
        """Fetch GitHub user profile information"""
        try:
            url = f"{self.base_url}/users/{self.username}"
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            return {
                "name": data.get("name", ""),
                "bio": data.get("bio", ""),
                "location": data.get("location", ""),
                "public_repos": data.get("public_repos", 0),
                "followers": data.get("followers", 0),
                "following": data.get("following", 0),
                "created_at": data.get("created_at", ""),
                "github_url": data.get("html_url", ""),
                "avatar_url": data.get("avatar_url", ""),
                "company": data.get("company", ""),
                "blog": data.get("blog", ""),
                "email": data.get("email", ""),
            }
        except Exception as e:
            print(f"❌ Error fetching user profile: {e}")
            return {}
    
    def fetch_repositories(self, sort: str = "stars", per_page: int = 100) -> List[Dict]:
        """
        Fetch repositories
        
        Args:
            sort: Sort by "stars", "updated", "pushed", or "forks"
            per_page: Number of repos to fetch (max 100)
            
        Returns:
            List of repository data
        """
        try:
            url = f"{self.base_url}/users/{self.username}/repos"
            params = {
                "sort": sort,
                "per_page": per_page,
                "type": "all"
            }
            
            response = requests.get(url, headers=self.headers, params=params, timeout=10)
            response.raise_for_status()
            
            repos = []
            for repo in response.json():
                repos.append({
                    "name": repo.get("name", ""),
                    "description": repo.get("description", ""),
                    "url": repo.get("html_url", ""),
                    "stars": repo.get("stargazers_count", 0),
                    "forks": repo.get("forks_count", 0),
                    "language": repo.get("language", ""),
                    "topics": repo.get("topics", []),
                    "updated_at": repo.get("updated_at", ""),
                    "created_at": repo.get("created_at", ""),
                    "size": repo.get("size", 0),
                    "is_fork": repo.get("fork", False),
                    "watchers": repo.get("watchers_count", 0),
                })
            
            return repos
        except Exception as e:
            print(f"❌ Error fetching repositories: {e}")
            return []
    
    def fetch_repository_details(self, repo_name: str) -> Dict:
        """Fetch detailed information about a specific repository"""
        try:
            url = f"{self.base_url}/repos/{self.username}/{repo_name}"
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            repo = response.json()
            return {
                "name": repo.get("name", ""),
                "description": repo.get("description", ""),
                "url": repo.get("html_url", ""),
                "stars": repo.get("stargazers_count", 0),
                "forks": repo.get("forks_count", 0),
                "language": repo.get("language", ""),
                "topics": repo.get("topics", []),
                "updated_at": repo.get("updated_at", ""),
                "created_at": repo.get("created_at", ""),
                "size": repo.get("size", 0),
                "is_fork": repo.get("fork", False),
                "watchers": repo.get("watchers_count", 0),
                "readme_url": f"https://raw.githubusercontent.com/{self.username}/{repo_name}/main/README.md",
            }
        except Exception as e:
            print(f"❌ Error fetching repository details: {e}")
            return {}
    
    def fetch_user_stats(self) -> Dict:
        """Fetch comprehensive user statistics"""
        try:
            profile = self.fetch_user_profile()
            repos = self.fetch_repositories()
            
            total_stars = sum(r["stars"] for r in repos)
            total_forks = sum(r["forks"] for r in repos)
            languages = {}
            
            for repo in repos:
                lang = repo.get("language")
                if lang:
                    languages[lang] = languages.get(lang, 0) + 1
            
            return {
                "profile": profile,
                "repo_count": len(repos),
                "total_stars": total_stars,
                "total_forks": total_forks,
                "languages": languages,
                "top_repos": sorted(repos, key=lambda r: r["stars"], reverse=True)[:5],
                "repos": repos,
            }
        except Exception as e:
            print(f"❌ Error fetching user stats: {e}")
            return {}
    
    def format_profile_for_rag(self) -> str:
        """Format GitHub profile data as a string for RAG embedding"""
        stats = self.fetch_user_stats()
        
        if not stats:
            return ""
        
        profile = stats.get("profile", {})
        repos = stats.get("repos", [])
        
        formatted_text = f"""
GitHub Profile Information:

Name: {profile.get('name', 'N/A')}
Bio: {profile.get('bio', 'N/A')}
Location: {profile.get('location', 'N/A')}
Company: {profile.get('company', 'N/A')}
Website: {profile.get('blog', 'N/A')}

Statistics:
- Public Repositories: {profile.get('public_repos', 0)}
- Total Followers: {profile.get('followers', 0)}
- Following: {profile.get('following', 0)}
- Total Stars Received: {stats.get('total_stars', 0)}
- Total Forks: {stats.get('total_forks', 0)}

Programming Languages:
{json.dumps(stats.get('languages', {}), indent=2)}

Recent Projects (with most stars):
"""
        
        for repo in repos[:10]:
            formatted_text += f"""
- {repo['name']} ({repo['stars']} stars)
  Language: {repo['language'] or 'Not specified'}
  Description: {repo['description'] or 'No description'}
  URL: {repo['url']}
  Topics: {', '.join(repo['topics']) if repo['topics'] else 'None'}
"""
        
        return formatted_text
