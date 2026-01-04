"""
GitHub Data Ingestion Module
Fetches and caches GitHub repositories with smart rate limiting
"""
import os
import json
import requests
from typing import List, Dict, Optional
from datetime import datetime, timedelta


class GitHubIngestion:
    """
    GitHub data fetcher with:
    - Smart caching to avoid API rate limits
    - Comprehensive repository data extraction
    - Automatic refresh on stale data
    """
    
    def __init__(self, cache_dir: str = "./github_cache", cache_hours: int = 24):
        """
        Initialize GitHub ingestion
        
        Args:
            cache_dir: Directory to store cached data
            cache_hours: Hours before cache is considered stale
        """
        self.cache_dir = cache_dir
        self.cache_hours = cache_hours
        os.makedirs(cache_dir, exist_ok=True)
    
    def get_cache_file(self, username: str) -> str:
        """Get cache file path for username"""
        return os.path.join(self.cache_dir, f"{username}_repos.json")
    
    def is_cache_valid(self, username: str) -> bool:
        """Check if cache exists and is not stale"""
        cache_file = self.get_cache_file(username)
        
        if not os.path.exists(cache_file):
            return False
        
        try:
            with open(cache_file, 'r') as f:
                data = json.load(f)
            
            # Check timestamp
            cache_time = datetime.fromisoformat(data.get('timestamp', ''))
            age = datetime.now() - cache_time
            
            if age > timedelta(hours=self.cache_hours):
                print(f"📅 Cache is {age.total_seconds()/3600:.1f} hours old (stale)")
                return False
            
            print(f"✅ Using cached GitHub data ({age.total_seconds()/3600:.1f}h old)")
            return True
            
        except Exception as e:
            print(f"⚠️  Cache validation error: {e}")
            return False
    
    def load_from_cache(self, username: str) -> Optional[List[Dict]]:
        """Load repositories from cache"""
        try:
            cache_file = self.get_cache_file(username)
            with open(cache_file, 'r') as f:
                data = json.load(f)
            return data.get('repositories', [])
        except Exception as e:
            print(f"⚠️  Cache load error: {e}")
            return None
    
    def save_to_cache(self, username: str, repositories: List[Dict]):
        """Save repositories to cache"""
        try:
            cache_file = self.get_cache_file(username)
            data = {
                'username': username,
                'timestamp': datetime.now().isoformat(),
                'count': len(repositories),
                'repositories': repositories
            }
            with open(cache_file, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"💾 Cached {len(repositories)} repositories")
        except Exception as e:
            print(f"⚠️  Cache save error: {e}")
    
    def fetch_repositories(self, username: str, 
                          use_cache: bool = True,
                          max_repos: int = 100) -> List[Dict]:
        """
        Fetch all public repositories for a GitHub user
        
        Args:
            username: GitHub username
            use_cache: Use cached data if available
            max_repos: Maximum number of repos to fetch
            
        Returns:
            List of repository dictionaries
        """
        # Check cache first
        if use_cache and self.is_cache_valid(username):
            cached = self.load_from_cache(username)
            if cached is not None:
                return cached
        
        # Fetch from GitHub API
        print(f"🌐 Fetching repositories from GitHub API for {username}...")
        
        all_repos = []
        page = 1
        
        while len(all_repos) < max_repos:
            try:
                url = f"https://api.github.com/users/{username}/repos"
                params = {
                    'type': 'owner',
                    'sort': 'updated',
                    'per_page': 100,
                    'page': page
                }
                
                response = requests.get(url, params=params, timeout=10)
                
                if response.status_code != 200:
                    print(f"⚠️  GitHub API returned status {response.status_code}")
                    break
                
                repos = response.json()
                
                if not repos:  # No more repos
                    break
                
                # Filter out forks if desired
                public_repos = [r for r in repos if not r.get('fork', False)]
                all_repos.extend(public_repos)
                
                print(f"   Fetched page {page}: {len(public_repos)} repos")
                
                # Check if there are more pages
                if len(repos) < 100:
                    break
                
                page += 1
                
            except Exception as e:
                print(f"⚠️  Error fetching page {page}: {e}")
                break
        
        # Sort by stars and recent activity
        sorted_repos = sorted(
            all_repos,
            key=lambda x: (x.get('stargazers_count', 0), x.get('updated_at', '')),
            reverse=True
        )
        
        # Take top max_repos
        final_repos = sorted_repos[:max_repos]
        
        print(f"📊 Fetched {len(final_repos)} repositories total")
        
        # Save to cache
        if final_repos:
            self.save_to_cache(username, final_repos)
        
        return final_repos
    
    def extract_repository_info(self, repo: Dict) -> Dict:
        """Extract relevant information from repository data"""
        return {
            'name': repo.get('name', 'Unknown'),
            'description': repo.get('description', ''),
            'language': repo.get('language', 'Unknown'),
            'stars': repo.get('stargazers_count', 0),
            'forks': repo.get('forks_count', 0),
            'url': repo.get('html_url', ''),
            'topics': repo.get('topics', []),
            'created_at': repo.get('created_at', ''),
            'updated_at': repo.get('updated_at', ''),
            'size': repo.get('size', 0),
            'watchers': repo.get('watchers_count', 0),
            'open_issues': repo.get('open_issues_count', 0),
            'is_archived': repo.get('archived', False),
            'license': repo.get('license', {}).get('name', 'No License') if repo.get('license') else 'No License'
        }
    
    def create_document_chunks(self, repositories: List[Dict]) -> List[Dict]:
        """
        Create document chunks for RAG from GitHub repositories
        
        Returns:
            List of document dicts with content, type, and metadata
        """
        documents = []
        
        for repo in repositories:
            info = self.extract_repository_info(repo)
            
            # Skip archived repos
            if info['is_archived']:
                continue
            
            # Main repository chunk
            content = f"""GITHUB REPOSITORY: {info['name']}
Description: {info['description'] or 'No description provided'}
Programming Language: {info['language']}
Stars: {info['stars']} | Forks: {info['forks']} | Watchers: {info['watchers']}
Topics: {', '.join(info['topics']) if info['topics'] else 'None'}
Repository URL: {info['url']}
Last Updated: {info['updated_at'][:10]}
License: {info['license']}
"""
            
            documents.append({
                'type': 'github',
                'content': content,
                'metadata': {
                    'name': info['name'],
                    'url': info['url'],
                    'language': info['language'],
                    'stars': info['stars'],
                    'topics': info['topics']
                }
            })
            
            # Add language-specific chunk for better retrieval
            if info['language'] and info['language'] != 'Unknown':
                lang_content = f"""GitHub project in {info['language']}: {info['name']}
This {info['language']} project showcases expertise in {info['language']} development.
{info['description'] or ''}
URL: {info['url']}
"""
                documents.append({
                    'type': 'github_language',
                    'content': lang_content,
                    'metadata': {
                        'language': info['language'],
                        'project': info['name']
                    }
                })
        
        print(f"📚 Created {len(documents)} document chunks from {len(repositories)} repositories")
        return documents
    
    def clear_cache(self, username: Optional[str] = None):
        """Clear cache for specific user or all users"""
        if username:
            cache_file = self.get_cache_file(username)
            if os.path.exists(cache_file):
                os.remove(cache_file)
                print(f"🗑️  Cleared cache for {username}")
        else:
            # Clear all cache files
            for file in os.listdir(self.cache_dir):
                if file.endswith('_repos.json'):
                    os.remove(os.path.join(self.cache_dir, file))
            print("🗑️  Cleared all GitHub cache")


# Quick test
if __name__ == "__main__":
    github = GitHubIngestion(cache_hours=24)
    
    # Test with a username
    username = "avikshithreddy"
    repos = github.fetch_repositories(username, use_cache=True, max_repos=50)
    
    print(f"\n📊 Repository Summary:")
    print(f"   Total repos: {len(repos)}")
    
    # Language distribution
    languages = {}
    for repo in repos:
        lang = repo.get('language', 'Unknown')
        languages[lang] = languages.get(lang, 0) + 1
    
    print(f"   Languages: {dict(sorted(languages.items(), key=lambda x: x[1], reverse=True))}")
    
    # Create document chunks
    docs = github.create_document_chunks(repos)
    print(f"\n📚 Created {len(docs)} document chunks")
    
    # Show sample
    if docs:
        print(f"\n📄 Sample document chunk:")
        print(docs[0]['content'][:200] + "...")
