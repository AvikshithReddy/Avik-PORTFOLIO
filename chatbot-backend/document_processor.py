"""
Document Processor for Portfolio Data
Extracts and chunks portfolio, resume, and experience data
"""
from typing import List, Dict, Optional


class DocumentProcessor:
    """
    Process portfolio data into optimized document chunks for RAG
    """
    
    @staticmethod
    def extract_experience_chunks(portfolio_data: Dict) -> List[Dict]:
        """Extract experience data into document chunks"""
        documents = []
        
        if not portfolio_data or "experience" not in portfolio_data:
            return documents
        
        for exp in portfolio_data["experience"]:
            title = exp.get("title", "")
            company = exp.get("company", "")
            duration = exp.get("duration", "")
            description = exp.get("description", "")
            skills = exp.get("skills", [])
            location = exp.get("location", "")
            
            # Main experience chunk
            is_current = "Present" in duration or "Current" in duration
            
            content = f"""{"CURRENT " if is_current else ""}PROFESSIONAL EXPERIENCE: {title} at {company}
Duration: {duration}
Location: {location}
Role Description: {description}
Key Skills & Technologies: {', '.join(skills)}
"""
            
            documents.append({
                "type": "experience",
                "content": content,
                "metadata": {
                    "title": title,
                    "company": company,
                    "duration": duration,
                    "current": is_current,
                    "skills": skills
                }
            })
            
            # Separate chunk for skills in this role
            if skills:
                skills_content = f"""Skills and technologies from {company} ({title} role):
{', '.join(skills)}

These skills were applied in: {description[:200]}...
"""
                documents.append({
                    "type": "experience_skills",
                    "content": skills_content,
                    "metadata": {
                        "company": company,
                        "role": title
                    }
                })
        
        return documents
    
    @staticmethod
    def extract_project_chunks(portfolio_data: Dict) -> List[Dict]:
        """Extract project data into document chunks"""
        documents = []
        
        if not portfolio_data or "projects" not in portfolio_data:
            return documents
        
        for project in portfolio_data["projects"]:
            name = project.get("name", "")
            description = project.get("description", "")
            technologies = project.get("technologies", [])
            results = project.get("results", "")
            
            # Main project chunk
            content = f"""PROJECT: {name}
Description: {description}
Technologies & Tools: {', '.join(technologies)}
Results & Impact: {results}
"""
            
            documents.append({
                "type": "project",
                "content": content,
                "metadata": {
                    "name": name,
                    "technologies": technologies,
                    "results": results
                }
            })
            
            # Technology-focused chunk
            if technologies:
                tech_content = f"""Technical stack for project '{name}':
Technologies used: {', '.join(technologies)}

This project demonstrates proficiency in: {', '.join(technologies[:4])}
Project outcome: {results}
"""
                documents.append({
                    "type": "project_tech",
                    "content": tech_content,
                    "metadata": {
                        "project": name,
                        "tech_stack": technologies
                    }
                })
        
        return documents
    
    @staticmethod
    def extract_skills_chunks(portfolio_data: Dict) -> List[Dict]:
        """Extract skills data into document chunks"""
        documents = []
        
        if not portfolio_data or "skills" not in portfolio_data:
            return documents
        
        skills_obj = portfolio_data["skills"]
        
        # Category-wise chunks
        for category, items in skills_obj.items():
            if items:
                content = f"""TECHNICAL SKILLS - {category.upper()}:
{', '.join(items)}

Expert proficiency in {category} domain with the following technologies:
{', '.join(items)}
"""
                documents.append({
                    "type": "skills",
                    "content": content,
                    "metadata": {
                        "category": category,
                        "skills": items,
                        "count": len(items)
                    }
                })
        
        # Comprehensive skills summary
        all_skills = []
        for category, items in skills_obj.items():
            all_skills.extend(items)
        
        if all_skills:
            unique_skills = list(set(all_skills))
            skills_summary = f"""COMPLETE TECHNICAL SKILL SET:
{', '.join(unique_skills)}

Broad expertise spanning {len(skills_obj)} domains with {len(unique_skills)} unique technologies.
Primary areas: {', '.join(skills_obj.keys())}
"""
            documents.append({
                "type": "all_skills",
                "content": skills_summary,
                "metadata": {
                    "total_skills": len(unique_skills),
                    "domains": list(skills_obj.keys())
                }
            })
        
        return documents
    
    @staticmethod
    def extract_education_chunks(portfolio_data: Dict) -> List[Dict]:
        """Extract education data into document chunks"""
        documents = []
        
        if not portfolio_data or "education" not in portfolio_data:
            return documents
        
        for edu in portfolio_data["education"]:
            degree = edu.get("degree", "")
            school = edu.get("school", "")
            duration = edu.get("duration", "")
            location = edu.get("location", "")
            gpa = edu.get("gpa", "")
            
            content = f"""EDUCATION: {degree}
Institution: {school}
Duration: {duration}
{f"Location: {location}" if location else ""}
{f"GPA: {gpa}" if gpa else ""}
"""
            
            documents.append({
                "type": "education",
                "content": content.strip(),
                "metadata": {
                    "degree": degree,
                    "school": school,
                    "duration": duration
                }
            })
        
        return documents
    
    @staticmethod
    def extract_profile_chunk(portfolio_data: Dict) -> List[Dict]:
        """Extract profile/bio data into document chunk"""
        if not portfolio_data:
            return []
        
        name = portfolio_data.get("name", "")
        title = portfolio_data.get("title", "")
        bio = portfolio_data.get("bio", "")
        github = portfolio_data.get("github", "")
        
        content = f"""PROFESSIONAL PROFILE: {name}
Title: {title}
Bio: {bio}
{f"GitHub: https://github.com/{github}" if github else ""}

Summary: {name} is a {title} specializing in data science, machine learning, and AI solutions.
"""
        
        return [{
            "type": "profile",
            "content": content.strip(),
            "metadata": {
                "name": name,
                "title": title,
                "github": github
            }
        }]
    
    @staticmethod
    def process_portfolio(portfolio_data: Dict) -> List[Dict]:
        """
        Process complete portfolio data into document chunks
        
        Returns:
            List of document dictionaries ready for embedding
        """
        all_documents = []
        
        # Extract all document types
        all_documents.extend(DocumentProcessor.extract_profile_chunk(portfolio_data))
        all_documents.extend(DocumentProcessor.extract_experience_chunks(portfolio_data))
        all_documents.extend(DocumentProcessor.extract_project_chunks(portfolio_data))
        all_documents.extend(DocumentProcessor.extract_skills_chunks(portfolio_data))
        all_documents.extend(DocumentProcessor.extract_education_chunks(portfolio_data))
        
        print(f"📄 Processed portfolio into {len(all_documents)} document chunks")
        return all_documents


# Quick test
if __name__ == "__main__":
    import json
    import os
    
    # Try to load portfolio data
    portfolio_file = "../portfolio_data.json"
    
    if os.path.exists(portfolio_file):
        with open(portfolio_file, 'r') as f:
            data = json.load(f)
        
        docs = DocumentProcessor.process_portfolio(data)
        
        print(f"\n📊 Document Statistics:")
        doc_types = {}
        for doc in docs:
            doc_type = doc['type']
            doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
        
        for doc_type, count in sorted(doc_types.items()):
            print(f"   {doc_type}: {count}")
        
        # Show sample
        if docs:
            print(f"\n📄 Sample document:")
            print(docs[0]['content'][:300] + "...")
    else:
        print(f"⚠️  Portfolio file not found: {portfolio_file}")
