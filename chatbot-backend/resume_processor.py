"""
PDF Resume Processor Module
Extracts and processes resume content from PDF files
"""
import os
from typing import List, Dict, Optional
import re


class ResumeProcessor:
    """Process and extract resume information from PDF"""
    
    def __init__(self, pdf_path: str):
        """
        Initialize resume processor
        
        Args:
            pdf_path: Path to PDF resume file
        """
        self.pdf_path = pdf_path
        self.raw_text = None
        self.structured_data = None
        self.PyPDF2 = None
        self.pdfplumber = None
        self.has_pdf_support = False
        self.has_pdfplumber = False
        
        # Try to import PyPDF2 for PDF processing
        try:
            import PyPDF2
            self.PyPDF2 = PyPDF2
            self.has_pdf_support = True
        except ImportError:
            pass
        
        # Try to import pdfplumber for better extraction
        try:
            import pdfplumber
            self.pdfplumber = pdfplumber
            self.has_pdfplumber = True
        except ImportError:
            pass
    
    def extract_text(self) -> str:
        """Extract text from PDF using available libraries"""
        if not os.path.exists(self.pdf_path):
            print(f"❌ PDF not found: {self.pdf_path}")
            return ""
        
        # Try pdfplumber first (better quality)
        if self.has_pdfplumber:
            try:
                with self.pdfplumber.open(self.pdf_path) as pdf:
                    text = ""
                    for page in pdf.pages:
                        text += page.extract_text() + "\n"
                    self.raw_text = text
                    print(f"✅ Resume extracted using pdfplumber ({len(text)} chars)")
                    return text
            except Exception as e:
                print(f"⚠️  pdfplumber extraction failed: {e}")
        
        # Fall back to PyPDF2
        if self.has_pdf_support:
            try:
                with open(self.pdf_path, 'rb') as file:
                    reader = self.PyPDF2.PdfReader(file)
                    text = ""
                    for page_num in range(len(reader.pages)):
                        page = reader.pages[page_num]
                        text += page.extract_text() + "\n"
                    self.raw_text = text
                    print(f"✅ Resume extracted using PyPDF2 ({len(text)} chars)")
                    return text
            except Exception as e:
                print(f"⚠️  PyPDF2 extraction failed: {e}")
        
        print("❌ Could not extract PDF text. Ensure PyPDF2 or pdfplumber is installed.")
        return ""
    
    def extract_sections(self, text: Optional[str] = None) -> Dict[str, str]:
        """Extract common resume sections"""
        if text is None:
            text = self.raw_text or self.extract_text()
        
        if not text:
            return {}
        
        sections = {
            "summary": "",
            "skills": "",
            "experience": "",
            "education": "",
            "projects": "",
            "certifications": "",
            "contact": "",
        }
        
        # Define regex patterns for common sections
        patterns = {
            "summary": r"(PROFESSIONAL\s+SUMMARY|SUMMARY|OBJECTIVE|PROFILE)(.*?)(?=SKILLS|EXPERIENCE|EDUCATION|$)",
            "skills": r"(SKILLS|TECHNICAL\s+SKILLS|CORE\s+COMPETENCIES)(.*?)(?=EXPERIENCE|EDUCATION|PROJECTS|$)",
            "experience": r"(EXPERIENCE|WORK\s+EXPERIENCE|PROFESSIONAL\s+EXPERIENCE)(.*?)(?=EDUCATION|SKILLS|PROJECTS|$)",
            "education": r"(EDUCATION|ACADEMIC\s+BACKGROUND)(.*?)(?=EXPERIENCE|SKILLS|PROJECTS|CERTIFICATIONS|$)",
            "projects": r"(PROJECTS|PORTFOLIO|PERSONAL\s+PROJECTS)(.*?)(?=EXPERIENCE|EDUCATION|CERTIFICATIONS|$)",
            "certifications": r"(CERTIFICATIONS|LICENSES|AWARDS)(.*?)(?=EXPERIENCE|EDUCATION|PROJECTS|$)",
        }
        
        for section, pattern in patterns.items():
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                sections[section] = match.group(2).strip()[:500]  # Limit to 500 chars
        
        return sections
    
    def extract_contact_info(self, text: Optional[str] = None) -> Dict[str, str]:
        """Extract contact information"""
        if text is None:
            text = self.raw_text or self.extract_text()
        
        if not text:
            return {}
        
        contact = {}
        
        # Email
        email_match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
        if email_match:
            contact["email"] = email_match.group(0)
        
        # Phone (common patterns)
        phone_match = re.search(r'(?:\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b', text)
        if phone_match:
            contact["phone"] = phone_match.group(0)
        
        # LinkedIn
        linkedin_match = re.search(r'linkedin\.com/in/[\w-]+', text)
        if linkedin_match:
            contact["linkedin"] = linkedin_match.group(0)
        
        # GitHub
        github_match = re.search(r'github\.com/[\w-]+', text)
        if github_match:
            contact["github"] = github_match.group(0)
        
        # Website
        website_match = re.search(r'https?://[\w\.\-]+\.\w+', text)
        if website_match:
            contact["website"] = website_match.group(0)
        
        return contact
    
    def extract_skills(self, text: Optional[str] = None) -> List[str]:
        """Extract skills with better parsing"""
        if text is None:
            text = self.raw_text or self.extract_text()
        
        if not text:
            return []
        
        sections = self.extract_sections(text)
        skills_section = sections.get("skills", "") or sections.get("summary", "")
        
        # Common skill keywords
        skill_keywords = [
            # Languages
            r'\bPython\b', r'\bJavaScript\b', r'\bTypeScript\b', r'\bJava\b',
            r'\bC\+\+\b', r'\bC#\b', r'\bGo\b', r'\bRust\b', r'\bSQL\b',
            r'\bHTML\b', r'\bCSS\b', r'\bPHP\b', r'\bRuby\b', r'\bSwift\b',
            
            # Frameworks/Libraries
            r'\bReact\b', r'\bVue\b', r'\bAngular\b', r'\bDjango\b',
            r'\bFlask\b', r'\bFastAPI\b', r'\bNode\.js\b', r'\bExpress\b',
            r'\bTensorFlow\b', r'\bPyTorch\b', r'\bKeras\b', r'\bScikit-learn\b',
            
            # Tools
            r'\bGit\b', r'\bDocker\b', r'\bKubernetes\b', r'\bJenkins\b',
            r'\bAWS\b', r'\bAzure\b', r'\bGCP\b',
            
            # Concepts
            r'\bMachine Learning\b', r'\bDeep Learning\b', r'\bNLP\b',
            r'\bRESTful API\b', r'\bMicroservices\b', r'\bAgile\b',
        ]
        
        found_skills = set()
        for pattern in skill_keywords:
            if re.search(pattern, skills_section, re.IGNORECASE):
                # Extract the actual skill name
                match = re.search(pattern, skills_section, re.IGNORECASE)
                if match:
                    found_skills.add(match.group(0))
        
        return sorted(list(found_skills))
    
    def format_resume_for_rag(self, text: Optional[str] = None) -> str:
        """Format resume content for RAG embedding"""
        if text is None:
            text = self.raw_text or self.extract_text()
        
        if not text:
            return ""
        
        sections = self.extract_sections(text)
        contact = self.extract_contact_info(text)
        skills = self.extract_skills(text)
        
        formatted = """
RESUME INFORMATION:

"""
        
        if contact:
            formatted += "CONTACT INFORMATION:\n"
            for key, value in contact.items():
                formatted += f"  {key.upper()}: {value}\n"
            formatted += "\n"
        
        if sections.get("summary"):
            formatted += f"PROFESSIONAL SUMMARY:\n{sections['summary']}\n\n"
        
        if skills:
            formatted += f"KEY SKILLS:\n{', '.join(skills)}\n\n"
        
        if sections.get("experience"):
            formatted += f"PROFESSIONAL EXPERIENCE:\n{sections['experience']}\n\n"
        
        if sections.get("education"):
            formatted += f"EDUCATION:\n{sections['education']}\n\n"
        
        if sections.get("projects"):
            formatted += f"PROJECTS:\n{sections['projects']}\n\n"
        
        if sections.get("certifications"):
            formatted += f"CERTIFICATIONS:\n{sections['certifications']}\n\n"
        
        return formatted
    
    def process_resume(self) -> Dict:
        """Process entire resume and return structured data"""
        text = self.extract_text()
        
        if not text:
            return {}
        
        self.structured_data = {
            "raw_text": text,
            "sections": self.extract_sections(text),
            "contact": self.extract_contact_info(text),
            "skills": self.extract_skills(text),
            "formatted_text": self.format_resume_for_rag(text),
        }
        
        return self.structured_data
