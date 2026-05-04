"""
Entity Extractor - Week 9 Day 1
Extract named entities from text using spaCy NER.
"""

import spacy
from typing import List
import re


class Entity:
    """Represents an extracted entity."""
    
    def __init__(self, text: str, label: str, start: int, end: int):
        self.text = text
        self.label = label
        self.start = start
        self.end = end
        self.normalized = self._normalize(text)
    
    def _normalize(self, text: str) -> str:
        """Normalize entity text (lowercase, strip)."""
        return text.strip().lower()
    
    def __repr__(self):
        return f"Entity({self.text}, {self.label})"


class EntityExtractor:
    """
    Extract entities from text using spaCy.
    
    Extracts:
    - PERSON: People, authors
    - ORG: Organizations, companies
    - PRODUCT: Products, technologies
    - CONCEPT: Key concepts, methods
    - GPE: Countries, cities
    """
    
    def __init__(self, model: str = "en_core_web_lg"):
        """Initialize with spaCy model."""
        self.nlp = spacy.load(model)
        
        # Entity types to extract
        self.target_labels = {
            'PERSON', 'ORG', 'PRODUCT', 'GPE',
            'WORK_OF_ART', 'LAW', 'LANGUAGE',
            'NORP', 'FAC', 'EVENT'
        }
        
        # ========== ADD CUSTOM TECH PATTERNS ==========
        # Known tech terms often missed by NER
        self.tech_terms = {
            # Languages
            'python', 'java', 'javascript', 'c++', 'rust', 'go',
            
            # Frameworks
            'tensorflow', 'pytorch', 'keras', 'react', 'angular', 'vue',
            'django', 'flask', 'fastapi', 'spring', 'express',
            
            # Cloud/Infra
            'docker', 'kubernetes', 'aws', 'gcp', 'azure',
            
            # ML/AI Concepts
            'machine learning', 'deep learning', 'ai', 'artificial intelligence',
            'neural network', 'neural networks', 'nlp', 'computer vision',
            'reinforcement learning', 'supervised learning', 'unsupervised learning',
            'rag', 'retrieval augmented generation', 'retrieval-augmented generation',
            'rag pipeline', 'retrieval pipeline', 'generation pipeline',
            'retriever', 'generator',
            
            # General Tech Concepts  
            'development', 'programming', 'software', 'application', 'system',
            'platform', 'framework', 'library', 'algorithm', 'model',
            'accuracy', 'performance', 'scalability', 'deployment',
            'api', 'database', 'architecture', 'microservices'
        }
    def extract(self, text: str) -> List[Entity]:
        """Extract entities with custom tech term detection."""
        
        doc = self.nlp(text)
        
        entities = []
        
        # Standard NER entities
        matched_spans = []
        for ent in doc.ents:
            if ent.label_ in self.target_labels:
                entity = Entity(
                    text=ent.text,
                    label=ent.label_,
                    start=ent.start_char,
                    end=ent.end_char
                )
                entities.append(entity)
                matched_spans.append((ent.start_char, ent.end_char))

        # Domain papers often use short acronyms that spaCy misses in terse
        # sub-queries, e.g. "AGCD two phases names".
        for match in re.finditer(r"(?<![A-Za-z0-9])(?:[A-Z][A-Z0-9]{1,})(?![A-Za-z0-9])", text):
            start, end = match.span()
            if any(start < existing_end and end > existing_start for existing_start, existing_end in matched_spans):
                continue
            matched_spans.append((start, end))
            entities.append(
                Entity(
                    text=match.group(0),
                    label="ACRONYM",
                    start=start,
                    end=end,
                )
            )
        
        # ========== ADD CUSTOM TECH TERMS ==========
        # Find tech terms manually
        text_lower = text.lower()
        for term in sorted(self.tech_terms, key=len, reverse=True):
            pattern = self._term_pattern(term)
            match = re.search(pattern, text_lower)
            if not match:
                continue
            start, end = match.span()
            if any(start < existing_end and end > existing_start for existing_start, existing_end in matched_spans):
                continue
            matched_spans.append((start, end))
            entity = Entity(
                text=text[start:end],
                label='TECH',  # Custom label
                start=start,
                end=end
            )
            entities.append(entity)
        # ========== END CUSTOM ==========
        
        return entities

    def _term_pattern(self, term: str) -> str:
        """Build a safe whole-term regex for custom tech vocabulary."""
        escaped = re.escape(term.lower()).replace(r"\ ", r"\s+")
        return rf"(?<!\w){escaped}(?!\w)"
    
