# models/monet_concepts.py

import torch
import numpy as np
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class ConceptScore:
    """Single MONET concept with score and evidence"""
    name: str
    score: float
    confidence: float
    description: str
    clinical_relevance: str  # How this affects diagnosis

class MONETConceptScorer:
    """
    MONET concept scoring using your trained metadata patterns
    Integrates the boosting logic from your ensemble code
    """
    
    # MONET concepts used in your training
    CONCEPT_DEFINITIONS = {
        'MONET_ulceration_crust': {
            'description': 'Ulceration or crusting present',
            'high_in': ['SCCKA', 'BCC', 'MAL_OTH'],
            'low_in': ['NV', 'BKL'],
            'threshold_high': 0.50
        },
        'MONET_erythema': {
            'description': 'Redness or inflammation',
            'high_in': ['INF', 'BCC', 'SCCKA'],
            'low_in': ['MEL', 'NV'],
            'threshold_high': 0.40
        },
        'MONET_pigmented': {
            'description': 'Pigmentation present',
            'high_in': ['MEL', 'NV', 'BKL'],
            'low_in': ['BCC', 'SCCKA', 'INF'],
            'threshold_high': 0.55
        },
        'MONET_vasculature_vessels': {
            'description': 'Vascular structures visible',
            'high_in': ['VASC', 'BCC'],
            'low_in': ['MEL', 'NV'],
            'threshold_high': 0.35
        },
        'MONET_hair': {
            'description': 'Hair follicles present',
            'high_in': ['NV', 'BKL'],
            'low_in': ['BCC', 'MEL'],
            'threshold_high': 0.30
        },
        'MONET_gel_water_drop_fluid_dermoscopy_liquid': {
            'description': 'Gel/fluid artifacts',
            'high_in': [],
            'low_in': [],
            'threshold_high': 0.40
        },
        'MONET_skin_markings_pen_ink_purple_pen': {
            'description': 'Pen markings present',
            'high_in': [],
            'low_in': [],
            'threshold_high': 0.40
        }
    }
    
    # Class-specific patterns from your metadata boosting
    CLASS_PATTERNS = {
        'MAL_OTH': {
            'sex': 'male',  # 88.9% male
            'site_preference': 'trunk',
            'age_range': (60, 80),
            'key_concepts': {'MONET_ulceration_crust': 0.35}
        },
        'INF': {
            'key_concepts': {
                'MONET_erythema': 0.42,
                'MONET_pigmented': (None, 0.30)  # Low pigmentation
            }
        },
        'BEN_OTH': {
            'site_preference': ['head', 'neck', 'face'],  # 47.7%
            'key_concepts': {'MONET_pigmented': (0.30, 0.50)}
        },
        'DF': {
            'site_preference': ['lower', 'leg', 'ankle', 'foot'],  # 65.4%
            'age_range': (40, 65)
        },
        'SCCKA': {
            'age_range': (65, None),
            'key_concepts': {
                'MONET_ulceration_crust': 0.50,
                'MONET_pigmented': (None, 0.15)
            }
        },
        'MEL': {
            'age_range': (55, None),  # 61.8 years average
            'key_concepts': {'MONET_pigmented': 0.55}
        },
        'NV': {
            'age_range': (None, 45),  # 42.0 years average
            'key_concepts': {'MONET_pigmented': 0.55}
        }
    }
    
    def __init__(self):
        """Initialize MONET scorer with class patterns"""
        self.class_names = [
            'AKIEC', 'BCC', 'BEN_OTH', 'BKL', 'DF', 
            'INF', 'MAL_OTH', 'MEL', 'NV', 'SCCKA', 'VASC'
        ]
    
    def compute_concept_scores(
        self,
        metadata: Dict[str, float]
    ) -> Dict[str, ConceptScore]:
        """
        Compute MONET concept scores from metadata
        
        Args:
            metadata: Dictionary with MONET scores, age, sex, site, etc.
            
        Returns:
            Dictionary of concept scores
        """
        concept_scores = {}
        
        for concept_name, definition in self.CONCEPT_DEFINITIONS.items():
            score = metadata.get(concept_name, 0.0)
            
            # Determine confidence based on how extreme the score is
            if score > definition['threshold_high']:
                confidence = min((score - definition['threshold_high']) / 0.2, 1.0)
                level = "HIGH"
            elif score < 0.2:
                confidence = min((0.2 - score) / 0.2, 1.0)
                level = "LOW"
            else:
                confidence = 0.5
                level = "MODERATE"
            
            # Clinical relevance
            if level == "HIGH":
                relevant_classes = definition['high_in']
                clinical_relevance = f"Supports: {', '.join(relevant_classes)}"
            elif level == "LOW":
                excluded_classes = definition['low_in']
                clinical_relevance = f"Against: {', '.join(excluded_classes)}"
            else:
                clinical_relevance = "Non-specific"
            
            concept_scores[concept_name] = ConceptScore(
                name=concept_name.replace('MONET_', '').replace('_', ' ').title(),
                score=score,
                confidence=confidence,
                description=f"{definition['description']} ({level})",
                clinical_relevance=clinical_relevance
            )
        
        return concept_scores
    
    def apply_metadata_boosting(
        self,
        probs: np.ndarray,
        metadata: Dict
    ) -> np.ndarray:
        """
        Apply your metadata boosting logic
        This is directly from your ensemble optimization code
        
        Args:
            probs: [11] probability array
            metadata: Dictionary with age, sex, site, MONET scores
            
        Returns:
            boosted_probs: [11] adjusted probabilities
        """
        boosted_probs = probs.copy()
        
        # 1. MAL_OTH boosting
        if metadata.get('sex') == 'male':
            site = str(metadata.get('site', '')).lower()
            if 'trunk' in site:
                age = metadata.get('age_approx', 60)
                ulceration = metadata.get('MONET_ulceration_crust', 0)
                
                score = 0
                score += 3 if metadata.get('sex') == 'male' else 0
                score += 2 if 'trunk' in site else 0
                score += 1 if 60 <= age <= 80 else 0
                score += 2 if ulceration > 0.35 else 0
                
                confidence = score / 8.0
                if confidence > 0.5:
                    boosted_probs[6] *= (1.0 + confidence)  # MAL_OTH index
        
        # 2. INF boosting
        erythema = metadata.get('MONET_erythema', 0)
        pigmentation = metadata.get('MONET_pigmented', 0)
        
        if erythema > 0.42 and pigmentation < 0.30:
            confidence = min((erythema - 0.42) / 0.10 + 0.5, 1.0)
            boosted_probs[5] *= (1.0 + confidence * 0.8)  # INF index
        
        # 3. BEN_OTH boosting
        site = str(metadata.get('site', '')).lower()
        is_head_neck = any(x in site for x in ['head', 'neck', 'face'])
        
        if is_head_neck and 0.30 < pigmentation < 0.50:
            ulceration = metadata.get('MONET_ulceration_crust', 0)
            confidence = 0.7 if ulceration < 0.30 else 0.4
            boosted_probs[2] *= (1.0 + confidence * 0.5)  # BEN_OTH index
        
        # 4. DF boosting
        is_lower_ext = any(x in site for x in ['lower', 'leg', 'ankle', 'foot'])
        
        if is_lower_ext:
            age = metadata.get('age_approx', 60)
            if 40 <= age <= 65:
                boosted_probs[4] *= 1.8  # DF index
            elif 30 <= age <= 75:
                boosted_probs[4] *= 1.5
        
        # 5. SCCKA boosting
        ulceration = metadata.get('MONET_ulceration_crust', 0)
        age = metadata.get('age_approx', 60)
        
        if ulceration > 0.50 and age >= 65 and pigmentation < 0.15:
            boosted_probs[9] *= 1.9  # SCCKA index
        elif ulceration > 0.45 and age >= 60 and pigmentation < 0.20:
            boosted_probs[9] *= 1.5
        
        # 6. MEL vs NV age separation
        if pigmentation > 0.55:
            if age >= 55:
                age_score = min((age - 55) / 20.0, 1.0)
                boosted_probs[7] *= (1.0 + age_score * 0.5)  # MEL
                boosted_probs[8] *= (1.0 - age_score * 0.3)  # NV
            elif age <= 45:
                age_score = min((45 - age) / 30.0, 1.0)
                boosted_probs[7] *= (1.0 - age_score * 0.3)  # MEL
                boosted_probs[8] *= (1.0 + age_score * 0.5)  # NV
        
        # 7. Exclusions based on pigmentation/erythema
        if pigmentation > 0.50:
            boosted_probs[0] *= 0.7  # AKIEC
            boosted_probs[1] *= 0.6  # BCC
            boosted_probs[5] *= 0.5  # INF
            boosted_probs[9] *= 0.3  # SCCKA
        
        if erythema > 0.40:
            boosted_probs[7] *= 0.7  # MEL
            boosted_probs[8] *= 0.7  # NV
        
        if pigmentation < 0.20:
            boosted_probs[7] *= 0.5  # MEL
            boosted_probs[8] *= 0.5  # NV
        
        # Renormalize
        return boosted_probs / boosted_probs.sum()
    
    def explain_prediction(
        self,
        probs: np.ndarray,
        concept_scores: Dict[str, ConceptScore],
        metadata: Dict
    ) -> str:
        """
        Generate natural language explanation
        
        Args:
            probs: Class probabilities
            concept_scores: MONET concept scores
            metadata: Clinical metadata
            
        Returns:
            Natural language explanation
        """
        predicted_idx = np.argmax(probs)
        predicted_class = self.class_names[predicted_idx]
        confidence = probs[predicted_idx]
        
        explanation = f"**Primary Diagnosis: {predicted_class}**\n"
        explanation += f"Confidence: {confidence:.1%}\n\n"
        
        # Key MONET features
        explanation += "**Key Dermoscopic Features:**\n"
        
        sorted_concepts = sorted(
            concept_scores.values(),
            key=lambda x: x.score * x.confidence,
            reverse=True
        )
        
        for i, concept in enumerate(sorted_concepts[:5], 1):
            if concept.score > 0.3 or concept.score < 0.2:
                explanation += f"{i}. {concept.name}: {concept.score:.2f} - {concept.description}\n"
                if concept.clinical_relevance != "Non-specific":
                    explanation += f"   → {concept.clinical_relevance}\n"
        
        # Clinical context
        explanation += "\n**Clinical Context:**\n"
        if 'age_approx' in metadata:
            explanation += f"• Age: {metadata['age_approx']} years\n"
        if 'sex' in metadata:
            explanation += f"• Sex: {metadata['sex']}\n"
        if 'site' in metadata:
            explanation += f"• Location: {metadata['site']}\n"
        
        return explanation
    
    def get_top_concepts(
        self,
        concept_scores: Dict[str, ConceptScore],
        top_k: int = 5,
        min_score: float = 0.3
    ) -> List[ConceptScore]:
        """Get top-k most important concepts"""
        filtered = [
            cs for cs in concept_scores.values() 
            if cs.score >= min_score or cs.score < 0.2  # High or low
        ]
        
        sorted_concepts = sorted(
            filtered,
            key=lambda x: x.score * x.confidence,
            reverse=True
        )
        
        return sorted_concepts[:top_k]