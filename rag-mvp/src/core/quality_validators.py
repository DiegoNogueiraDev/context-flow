"""
Quality Validation Components for Enhanced RAG System

This module provides specialized validators for different aspects of the RAG pipeline,
supporting the main Quality Framework with detailed validation logic.
"""

import numpy as np
import re
import json
import logging
from typing import List, Dict, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
from collections import Counter, defaultdict
from enum import Enum
import statistics
from datetime import datetime, timedelta

from .models import Document, Chunk


class ValidationSeverity(Enum):
    """Severity levels for validation issues"""
    CRITICAL = "critical"    # System-breaking issues
    HIGH = "high"           # Major quality problems
    MEDIUM = "medium"       # Moderate quality issues  
    LOW = "low"            # Minor quality concerns
    INFO = "info"          # Informational findings


@dataclass
class ValidationIssue:
    """Individual validation issue"""
    issue_type: str
    severity: ValidationSeverity
    message: str
    affected_component: str
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'issue_type': self.issue_type,
            'severity': self.severity.value,
            'message': self.message,
            'affected_component': self.affected_component,
            'confidence': self.confidence,
            'metadata': self.metadata,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class ValidationResult:
    """Comprehensive validation result"""
    is_valid: bool
    overall_score: float
    issues: List[ValidationIssue] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    validation_duration: float = 0.0
    
    def get_issues_by_severity(self, severity: ValidationSeverity) -> List[ValidationIssue]:
        """Get issues filtered by severity"""
        return [issue for issue in self.issues if issue.severity == severity]
    
    def get_critical_issues_count(self) -> int:
        """Get count of critical issues"""
        return len(self.get_issues_by_severity(ValidationSeverity.CRITICAL))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'is_valid': self.is_valid,
            'overall_score': self.overall_score,
            'issues': [issue.to_dict() for issue in self.issues],
            'metrics': self.metrics,
            'recommendations': self.recommendations,
            'validation_duration': self.validation_duration,
            'critical_issues_count': self.get_critical_issues_count()
        }


class DocumentContentValidator:
    """Validates document content quality and consistency"""
    
    def __init__(self):
        self.min_content_length = 50
        self.max_content_length = 10_000_000  # 10MB of text
        self.min_word_count = 10
        self.suspicious_patterns = [
            r'�',  # Encoding issues
            r'\x00',  # Null bytes
            r'[\x01-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]',  # Control characters
        ]
    
    def validate(self, document: Document) -> ValidationResult:
        """Validate document content quality"""
        issues = []
        metrics = {}
        recommendations = []
        
        # Content length validation
        content_length = len(document.content)
        metrics['content_length'] = content_length
        
        if content_length < self.min_content_length:
            issues.append(ValidationIssue(
                issue_type='content_too_short',
                severity=ValidationSeverity.HIGH,
                message=f'Document content too short: {content_length} chars (minimum: {self.min_content_length})',
                affected_component='document_content'
            ))
            recommendations.append('Ensure document has sufficient content for meaningful processing')
        
        if content_length > self.max_content_length:
            issues.append(ValidationIssue(
                issue_type='content_too_long',
                severity=ValidationSeverity.MEDIUM,
                message=f'Document content very long: {content_length} chars (consider chunking)',
                affected_component='document_content'
            ))
            recommendations.append('Consider splitting large documents for better processing')
        
        # Word count validation
        word_count = len(document.content.split())
        metrics['word_count'] = word_count
        
        if word_count < self.min_word_count:
            issues.append(ValidationIssue(
                issue_type='insufficient_words',
                severity=ValidationSeverity.HIGH,
                message=f'Document has too few words: {word_count} (minimum: {self.min_word_count})',
                affected_component='document_content'
            ))
        
        # Character encoding validation
        encoding_issues = 0
        for pattern in self.suspicious_patterns:
            matches = re.findall(pattern, document.content)
            if matches:
                encoding_issues += len(matches)
        
        metrics['encoding_issues'] = encoding_issues
        if encoding_issues > 0:
            severity = ValidationSeverity.HIGH if encoding_issues > 10 else ValidationSeverity.MEDIUM
            issues.append(ValidationIssue(
                issue_type='encoding_issues',
                severity=severity,
                message=f'Found {encoding_issues} potential encoding issues in document',
                affected_component='document_content',
                metadata={'issue_count': encoding_issues}
            ))
            recommendations.append('Check document encoding and preprocessing')
        
        # Language detection and consistency
        language_score = self._assess_language_consistency(document.content)
        metrics['language_consistency'] = language_score
        
        if language_score < 0.7:
            issues.append(ValidationIssue(
                issue_type='mixed_languages',
                severity=ValidationSeverity.LOW,
                message=f'Document may contain mixed languages (consistency: {language_score:.2f})',
                affected_component='document_content'
            ))
        
        # Content structure validation
        structure_score = self._assess_content_structure(document.content)
        metrics['structure_score'] = structure_score
        
        if structure_score < 0.5:
            issues.append(ValidationIssue(
                issue_type='poor_structure',
                severity=ValidationSeverity.MEDIUM,
                message=f'Document has poor structural organization (score: {structure_score:.2f})',
                affected_component='document_content'
            ))
            recommendations.append('Improve document structure with headers and paragraphs')
        
        # Calculate overall score
        overall_score = self._calculate_content_quality_score(metrics, issues)
        
        return ValidationResult(
            is_valid=len([i for i in issues if i.severity in [ValidationSeverity.CRITICAL, ValidationSeverity.HIGH]]) == 0,
            overall_score=overall_score,
            issues=issues,
            metrics=metrics,
            recommendations=recommendations
        )
    
    def _assess_language_consistency(self, content: str) -> float:
        """Assess language consistency in content"""
        # Simple heuristic - check for consistent character patterns
        # In practice, would use language detection libraries
        
        # Count different character types
        latin_chars = len(re.findall(r'[a-zA-Z]', content))
        total_chars = len(re.findall(r'[^\s\d\W]', content))
        
        if total_chars == 0:
            return 0.0
        
        consistency = latin_chars / total_chars
        return min(consistency, 1.0)
    
    def _assess_content_structure(self, content: str) -> float:
        """Assess structural quality of content"""
        score = 0.5  # Base score
        
        # Check for headers
        if re.search(r'^#{1,6}\s', content, re.MULTILINE):
            score += 0.2
        
        # Check for paragraphs
        paragraph_count = len(content.split('\n\n'))
        if paragraph_count > 2:
            score += 0.2
        
        # Check for lists
        if re.search(r'^[-*+]\s', content, re.MULTILINE) or re.search(r'^\d+\.\s', content, re.MULTILINE):
            score += 0.1
        
        return min(score, 1.0)
    
    def _calculate_content_quality_score(self, metrics: Dict[str, float], 
                                       issues: List[ValidationIssue]) -> float:
        """Calculate overall content quality score"""
        base_score = 1.0
        
        # Deduct for issues
        for issue in issues:
            if issue.severity == ValidationSeverity.CRITICAL:
                base_score -= 0.3
            elif issue.severity == ValidationSeverity.HIGH:
                base_score -= 0.2
            elif issue.severity == ValidationSeverity.MEDIUM:
                base_score -= 0.1
            elif issue.severity == ValidationSeverity.LOW:
                base_score -= 0.05
        
        # Add for positive metrics
        if metrics.get('structure_score', 0) > 0.8:
            base_score += 0.1
        if metrics.get('language_consistency', 0) > 0.9:
            base_score += 0.05
        
        return max(min(base_score, 1.0), 0.0)


class ChunkQualityValidator:
    """Validates quality of document chunking"""
    
    def __init__(self):
        self.min_chunk_length = 50
        self.max_chunk_length = 5000
        self.optimal_chunk_length = 1000
        self.max_chunk_overlap_ratio = 0.3
        self.min_chunk_count = 1
    
    def validate(self, document: Document) -> ValidationResult:
        """Validate chunk quality"""
        issues = []
        metrics = {}
        recommendations = []
        
        if not document.chunks:
            issues.append(ValidationIssue(
                issue_type='no_chunks',
                severity=ValidationSeverity.CRITICAL,
                message='Document has no chunks',
                affected_component='chunking'
            ))
            return ValidationResult(
                is_valid=False,
                overall_score=0.0,
                issues=issues,
                recommendations=['Process document to generate chunks']
            )
        
        chunk_lengths = [len(chunk.content) for chunk in document.chunks]
        metrics['chunk_count'] = len(document.chunks)
        metrics['avg_chunk_length'] = statistics.mean(chunk_lengths)
        metrics['chunk_length_variance'] = statistics.variance(chunk_lengths) if len(chunk_lengths) > 1 else 0
        
        # Chunk count validation
        if len(document.chunks) < self.min_chunk_count:
            issues.append(ValidationIssue(
                issue_type='insufficient_chunks',
                severity=ValidationSeverity.HIGH,
                message=f'Too few chunks: {len(document.chunks)} (minimum: {self.min_chunk_count})',
                affected_component='chunking'
            ))
        
        # Chunk length validation
        short_chunks = [i for i, length in enumerate(chunk_lengths) if length < self.min_chunk_length]
        long_chunks = [i for i, length in enumerate(chunk_lengths) if length > self.max_chunk_length]
        
        metrics['short_chunks_count'] = len(short_chunks)
        metrics['long_chunks_count'] = len(long_chunks)
        
        if short_chunks:
            severity = ValidationSeverity.HIGH if len(short_chunks) > len(chunk_lengths) * 0.3 else ValidationSeverity.MEDIUM
            issues.append(ValidationIssue(
                issue_type='chunks_too_short',
                severity=severity,
                message=f'{len(short_chunks)} chunks are too short (< {self.min_chunk_length} chars)',
                affected_component='chunking',
                metadata={'short_chunk_indices': short_chunks}
            ))
            recommendations.append('Increase minimum chunk size or merge short chunks')
        
        if long_chunks:
            issues.append(ValidationIssue(
                issue_type='chunks_too_long',
                severity=ValidationSeverity.MEDIUM,
                message=f'{len(long_chunks)} chunks are very long (> {self.max_chunk_length} chars)',
                affected_component='chunking',
                metadata={'long_chunk_indices': long_chunks}
            ))
            recommendations.append('Consider splitting very long chunks')
        
        # Chunk overlap validation
        overlap_score = self._assess_chunk_overlap(document.chunks)
        metrics['chunk_overlap_score'] = overlap_score
        
        if overlap_score > self.max_chunk_overlap_ratio:
            issues.append(ValidationIssue(
                issue_type='excessive_chunk_overlap',
                severity=ValidationSeverity.MEDIUM,
                message=f'High chunk overlap detected: {overlap_score:.2%}',
                affected_component='chunking'
            ))
            recommendations.append('Reduce chunk overlap for better content coverage')
        
        # Content coverage validation
        coverage_score = self._assess_content_coverage(document)
        metrics['content_coverage'] = coverage_score
        
        if coverage_score < 0.8:
            issues.append(ValidationIssue(
                issue_type='poor_content_coverage',
                severity=ValidationSeverity.HIGH,
                message=f'Chunks cover only {coverage_score:.1%} of document content',
                affected_component='chunking'
            ))
            recommendations.append('Improve chunking to cover more document content')
        
        # Semantic boundary assessment
        boundary_score = self._assess_semantic_boundaries(document.chunks)
        metrics['semantic_boundary_score'] = boundary_score
        
        if boundary_score < 0.6:
            issues.append(ValidationIssue(
                issue_type='poor_semantic_boundaries',
                severity=ValidationSeverity.MEDIUM,
                message=f'Chunks may not follow semantic boundaries (score: {boundary_score:.2f})',
                affected_component='chunking'
            ))
            recommendations.append('Use semantic-aware chunking methods')
        
        # Calculate overall score
        overall_score = self._calculate_chunking_quality_score(metrics, issues)
        
        return ValidationResult(
            is_valid=len([i for i in issues if i.severity == ValidationSeverity.CRITICAL]) == 0,
            overall_score=overall_score,
            issues=issues,
            metrics=metrics,
            recommendations=recommendations
        )
    
    def _assess_chunk_overlap(self, chunks: List[Chunk]) -> float:
        """Assess overlap between chunks"""
        if len(chunks) < 2:
            return 0.0
        
        total_overlaps = 0
        comparisons = 0
        
        for i, chunk1 in enumerate(chunks):
            for chunk2 in chunks[i+1:]:
                words1 = set(chunk1.content.lower().split())
                words2 = set(chunk2.content.lower().split())
                
                if words1 and words2:
                    overlap = len(words1.intersection(words2))
                    union = len(words1.union(words2))
                    overlap_ratio = overlap / union if union > 0 else 0
                    total_overlaps += overlap_ratio
                    comparisons += 1
        
        return total_overlaps / comparisons if comparisons > 0 else 0.0
    
    def _assess_content_coverage(self, document: Document) -> float:
        """Assess how well chunks cover original document content"""
        if not document.chunks:
            return 0.0
        
        original_words = set(document.content.lower().split())
        chunk_words = set()
        
        for chunk in document.chunks:
            chunk_words.update(chunk.content.lower().split())
        
        if not original_words:
            return 1.0
        
        coverage = len(chunk_words.intersection(original_words)) / len(original_words)
        return min(coverage, 1.0)
    
    def _assess_semantic_boundaries(self, chunks: List[Chunk]) -> float:
        """Assess if chunks respect semantic boundaries"""
        score = 0.0
        
        for chunk in chunks:
            content = chunk.content.strip()
            
            # Check for sentence boundaries
            if content.endswith(('.', '!', '?')):
                score += 0.3
            
            # Check for paragraph boundaries
            if '\n\n' in content or content.startswith('\n'):
                score += 0.2
            
            # Check for section boundaries
            if content.startswith('#') or 'section' in content.lower()[:50]:
                score += 0.3
            
            # Check for natural breaks
            if any(phrase in content.lower()[:100] for phrase in ['however', 'furthermore', 'in conclusion', 'meanwhile']):
                score += 0.2
        
        return min(score / len(chunks) if chunks else 0.0, 1.0)
    
    def _calculate_chunking_quality_score(self, metrics: Dict[str, float], 
                                        issues: List[ValidationIssue]) -> float:
        """Calculate overall chunking quality score"""
        base_score = 1.0
        
        # Deduct for issues
        for issue in issues:
            if issue.severity == ValidationSeverity.CRITICAL:
                base_score -= 0.4
            elif issue.severity == ValidationSeverity.HIGH:
                base_score -= 0.25
            elif issue.severity == ValidationSeverity.MEDIUM:
                base_score -= 0.15
            elif issue.severity == ValidationSeverity.LOW:
                base_score -= 0.05
        
        # Add for good metrics
        if metrics.get('content_coverage', 0) > 0.9:
            base_score += 0.1
        if metrics.get('semantic_boundary_score', 0) > 0.8:
            base_score += 0.1
        
        return max(min(base_score, 1.0), 0.0)


class EmbeddingQualityValidator:
    """Validates quality of embeddings and vector representations"""
    
    def __init__(self):
        self.min_embedding_dimension = 100
        self.max_embedding_dimension = 2000
        self.expected_norm_range = (0.5, 2.0)
        self.similarity_threshold = 0.95  # For detecting identical embeddings
    
    def validate(self, embeddings: List[np.ndarray], 
                chunks: Optional[List[Chunk]] = None) -> ValidationResult:
        """Validate embedding quality"""
        issues = []
        metrics = {}
        recommendations = []
        
        if not embeddings:
            issues.append(ValidationIssue(
                issue_type='no_embeddings',
                severity=ValidationSeverity.CRITICAL,
                message='No embeddings provided for validation',
                affected_component='embeddings'
            ))
            return ValidationResult(
                is_valid=False,
                overall_score=0.0,
                issues=issues,
                recommendations=['Generate embeddings for document chunks']
            )
        
        # Dimension consistency validation
        dimensions = [emb.shape[0] if emb.ndim == 1 else emb.shape[-1] for emb in embeddings]
        unique_dimensions = set(dimensions)
        
        metrics['embedding_count'] = len(embeddings)
        metrics['dimension'] = dimensions[0] if dimensions else 0
        metrics['dimension_consistency'] = len(unique_dimensions) == 1
        
        if len(unique_dimensions) > 1:
            issues.append(ValidationIssue(
                issue_type='inconsistent_dimensions',
                severity=ValidationSeverity.CRITICAL,
                message=f'Embeddings have inconsistent dimensions: {unique_dimensions}',
                affected_component='embeddings',
                metadata={'dimensions_found': list(unique_dimensions)}
            ))
        
        # Dimension range validation
        if dimensions and dimensions[0] < self.min_embedding_dimension:
            issues.append(ValidationIssue(
                issue_type='dimension_too_small',
                severity=ValidationSeverity.HIGH,
                message=f'Embedding dimension too small: {dimensions[0]} (minimum: {self.min_embedding_dimension})',
                affected_component='embeddings'
            ))
            recommendations.append('Use higher-dimensional embeddings for better representation')
        
        if dimensions and dimensions[0] > self.max_embedding_dimension:
            issues.append(ValidationIssue(
                issue_type='dimension_too_large',
                severity=ValidationSeverity.LOW,
                message=f'Very high embedding dimension: {dimensions[0]} (consider dimensionality reduction)',
                affected_component='embeddings'
            ))
        
        # Vector norm validation
        norms = [np.linalg.norm(emb) for emb in embeddings if emb.size > 0]
        if norms:
            metrics['avg_norm'] = statistics.mean(norms)
            metrics['min_norm'] = min(norms)
            metrics['max_norm'] = max(norms)
            metrics['norm_variance'] = statistics.variance(norms) if len(norms) > 1 else 0
            
            # Check for zero vectors
            zero_vectors = sum(1 for norm in norms if norm < 1e-10)
            if zero_vectors > 0:
                issues.append(ValidationIssue(
                    issue_type='zero_vectors',
                    severity=ValidationSeverity.HIGH,
                    message=f'{zero_vectors} embeddings are zero or near-zero vectors',
                    affected_component='embeddings',
                    metadata={'zero_vector_count': zero_vectors}
                ))
                recommendations.append('Check embedding generation process for zero vectors')
            
            # Check norm distribution
            out_of_range_norms = [norm for norm in norms 
                                if norm < self.expected_norm_range[0] or norm > self.expected_norm_range[1]]
            if len(out_of_range_norms) > len(norms) * 0.1:  # More than 10%
                issues.append(ValidationIssue(
                    issue_type='unusual_norm_distribution',
                    severity=ValidationSeverity.MEDIUM,
                    message=f'{len(out_of_range_norms)} embeddings have unusual norms',
                    affected_component='embeddings'
                ))
        
        # Similarity analysis
        similarity_analysis = self._analyze_embedding_similarities(embeddings)
        metrics.update(similarity_analysis['metrics'])
        
        if similarity_analysis['identical_pairs'] > 0:
            severity = ValidationSeverity.HIGH if similarity_analysis['identical_pairs'] > len(embeddings) * 0.1 else ValidationSeverity.MEDIUM
            issues.append(ValidationIssue(
                issue_type='identical_embeddings',
                severity=severity,
                message=f'{similarity_analysis["identical_pairs"]} pairs of nearly identical embeddings found',
                affected_component='embeddings',
                metadata={'identical_pairs': similarity_analysis['identical_pairs']}
            ))
            recommendations.append('Check for duplicate content or embedding generation issues')
        
        # Content-embedding alignment validation (if chunks provided)
        if chunks and len(chunks) == len(embeddings):
            alignment_score = self._assess_content_embedding_alignment(chunks, embeddings)
            metrics['content_alignment_score'] = alignment_score
            
            if alignment_score < 0.7:
                issues.append(ValidationIssue(
                    issue_type='poor_content_alignment',
                    severity=ValidationSeverity.MEDIUM,
                    message=f'Poor alignment between content and embeddings (score: {alignment_score:.2f})',
                    affected_component='embeddings'
                ))
                recommendations.append('Review embedding model selection and preprocessing')
        
        # Calculate overall score
        overall_score = self._calculate_embedding_quality_score(metrics, issues)
        
        return ValidationResult(
            is_valid=len([i for i in issues if i.severity == ValidationSeverity.CRITICAL]) == 0,
            overall_score=overall_score,
            issues=issues,
            metrics=metrics,
            recommendations=recommendations
        )
    
    def _analyze_embedding_similarities(self, embeddings: List[np.ndarray]) -> Dict[str, Any]:
        """Analyze pairwise similarities between embeddings"""
        if len(embeddings) < 2:
            return {'metrics': {'avg_similarity': 0, 'max_similarity': 0}, 'identical_pairs': 0}
        
        similarities = []
        identical_pairs = 0
        
        # Sample pairs to avoid O(n²) complexity for large collections
        max_pairs = min(1000, len(embeddings) * (len(embeddings) - 1) // 2)
        pair_count = 0
        
        for i, emb1 in enumerate(embeddings):
            for j, emb2 in enumerate(embeddings[i+1:], i+1):
                if pair_count >= max_pairs:
                    break
                
                # Cosine similarity
                if emb1.size > 0 and emb2.size > 0:
                    norm1, norm2 = np.linalg.norm(emb1), np.linalg.norm(emb2)
                    if norm1 > 0 and norm2 > 0:
                        similarity = np.dot(emb1, emb2) / (norm1 * norm2)
                        similarities.append(float(similarity))
                        
                        if similarity > self.similarity_threshold:
                            identical_pairs += 1
                
                pair_count += 1
            
            if pair_count >= max_pairs:
                break
        
        metrics = {
            'avg_similarity': statistics.mean(similarities) if similarities else 0,
            'max_similarity': max(similarities) if similarities else 0,
            'min_similarity': min(similarities) if similarities else 0,
            'similarity_std': statistics.stdev(similarities) if len(similarities) > 1 else 0,
            'pairs_analyzed': len(similarities)
        }
        
        return {'metrics': metrics, 'identical_pairs': identical_pairs}
    
    def _assess_content_embedding_alignment(self, chunks: List[Chunk], 
                                          embeddings: List[np.ndarray]) -> float:
        """Assess how well embeddings represent their corresponding content"""
        if len(chunks) != len(embeddings):
            return 0.0
        
        alignment_scores = []
        
        for i, (chunk1, emb1) in enumerate(zip(chunks, embeddings)):
            best_content_similarity = 0
            best_embedding_similarity = 0
            
            # Compare with other chunks
            for j, (chunk2, emb2) in enumerate(zip(chunks, embeddings)):
                if i != j:
                    # Content similarity (simple word overlap)
                    words1 = set(chunk1.content.lower().split())
                    words2 = set(chunk2.content.lower().split())
                    if words1 and words2:
                        content_sim = len(words1.intersection(words2)) / len(words1.union(words2))
                        
                        # Embedding similarity
                        if emb1.size > 0 and emb2.size > 0:
                            norm1, norm2 = np.linalg.norm(emb1), np.linalg.norm(emb2)
                            if norm1 > 0 and norm2 > 0:
                                emb_sim = np.dot(emb1, emb2) / (norm1 * norm2)
                                
                                if content_sim > best_content_similarity:
                                    best_content_similarity = content_sim
                                    best_embedding_similarity = emb_sim
            
            # Alignment score: how well embedding similarity matches content similarity
            if best_content_similarity > 0:
                alignment = 1 - abs(best_content_similarity - best_embedding_similarity)
                alignment_scores.append(alignment)
        
        return statistics.mean(alignment_scores) if alignment_scores else 0.5
    
    def _calculate_embedding_quality_score(self, metrics: Dict[str, float], 
                                         issues: List[ValidationIssue]) -> float:
        """Calculate overall embedding quality score"""
        base_score = 1.0
        
        # Deduct for issues
        for issue in issues:
            if issue.severity == ValidationSeverity.CRITICAL:
                base_score -= 0.5
            elif issue.severity == ValidationSeverity.HIGH:
                base_score -= 0.3
            elif issue.severity == ValidationSeverity.MEDIUM:
                base_score -= 0.15
            elif issue.severity == ValidationSeverity.LOW:
                base_score -= 0.05
        
        # Add for good metrics
        if metrics.get('dimension_consistency', False):
            base_score += 0.1
        if metrics.get('content_alignment_score', 0) > 0.8:
            base_score += 0.15
        
        return max(min(base_score, 1.0), 0.0)


class SearchQualityValidator:
    """Validates quality of search results and relevance"""
    
    def __init__(self):
        self.min_relevance_threshold = 0.3
        self.diversity_threshold = 0.7
        self.min_result_count = 1
    
    def validate(self, query: str, results: List[Dict[str, Any]], 
                expected_results: Optional[List[str]] = None) -> ValidationResult:
        """Validate search result quality"""
        issues = []
        metrics = {}
        recommendations = []
        
        if not results:
            issues.append(ValidationIssue(
                issue_type='no_results',
                severity=ValidationSeverity.HIGH,
                message='Search returned no results',
                affected_component='search'
            ))
            recommendations.append('Check query processing and index coverage')
            return ValidationResult(
                is_valid=False,
                overall_score=0.0,
                issues=issues,
                recommendations=recommendations
            )
        
        # Basic result metrics
        metrics['result_count'] = len(results)
        similarities = [r.get('similarity', 0.0) for r in results]
        
        if similarities:
            metrics['avg_similarity'] = statistics.mean(similarities)
            metrics['min_similarity'] = min(similarities)
            metrics['max_similarity'] = max(similarities)
            metrics['similarity_range'] = max(similarities) - min(similarities)
        
        # Relevance validation
        low_relevance_count = sum(1 for sim in similarities if sim < self.min_relevance_threshold)
        metrics['low_relevance_ratio'] = low_relevance_count / len(results)
        
        if low_relevance_count > len(results) * 0.5:  # More than 50% low relevance
            issues.append(ValidationIssue(
                issue_type='low_relevance_results',
                severity=ValidationSeverity.HIGH,
                message=f'{low_relevance_count} results have low relevance (< {self.min_relevance_threshold})',
                affected_component='search'
            ))
            recommendations.append('Improve query processing or embedding quality')
        
        # Result diversity validation
        diversity_score = self._assess_result_diversity(results)
        metrics['diversity_score'] = diversity_score
        
        if diversity_score < self.diversity_threshold:
            issues.append(ValidationIssue(
                issue_type='low_result_diversity',
                severity=ValidationSeverity.MEDIUM,
                message=f'Low result diversity (score: {diversity_score:.2f})',
                affected_component='search'
            ))
            recommendations.append('Improve result diversification algorithms')
        
        # Query-result alignment validation
        alignment_score = self._assess_query_result_alignment(query, results)
        metrics['query_alignment_score'] = alignment_score
        
        if alignment_score < 0.6:
            issues.append(ValidationIssue(
                issue_type='poor_query_alignment',
                severity=ValidationSeverity.MEDIUM,
                message=f'Poor alignment between query and results (score: {alignment_score:.2f})',
                affected_component='search'
            ))
            recommendations.append('Review query understanding and matching algorithms')
        
        # Expected results validation (if provided)
        if expected_results:
            precision_recall = self._calculate_precision_recall(results, expected_results)
            metrics.update(precision_recall)
            
            if precision_recall['precision'] < 0.7:
                issues.append(ValidationIssue(
                    issue_type='low_precision',
                    severity=ValidationSeverity.HIGH,
                    message=f'Low precision: {precision_recall["precision"]:.2%}',
                    affected_component='search'
                ))
            
            if precision_recall['recall'] < 0.7:
                issues.append(ValidationIssue(
                    issue_type='low_recall',
                    severity=ValidationSeverity.HIGH,
                    message=f'Low recall: {precision_recall["recall"]:.2%}',
                    affected_component='search'
                ))
        
        # Calculate overall score
        overall_score = self._calculate_search_quality_score(metrics, issues)
        
        return ValidationResult(
            is_valid=len([i for i in issues if i.severity in [ValidationSeverity.CRITICAL, ValidationSeverity.HIGH]]) == 0,
            overall_score=overall_score,
            issues=issues,
            metrics=metrics,
            recommendations=recommendations
        )
    
    def _assess_result_diversity(self, results: List[Dict[str, Any]]) -> float:
        """Assess diversity of search results"""
        if len(results) < 2:
            return 1.0
        
        # Document diversity
        unique_docs = len(set(r.get('document_id', '') for r in results))
        doc_diversity = unique_docs / len(results)
        
        # Content diversity
        content_similarities = []
        for i, result1 in enumerate(results):
            for result2 in results[i+1:]:
                content1 = result1.get('content', '').lower()
                content2 = result2.get('content', '').lower()
                
                if content1 and content2:
                    words1 = set(content1.split())
                    words2 = set(content2.split())
                    
                    if words1 and words2:
                        similarity = len(words1.intersection(words2)) / len(words1.union(words2))
                        content_similarities.append(similarity)
        
        avg_content_similarity = statistics.mean(content_similarities) if content_similarities else 0
        content_diversity = 1 - avg_content_similarity
        
        return (doc_diversity + content_diversity) / 2
    
    def _assess_query_result_alignment(self, query: str, results: List[Dict[str, Any]]) -> float:
        """Assess how well results align with query intent"""
        if not query or not results:
            return 0.0
        
        query_words = set(query.lower().split())
        alignment_scores = []
        
        for result in results:
            content = result.get('content', '').lower()
            content_words = set(content.split())
            
            if query_words and content_words:
                overlap = len(query_words.intersection(content_words))
                alignment = overlap / len(query_words)
                alignment_scores.append(alignment)
        
        return statistics.mean(alignment_scores) if alignment_scores else 0.0
    
    def _calculate_precision_recall(self, results: List[Dict[str, Any]], 
                                  expected: List[str]) -> Dict[str, float]:
        """Calculate precision and recall metrics"""
        result_doc_ids = set(r.get('document_id', '') for r in results)
        expected_doc_ids = set(expected)
        
        true_positives = len(result_doc_ids.intersection(expected_doc_ids))
        
        precision = true_positives / len(results) if results else 0
        recall = true_positives / len(expected) if expected else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'true_positives': true_positives
        }
    
    def _calculate_search_quality_score(self, metrics: Dict[str, float], 
                                      issues: List[ValidationIssue]) -> float:
        """Calculate overall search quality score"""
        base_score = 1.0
        
        # Deduct for issues
        for issue in issues:
            if issue.severity == ValidationSeverity.CRITICAL:
                base_score -= 0.4
            elif issue.severity == ValidationSeverity.HIGH:
                base_score -= 0.25
            elif issue.severity == ValidationSeverity.MEDIUM:
                base_score -= 0.15
            elif issue.severity == ValidationSeverity.LOW:
                base_score -= 0.05
        
        # Add for good metrics
        if metrics.get('diversity_score', 0) > 0.8:
            base_score += 0.1
        if metrics.get('avg_similarity', 0) > 0.7:
            base_score += 0.1
        if metrics.get('precision', 0) > 0.8:
            base_score += 0.1
        
        return max(min(base_score, 1.0), 0.0)


# Composite validator that runs all validations
class ComprehensiveValidator:
    """Runs all validation components and provides unified results"""
    
    def __init__(self):
        self.content_validator = DocumentContentValidator()
        self.chunk_validator = ChunkQualityValidator()
        self.embedding_validator = EmbeddingQualityValidator()
        self.search_validator = SearchQualityValidator()
    
    def validate_document_processing(self, document: Document, 
                                   embeddings: Optional[List[np.ndarray]] = None) -> Dict[str, ValidationResult]:
        """Run comprehensive validation on document processing"""
        results = {}
        
        # Content validation
        results['content'] = self.content_validator.validate(document)
        
        # Chunk validation
        results['chunking'] = self.chunk_validator.validate(document)
        
        # Embedding validation (if provided)
        if embeddings:
            results['embeddings'] = self.embedding_validator.validate(embeddings, document.chunks)
        
        return results
    
    def validate_search_operation(self, query: str, results: List[Dict[str, Any]], 
                                expected_results: Optional[List[str]] = None) -> ValidationResult:
        """Validate search operation quality"""
        return self.search_validator.validate(query, results, expected_results)
    
    def generate_comprehensive_report(self, validation_results: Dict[str, ValidationResult]) -> Dict[str, Any]:
        """Generate comprehensive validation report"""
        total_issues = []
        total_recommendations = set()
        component_scores = {}
        
        for component, result in validation_results.items():
            total_issues.extend(result.issues)
            total_recommendations.update(result.recommendations)
            component_scores[component] = result.overall_score
        
        # Calculate overall quality score
        overall_score = statistics.mean(component_scores.values()) if component_scores else 0.0
        
        # Categorize issues by severity
        critical_issues = len([i for i in total_issues if i.severity == ValidationSeverity.CRITICAL])
        high_issues = len([i for i in total_issues if i.severity == ValidationSeverity.HIGH])
        medium_issues = len([i for i in total_issues if i.severity == ValidationSeverity.MEDIUM])
        low_issues = len([i for i in total_issues if i.severity == ValidationSeverity.LOW])
        
        return {
            'overall_score': overall_score,
            'component_scores': component_scores,
            'total_issues': len(total_issues),
            'issues_by_severity': {
                'critical': critical_issues,
                'high': high_issues,
                'medium': medium_issues,
                'low': low_issues
            },
            'all_issues': [issue.to_dict() for issue in total_issues],
            'recommendations': list(total_recommendations),
            'validation_summary': {
                'components_validated': len(validation_results),
                'components_passed': len([r for r in validation_results.values() if r.is_valid]),
                'has_critical_issues': critical_issues > 0,
                'quality_level': self._determine_quality_level(overall_score)
            }
        }
    
    def _determine_quality_level(self, score: float) -> str:
        """Determine quality level from score"""
        if score >= 0.9:
            return 'excellent'
        elif score >= 0.7:
            return 'good'
        elif score >= 0.5:
            return 'acceptable'
        elif score >= 0.3:
            return 'poor'
        else:
            return 'critical'