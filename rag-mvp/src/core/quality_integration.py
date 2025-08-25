"""
Quality Framework Integration Module

This module provides seamless integration between the Quality Framework and existing
RAG system components (DocumentProcessor v3, VectorStore v3, etc.).
"""

import logging
import time
import numpy as np
from typing import Dict, List, Any, Optional, Callable, Union
from pathlib import Path
from datetime import datetime

from .quality_framework import (
    QualityManager, 
    ComponentType, 
    ValidationLevel,
    QualityLevel,
    QualityMetric,
    PerformanceMetrics
)
from .quality_validators import (
    ComprehensiveValidator,
    ValidationSeverity,
    ValidationResult
)
from .quality_monitoring import (
    QualityMonitoringService,
    MonitoringRule,
    AlertSeverity,
    LogAlertHandler,
    EmailAlertHandler
)
from .models import Document, Chunk


class EnhancedRAGQualitySystem:
    """
    Comprehensive quality management system that integrates with all RAG components
    
    This class provides a unified interface for quality assessment, monitoring,
    and optimization across the entire RAG pipeline.
    """
    
    def __init__(self, 
                 db_path: str = "enhanced_rag_quality.db",
                 validation_level: ValidationLevel = ValidationLevel.COMPREHENSIVE,
                 enable_monitoring: bool = True,
                 monitoring_interval: int = 30,
                 quality_thresholds: Optional[Dict[str, float]] = None):
        """
        Initialize Enhanced RAG Quality System
        
        Args:
            db_path: Path to quality database
            validation_level: Quality validation level
            enable_monitoring: Enable real-time monitoring
            monitoring_interval: Monitoring interval in seconds
            quality_thresholds: Custom quality thresholds
        """
        self.db_path = Path(db_path)
        self.validation_level = validation_level
        self.monitoring_interval = monitoring_interval
        
        # Set quality thresholds
        self.quality_thresholds = quality_thresholds or {
            'document_processing': 0.7,
            'chunking_quality': 0.6,
            'embedding_quality': 0.7,
            'search_relevance': 0.6,
            'correlation_accuracy': 0.5,
            'overall_pipeline': 0.65
        }
        
        # Initialize core components
        self.quality_manager = QualityManager(
            db_path=str(db_path),
            validation_level=validation_level,
            enable_real_time_monitoring=enable_monitoring
        )
        
        self.validator = ComprehensiveValidator()
        
        # Initialize monitoring service if enabled
        self.monitoring_service = None
        if enable_monitoring:
            self.monitoring_service = QualityMonitoringService(
                db_path=str(db_path),
                monitoring_interval=monitoring_interval
            )
            
            # Configure custom monitoring rules
            self._setup_monitoring_rules()
        
        # Component registries
        self.registered_components = {}
        self.component_wrappers = {}
        
        logging.info("Enhanced RAG Quality System initialized")
    
    def register_document_processor(self, document_processor):
        """Register DocumentProcessor for quality monitoring"""
        self.quality_manager.register_component(
            ComponentType.DOCUMENT_PROCESSOR, 
            document_processor
        )
        
        self.registered_components['document_processor'] = document_processor
        
        # Create enhanced wrapper
        enhanced_processor = EnhancedDocumentProcessor(
            document_processor, 
            self.quality_manager,
            self.quality_thresholds
        )
        
        self.component_wrappers['document_processor'] = enhanced_processor
        
        logging.info("DocumentProcessor registered and enhanced with quality monitoring")
        return enhanced_processor
    
    def register_vector_store(self, vector_store):
        """Register VectorStore for quality monitoring"""
        self.quality_manager.register_component(
            ComponentType.VECTOR_STORE, 
            vector_store
        )
        
        self.registered_components['vector_store'] = vector_store
        
        # Create enhanced wrapper
        enhanced_store = EnhancedVectorStore(
            vector_store,
            self.quality_manager,
            self.quality_thresholds
        )
        
        self.component_wrappers['vector_store'] = enhanced_store
        
        logging.info("VectorStore registered and enhanced with quality monitoring")
        return enhanced_store
    
    def register_embedding_service(self, embedding_service):
        """Register EmbeddingService for quality monitoring"""
        self.quality_manager.register_component(
            ComponentType.EMBEDDING_SERVICE,
            embedding_service
        )
        
        self.registered_components['embedding_service'] = embedding_service
        
        logging.info("EmbeddingService registered for quality monitoring")
        return embedding_service
    
    def register_component(self, component_type: ComponentType, component):
        """Register any custom component for quality monitoring"""
        self.quality_manager.register_component(component_type, component)
        self.registered_components[component_type.value] = component
        
        logging.info(f"Custom component {component_type.value} registered")
        return component
    
    def process_document_with_quality_assessment(self, 
                                               file_path: str,
                                               filename: str,
                                               embeddings: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Process document with comprehensive quality assessment
        
        Args:
            file_path: Path to document file
            filename: Document filename
            embeddings: Optional pre-computed embeddings
            
        Returns:
            Processing result with quality assessment
        """
        start_time = time.time()
        
        try:
            # Get document processor
            doc_processor = self.registered_components.get('document_processor')
            if not doc_processor:
                raise ValueError("DocumentProcessor not registered")
            
            # Process document
            document = doc_processor.process_file(file_path, filename)
            
            # Quality assessment
            quality_report = self.quality_manager.assess_document_quality(
                document, embeddings
            )
            
            # Determine if quality meets standards
            meets_standards = quality_report.overall_score >= self.quality_thresholds['document_processing']
            
            processing_time = time.time() - start_time
            
            return {
                'document': document,
                'quality_score': quality_report.overall_score,
                'quality_level': self._determine_quality_level(quality_report.overall_score),
                'meets_quality_standards': meets_standards,
                'quality_report': quality_report,
                'quality_recommendations': quality_report.recommendations,
                'alerts': quality_report.alerts,
                'processing_time': processing_time,
                'component_scores': quality_report.component_scores,
                'validation_results': quality_report.validation_results
            }
            
        except Exception as e:
            logging.error(f"Error in document processing with quality assessment: {e}")
            return {
                'error': str(e),
                'quality_score': 0.0,
                'quality_level': 'critical',
                'meets_quality_standards': False
            }
    
    def search_with_quality_assessment(self,
                                     query: str,
                                     query_embedding: Optional[np.ndarray] = None,
                                     top_k: int = 5,
                                     expected_results: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Perform search with quality assessment
        
        Args:
            query: Search query
            query_embedding: Optional query embedding
            top_k: Number of results to return
            expected_results: Optional expected result IDs for validation
            
        Returns:
            Search results with quality assessment
        """
        start_time = time.time()
        
        try:
            # Get vector store
            vector_store = self.registered_components.get('vector_store')
            if not vector_store:
                raise ValueError("VectorStore not registered")
            
            # Perform search
            if hasattr(vector_store, 'search'):
                results = vector_store.search(query, top_k)
            else:
                # Fallback for different vector store interfaces
                results = []
            
            # Quality assessment
            quality_report = self.quality_manager.assess_search_quality(
                query, results, query_embedding, expected_results
            )
            
            # Enhance results with quality metadata
            enhanced_results = []
            for result in results:
                enhanced_result = result.copy()
                enhanced_result['quality_metadata'] = {
                    'confidence_adjusted_similarity': result.get('similarity', 0) * quality_report.overall_score,
                    'quality_score': quality_report.overall_score,
                    'meets_quality_threshold': quality_report.overall_score >= self.quality_thresholds['search_relevance']
                }
                enhanced_results.append(enhanced_result)
            
            search_time = time.time() - start_time
            
            return {
                'results': enhanced_results,
                'result_count': len(results),
                'quality_score': quality_report.overall_score,
                'quality_report': quality_report,
                'search_time': search_time,
                'meets_quality_standards': quality_report.overall_score >= self.quality_thresholds['search_relevance'],
                'recommendations': quality_report.recommendations,
                'alerts': quality_report.alerts
            }
            
        except Exception as e:
            logging.error(f"Error in search with quality assessment: {e}")
            return {
                'results': [],
                'result_count': 0,
                'quality_score': 0.0,
                'error': str(e),
                'meets_quality_standards': False
            }
    
    def assess_pipeline_health(self, include_trends: bool = True) -> Dict[str, Any]:
        """
        Assess overall pipeline health
        
        Args:
            include_trends: Include trend analysis
            
        Returns:
            Pipeline health assessment
        """
        try:
            # Get pipeline quality report
            pipeline_report = self.quality_manager.assess_pipeline_quality(include_trends)
            
            # Determine system status
            overall_score = pipeline_report.overall_score
            
            if overall_score >= 0.8:
                system_status = "excellent"
            elif overall_score >= 0.6:
                system_status = "good"
            elif overall_score >= 0.4:
                system_status = "acceptable"
            elif overall_score >= 0.2:
                system_status = "poor"
            else:
                system_status = "critical"
            
            # Risk assessment
            high_risk_areas = []
            medium_risk_areas = []
            
            for component, score in pipeline_report.component_scores.items():
                if score < 0.5:
                    high_risk_areas.append(f"{component}: {score:.3f}")
                elif score < 0.7:
                    medium_risk_areas.append(f"{component}: {score:.3f}")
            
            return {
                'system_status': system_status,
                'overall_health_score': overall_score,
                'component_scores': pipeline_report.component_scores,
                'performance_metrics': pipeline_report.performance_metrics,
                'alerts': pipeline_report.alerts,
                'recommendations': pipeline_report.recommendations,
                'risk_assessment': {
                    'high_risk_areas': high_risk_areas,
                    'medium_risk_areas': medium_risk_areas,
                    'overall_risk_level': 'high' if high_risk_areas else 'medium' if medium_risk_areas else 'low'
                },
                'trend_analysis': pipeline_report.trend_analysis,
                'validation_results': pipeline_report.validation_results,
                'components_registered': len(self.registered_components),
                'monitoring_active': self.monitoring_service is not None and self.monitoring_service.monitoring_active if self.monitoring_service else False
            }
            
        except Exception as e:
            logging.error(f"Error assessing pipeline health: {e}")
            return {
                'system_status': 'error',
                'overall_health_score': 0.0,
                'error': str(e)
            }
    
    def optimize_quality_settings(self, 
                                target_quality: float = 0.8,
                                focus_areas: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Optimize quality settings to achieve target quality
        
        Args:
            target_quality: Target overall quality score
            focus_areas: Specific areas to focus optimization on
            
        Returns:
            Optimization recommendations
        """
        try:
            # Get current pipeline assessment
            current_health = self.assess_pipeline_health(include_trends=False)
            current_score = current_health['overall_health_score']
            
            recommendations = []
            adjustments = {}
            
            # If already meeting target, suggest maintaining current settings
            if current_score >= target_quality:
                recommendations.append("Current quality exceeds target - maintain current settings")
                return {
                    'status': 'target_achieved',
                    'current_score': current_score,
                    'target_score': target_quality,
                    'recommendations': recommendations,
                    'adjustments': adjustments
                }
            
            # Identify areas for improvement
            component_scores = current_health.get('component_scores', {})
            
            for component, score in component_scores.items():
                if score < target_quality:
                    gap = target_quality - score
                    
                    if component == 'document_processor':
                        recommendations.append(f"Improve document processing: increase validation strictness")
                        adjustments['document_processing_threshold'] = min(0.8, score + gap * 0.8)
                    
                    elif component == 'vector_store':
                        recommendations.append(f"Optimize vector storage: review embedding quality")
                        adjustments['embedding_quality_threshold'] = min(0.8, score + gap * 0.8)
                    
                    elif 'search' in component:
                        recommendations.append(f"Enhance search quality: tune similarity thresholds")
                        adjustments['search_relevance_threshold'] = min(0.8, score + gap * 0.8)
            
            # Focus area specific recommendations
            if focus_areas:
                for area in focus_areas:
                    if area == 'performance':
                        recommendations.append("Enable performance monitoring with tighter SLAs")
                    elif area == 'accuracy':
                        recommendations.append("Increase validation level to ENTERPRISE")
                    elif area == 'monitoring':
                        recommendations.append("Reduce monitoring interval to 15 seconds")
            
            return {
                'status': 'optimization_needed',
                'current_score': current_score,
                'target_score': target_quality,
                'quality_gap': target_quality - current_score,
                'recommendations': recommendations,
                'adjustments': adjustments,
                'estimated_improvement': min(0.3, target_quality - current_score)
            }
            
        except Exception as e:
            logging.error(f"Error optimizing quality settings: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def generate_quality_dashboard_data(self) -> Dict[str, Any]:
        """
        Generate data for quality dashboard display
        
        Returns:
            Dashboard data structure
        """
        try:
            # Get current system health
            health_data = self.assess_pipeline_health(include_trends=True)
            
            # Get recent alerts from monitoring service
            active_alerts = []
            if self.monitoring_service:
                recent_alerts = self.monitoring_service.get_active_alerts()
                active_alerts = [
                    {
                        'severity': alert.severity.value,
                        'message': alert.message,
                        'component': alert.component.value,
                        'timestamp': alert.timestamp.isoformat()
                    }
                    for alert in recent_alerts[-10:]  # Last 10 alerts
                ]
            
            # Performance summary
            performance_metrics = health_data.get('performance_metrics', {})
            
            # Quality trends (simplified)
            quality_trends = {
                'current_period': health_data['overall_health_score'],
                'previous_period': health_data['overall_health_score'] * 0.95,  # Simulated
                'trend_direction': 'stable',
                'change_percentage': 5.0
            }
            
            return {
                'system_overview': {
                    'status': health_data['system_status'],
                    'overall_score': health_data['overall_health_score'],
                    'components_registered': health_data['components_registered'],
                    'monitoring_active': health_data['monitoring_active']
                },
                'component_health': health_data['component_scores'],
                'performance_metrics': {
                    'throughput': performance_metrics.get('processing_throughput', 0),
                    'latency': performance_metrics.get('search_latency', 0),
                    'error_rate': performance_metrics.get('error_rate', 0),
                    'availability': performance_metrics.get('availability', 1.0)
                },
                'active_alerts': active_alerts,
                'quality_trends': quality_trends,
                'recommendations': health_data.get('recommendations', []),
                'risk_assessment': health_data.get('risk_assessment', {}),
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logging.error(f"Error generating dashboard data: {e}")
            return {
                'system_overview': {
                    'status': 'error',
                    'overall_score': 0.0,
                    'error': str(e)
                }
            }
    
    def start_monitoring(self):
        """Start quality monitoring service"""
        if self.monitoring_service:
            self.monitoring_service.start_monitoring()
            logging.info("Quality monitoring started")
        else:
            logging.warning("Monitoring service not initialized")
    
    def stop_monitoring(self):
        """Stop quality monitoring service"""
        if self.monitoring_service:
            self.monitoring_service.stop_monitoring()
            logging.info("Quality monitoring stopped")
    
    def add_custom_monitoring_rule(self, rule: MonitoringRule):
        """Add custom monitoring rule"""
        if self.monitoring_service:
            self.monitoring_service.add_monitoring_rule(rule)
            logging.info(f"Added custom monitoring rule: {rule.rule_id}")
        else:
            logging.warning("Monitoring service not available for custom rules")
    
    def add_email_alerts(self, email_config: Dict[str, Any]):
        """Configure email alerts"""
        if self.monitoring_service:
            self.monitoring_service.add_alert_handler(
                EmailAlertHandler(email_config)
            )
            logging.info("Email alerts configured")
        else:
            logging.warning("Monitoring service not available for email alerts")
    
    def _setup_monitoring_rules(self):
        """Setup default monitoring rules for the integrated system"""
        if not self.monitoring_service:
            return
        
        # Document processing quality rules
        self.monitoring_service.add_monitoring_rule(
            MonitoringRule(
                rule_id="document_quality_critical",
                component=ComponentType.DOCUMENT_PROCESSOR,
                metric_name="document_processing",
                threshold_value=0.3,
                comparison_operator="<",
                alert_severity=AlertSeverity.CRITICAL,
                consecutive_violations=2
            )
        )
        
        # Search quality rules
        self.monitoring_service.add_monitoring_rule(
            MonitoringRule(
                rule_id="search_quality_poor",
                component=ComponentType.VECTOR_STORE,
                metric_name="search_relevance",
                threshold_value=0.4,
                comparison_operator="<",
                alert_severity=AlertSeverity.HIGH,
                consecutive_violations=3
            )
        )
        
        logging.info("Default monitoring rules configured")
    
    def _determine_quality_level(self, score: float) -> str:
        """Determine quality level from score"""
        if score >= 0.9:
            return "excellent"
        elif score >= 0.7:
            return "good"
        elif score >= 0.5:
            return "acceptable"
        elif score >= 0.3:
            return "poor"
        else:
            return "critical"
    
    def close(self):
        """Close quality system and cleanup resources"""
        try:
            # Stop monitoring
            self.stop_monitoring()
            
            # Close monitoring service
            if self.monitoring_service:
                self.monitoring_service.close()
            
            # Close quality manager
            self.quality_manager.close()
            
            logging.info("Enhanced RAG Quality System closed successfully")
            
        except Exception as e:
            logging.error(f"Error closing quality system: {e}")


class EnhancedDocumentProcessor:
    """Enhanced DocumentProcessor with quality monitoring"""
    
    def __init__(self, document_processor, quality_manager, quality_thresholds):
        self.processor = document_processor
        self.quality_manager = quality_manager
        self.quality_thresholds = quality_thresholds
    
    def process_file(self, file_path: str, filename: str = None) -> Document:
        """Process file with quality assessment"""
        # Delegate to original processor
        document = self.processor.process_file(file_path, filename)
        
        # Quality assessment
        quality_report = self.quality_manager.assess_document_quality(document)
        
        # Add quality metadata to document
        if hasattr(document, 'metadata'):
            document.metadata.update({
                'quality_score': quality_report.overall_score,
                'quality_level': 'excellent' if quality_report.overall_score >= 0.9 else 'good' if quality_report.overall_score >= 0.7 else 'acceptable',
                'quality_alerts': quality_report.alerts,
                'quality_recommendations': quality_report.recommendations
            })
        
        return document
    
    def __getattr__(self, name):
        """Delegate other attributes to original processor"""
        return getattr(self.processor, name)


class EnhancedVectorStore:
    """Enhanced VectorStore with quality monitoring"""
    
    def __init__(self, vector_store, quality_manager, quality_thresholds):
        self.store = vector_store
        self.quality_manager = quality_manager
        self.quality_thresholds = quality_thresholds
    
    def search(self, query: str, top_k: int = 5, **kwargs):
        """Search with quality assessment"""
        # Delegate to original store
        if hasattr(self.store, 'search'):
            results = self.store.search(query, top_k, **kwargs)
        else:
            results = []
        
        # Quality assessment
        quality_report = self.quality_manager.assess_search_quality(query, results)
        
        # Enhance results with quality scores
        enhanced_results = []
        for result in results:
            enhanced_result = result.copy() if isinstance(result, dict) else {'content': str(result)}
            enhanced_result['quality_score'] = quality_report.overall_score
            enhanced_result['meets_quality_threshold'] = quality_report.overall_score >= self.quality_thresholds.get('search_relevance', 0.6)
            enhanced_results.append(enhanced_result)
        
        return enhanced_results
    
    def __getattr__(self, name):
        """Delegate other attributes to original store"""
        return getattr(self.store, name)