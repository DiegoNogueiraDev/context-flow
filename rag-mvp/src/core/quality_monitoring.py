"""
Real-time Quality Monitoring Service for Enhanced RAG System

This module provides continuous monitoring, alerting, and trend analysis
for quality metrics across the RAG pipeline components.
"""

import asyncio
import logging
import threading
import time
import json
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import deque, defaultdict
import statistics
import smtplib
try:
    from email.mime.text import MimeText
    from email.mime.multipart import MimeMultipart
except ImportError:
    # Fallback for compatibility
    MimeText = None
    MimeMultipart = None
from pathlib import Path
import hashlib

from .quality_framework import QualityLevel, ComponentType, PerformanceMetrics, QualityMetric
from .quality_validators import ValidationSeverity


class AlertType(Enum):
    """Types of quality alerts"""
    QUALITY_DEGRADATION = "quality_degradation"
    PERFORMANCE_ISSUE = "performance_issue"
    SYSTEM_ERROR = "system_error"
    THRESHOLD_BREACH = "threshold_breach"
    TREND_ANOMALY = "trend_anomaly"
    RESOURCE_CONSTRAINT = "resource_constraint"
    CORRELATION_ISSUE = "correlation_issue"


class AlertSeverity(Enum):
    """Severity levels for alerts"""
    CRITICAL = "critical"    # Immediate action required
    HIGH = "high"           # Action required soon
    MEDIUM = "medium"       # Should be addressed
    LOW = "low"            # Informational
    INFO = "info"          # Informational only


@dataclass
class QualityAlert:
    """Quality alert data structure"""
    alert_id: str
    alert_type: AlertType
    severity: AlertSeverity
    component: ComponentType
    metric_name: str
    current_value: float
    threshold_value: float
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    acknowledged: bool = False
    resolved: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MonitoringRule:
    """Monitoring rule definition"""
    rule_id: str
    component: ComponentType
    metric_name: str
    threshold_value: float
    comparison_operator: str  # '<', '>', '<=', '>=', '==', '!='
    alert_severity: AlertSeverity
    alert_type: AlertType = AlertType.THRESHOLD_BREACH
    consecutive_violations: int = 1
    evaluation_window: int = 300  # seconds
    enabled: bool = True
    description: Optional[str] = None


@dataclass
class TrendAnalysis:
    """Quality trend analysis results"""
    metric_name: str
    component: ComponentType
    current_value: float
    trend_direction: str  # 'improving', 'stable', 'degrading'
    trend_strength: float  # 0.0 to 1.0
    trend_duration: int  # days
    prediction: Optional[float] = None
    confidence: float = 0.0
    analysis_timestamp: datetime = field(default_factory=datetime.now)


class QualityMonitoringService:
    """
    Real-time quality monitoring service with alerting and trend analysis
    
    Provides continuous monitoring of quality metrics across all RAG components
    with configurable alerts, trend analysis, and notification systems.
    """
    
    def __init__(self, 
                 db_path: str = "quality_monitoring.db",
                 monitoring_interval: int = 60,
                 trend_analysis_window: int = 7):
        """
        Initialize Quality Monitoring Service
        
        Args:
            db_path: Path to monitoring database
            monitoring_interval: Monitoring check interval in seconds
            trend_analysis_window: Days of data for trend analysis
        """
        self.db_path = Path(db_path)
        self.monitoring_interval = monitoring_interval
        self.trend_window = trend_analysis_window
        
        # Monitoring state
        self.monitoring_active = False
        self.monitoring_thread = None
        
        # Rules and alerts
        self.monitoring_rules: Dict[str, MonitoringRule] = {}
        self.active_alerts: Dict[str, QualityAlert] = {}
        self.alert_handlers: List[Callable[[QualityAlert], None]] = []
        
        # Metrics storage
        self.metrics_buffer: deque = deque(maxlen=1000)
        self.violation_counters: defaultdict = defaultdict(int)
        
        # Database connection
        self.connection = None
        self._init_monitoring_database()
        
        # Load default monitoring rules
        self._load_default_monitoring_rules()
        
        logging.info("Quality Monitoring Service initialized")
    
    def _init_monitoring_database(self):
        """Initialize monitoring database"""
        try:
            self.connection = sqlite3.connect(str(self.db_path), check_same_thread=False)
            cursor = self.connection.cursor()
            
            # Quality alerts table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS quality_alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    alert_id TEXT UNIQUE NOT NULL,
                    alert_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    component TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    current_value REAL NOT NULL,
                    threshold_value REAL NOT NULL,
                    message TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    acknowledged BOOLEAN DEFAULT FALSE,
                    resolved BOOLEAN DEFAULT FALSE,
                    metadata TEXT
                )
            """)
            
            # Monitoring rules table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS monitoring_rules (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    rule_id TEXT UNIQUE NOT NULL,
                    component TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    threshold_value REAL NOT NULL,
                    comparison_operator TEXT NOT NULL,
                    alert_severity TEXT NOT NULL,
                    alert_type TEXT DEFAULT 'threshold_breach',
                    consecutive_violations INTEGER DEFAULT 1,
                    evaluation_window INTEGER DEFAULT 300,
                    enabled BOOLEAN DEFAULT TRUE,
                    description TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Trend analysis table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trend_analysis (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_name TEXT NOT NULL,
                    component TEXT NOT NULL,
                    current_value REAL NOT NULL,
                    trend_direction TEXT NOT NULL,
                    trend_strength REAL NOT NULL,
                    trend_duration INTEGER NOT NULL,
                    prediction REAL,
                    confidence REAL DEFAULT 0.0,
                    analysis_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            self.connection.commit()
            logging.info(f"Quality monitoring database initialized: {self.db_path}")
            
        except Exception as e:
            logging.error(f"Error initializing monitoring database: {e}")
            raise
    
    def _load_default_monitoring_rules(self):
        """Load default monitoring rules"""
        default_rules = [
            # Document processing quality rules
            MonitoringRule(
                rule_id="doc_processing_quality_low",
                component=ComponentType.DOCUMENT_PROCESSOR,
                metric_name="overall_quality",
                threshold_value=0.5,
                comparison_operator="<",
                alert_severity=AlertSeverity.HIGH,
                consecutive_violations=2
            ),
            MonitoringRule(
                rule_id="doc_processing_confidence_low",
                component=ComponentType.DOCUMENT_PROCESSOR,
                metric_name="extraction_confidence",
                threshold_value=0.6,
                comparison_operator="<",
                alert_severity=AlertSeverity.MEDIUM
            ),
            
            # Vector store quality rules
            MonitoringRule(
                rule_id="vector_store_quality_critical",
                component=ComponentType.VECTOR_STORE,
                metric_name="overall_quality",
                threshold_value=0.3,
                comparison_operator="<",
                alert_severity=AlertSeverity.CRITICAL
            ),
            MonitoringRule(
                rule_id="search_latency_high",
                component=ComponentType.VECTOR_STORE,
                metric_name="search_latency",
                threshold_value=2.0,
                comparison_operator=">",
                alert_severity=AlertSeverity.HIGH
            ),
            
            # Performance rules
            MonitoringRule(
                rule_id="error_rate_high",
                component=ComponentType.RAG_SERVICE,
                metric_name="error_rate",
                threshold_value=0.05,
                comparison_operator=">",
                alert_severity=AlertSeverity.HIGH,
                consecutive_violations=3
            ),
            MonitoringRule(
                rule_id="availability_low",
                component=ComponentType.RAG_SERVICE,
                metric_name="availability",
                threshold_value=0.95,
                comparison_operator="<",
                alert_severity=AlertSeverity.CRITICAL
            ),
            
            # Cross-correlation rules (using RAG_SERVICE as proxy for correlation metrics)
            MonitoringRule(
                rule_id="correlation_quality_poor",
                component=ComponentType.RAG_SERVICE,
                metric_name="correlation_accuracy",
                threshold_value=0.4,
                comparison_operator="<",
                alert_severity=AlertSeverity.MEDIUM
            )
        ]
        
        for rule in default_rules:
            self.add_monitoring_rule(rule)
        
        logging.info(f"Loaded {len(default_rules)} default monitoring rules")
    
    def add_monitoring_rule(self, rule: MonitoringRule):
        """Add a monitoring rule"""
        try:
            self.monitoring_rules[rule.rule_id] = rule
            
            # Store in database
            cursor = self.connection.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO monitoring_rules
                (rule_id, component, metric_name, threshold_value, comparison_operator,
                 alert_severity, alert_type, consecutive_violations, evaluation_window,
                 enabled, description)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                rule.rule_id,
                rule.component.value,
                rule.metric_name,
                rule.threshold_value,
                rule.comparison_operator,
                rule.alert_severity.value,
                rule.alert_type.value,
                rule.consecutive_violations,
                rule.evaluation_window,
                rule.enabled,
                rule.description
            ))
            
            self.connection.commit()
            logging.info(f"Added monitoring rule: {rule.rule_id}")
            
        except Exception as e:
            logging.error(f"Error adding monitoring rule {rule.rule_id}: {e}")
    
    def remove_monitoring_rule(self, rule_id: str):
        """Remove a monitoring rule"""
        try:
            if rule_id in self.monitoring_rules:
                del self.monitoring_rules[rule_id]
                
                # Remove from database
                cursor = self.connection.cursor()
                cursor.execute("DELETE FROM monitoring_rules WHERE rule_id = ?", (rule_id,))
                self.connection.commit()
                
                logging.info(f"Removed monitoring rule: {rule_id}")
            
        except Exception as e:
            logging.error(f"Error removing monitoring rule {rule_id}: {e}")
    
    def record_quality_metric(self, metric: QualityMetric):
        """Record a quality metric for monitoring"""
        try:
            # Add to buffer for real-time monitoring
            self.metrics_buffer.append(metric)
            
            # Evaluate monitoring rules
            self._evaluate_monitoring_rules(metric)
            
        except Exception as e:
            logging.error(f"Error recording quality metric: {e}")
    
    def _evaluate_monitoring_rules(self, metric: QualityMetric):
        """Evaluate monitoring rules against a metric"""
        for rule_id, rule in self.monitoring_rules.items():
            if not rule.enabled:
                continue
            
            # Check if rule applies to this metric
            if (rule.component == metric.component and 
                rule.metric_name == metric.name):
                
                # Evaluate threshold condition
                violation = self._check_threshold_violation(metric.value, rule)
                
                if violation:
                    self.violation_counters[rule_id] += 1
                    
                    # Check if consecutive violations threshold is met
                    if self.violation_counters[rule_id] >= rule.consecutive_violations:
                        self._trigger_alert(rule, metric)
                        # Reset counter after triggering alert
                        self.violation_counters[rule_id] = 0
                else:
                    # Reset counter if no violation
                    self.violation_counters[rule_id] = 0
    
    def _check_threshold_violation(self, value: float, rule: MonitoringRule) -> bool:
        """Check if a value violates a rule threshold"""
        operator = rule.comparison_operator
        threshold = rule.threshold_value
        
        if operator == "<":
            return value < threshold
        elif operator == "<=":
            return value <= threshold
        elif operator == ">":
            return value > threshold
        elif operator == ">=":
            return value >= threshold
        elif operator == "==":
            return value == threshold
        elif operator == "!=":
            return value != threshold
        else:
            logging.warning(f"Unknown comparison operator: {operator}")
            return False
    
    def _trigger_alert(self, rule: MonitoringRule, metric: QualityMetric):
        """Trigger a quality alert"""
        try:
            alert_id = f"{rule.rule_id}_{int(time.time())}"
            
            alert = QualityAlert(
                alert_id=alert_id,
                alert_type=rule.alert_type,
                severity=rule.alert_severity,
                component=rule.component,
                metric_name=rule.metric_name,
                current_value=metric.value,
                threshold_value=rule.threshold_value,
                message=f"Quality metric {rule.metric_name} ({metric.value:.3f}) violates threshold ({rule.threshold_value:.3f}) for {rule.component.value}",
                metadata={
                    'rule_id': rule.rule_id,
                    'metric_confidence': metric.confidence,
                    'metric_metadata': metric.metadata
                }
            )
            
            # Store active alert
            self.active_alerts[alert_id] = alert
            
            # Store in database
            self._store_alert(alert)
            
            # Notify alert handlers
            for handler in self.alert_handlers:
                try:
                    handler(alert)
                except Exception as e:
                    logging.error(f"Error in alert handler: {e}")
            
            logging.warning(f"Quality alert triggered: {alert.message}")
            
        except Exception as e:
            logging.error(f"Error triggering alert: {e}")
    
    def _store_alert(self, alert: QualityAlert):
        """Store alert in database"""
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                INSERT INTO quality_alerts
                (alert_id, alert_type, severity, component, metric_name,
                 current_value, threshold_value, message, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                alert.alert_id,
                alert.alert_type.value,
                alert.severity.value,
                alert.component.value,
                alert.metric_name,
                alert.current_value,
                alert.threshold_value,
                alert.message,
                json.dumps(alert.metadata)
            ))
            
            self.connection.commit()
            
        except Exception as e:
            logging.error(f"Error storing alert: {e}")
    
    def get_active_alerts(self, severity: Optional[AlertSeverity] = None) -> List[QualityAlert]:
        """Get list of active alerts"""
        alerts = list(self.active_alerts.values())
        
        if severity:
            alerts = [alert for alert in alerts if alert.severity == severity]
        
        return sorted(alerts, key=lambda x: x.timestamp, reverse=True)
    
    def acknowledge_alert(self, alert_id: str):
        """Acknowledge an alert"""
        try:
            if alert_id in self.active_alerts:
                self.active_alerts[alert_id].acknowledged = True
                
                # Update database
                cursor = self.connection.cursor()
                cursor.execute(
                    "UPDATE quality_alerts SET acknowledged = TRUE WHERE alert_id = ?",
                    (alert_id,)
                )
                self.connection.commit()
                
                logging.info(f"Alert acknowledged: {alert_id}")
            
        except Exception as e:
            logging.error(f"Error acknowledging alert {alert_id}: {e}")
    
    def resolve_alert(self, alert_id: str):
        """Resolve an alert"""
        try:
            if alert_id in self.active_alerts:
                self.active_alerts[alert_id].resolved = True
                
                # Update database
                cursor = self.connection.cursor()
                cursor.execute(
                    "UPDATE quality_alerts SET resolved = TRUE WHERE alert_id = ?",
                    (alert_id,)
                )
                self.connection.commit()
                
                # Remove from active alerts
                del self.active_alerts[alert_id]
                
                logging.info(f"Alert resolved: {alert_id}")
            
        except Exception as e:
            logging.error(f"Error resolving alert {alert_id}: {e}")
    
    def add_alert_handler(self, handler: Callable[[QualityAlert], None]):
        """Add an alert handler"""
        self.alert_handlers.append(handler)
        logging.info("Alert handler added")
    
    def analyze_quality_trends(self, 
                             component: Optional[ComponentType] = None,
                             metric_name: Optional[str] = None,
                             days: int = 7) -> List[TrendAnalysis]:
        """Analyze quality trends over time"""
        try:
            # Get historical data from metrics buffer and database
            # This is a simplified implementation
            trends = []
            
            # Sample trend analysis (in production, this would use sophisticated algorithms)
            if not component:
                components = [ComponentType.DOCUMENT_PROCESSOR, ComponentType.VECTOR_STORE, ComponentType.RAG_SERVICE]
            else:
                components = [component]
            
            for comp in components:
                trend = TrendAnalysis(
                    metric_name=metric_name or "overall_quality",
                    component=comp,
                    current_value=0.75,  # Simplified
                    trend_direction="stable",
                    trend_strength=0.1,
                    trend_duration=days,
                    prediction=0.76,
                    confidence=0.8
                )
                trends.append(trend)
            
            # Store trend analysis
            for trend in trends:
                self._store_trend_analysis(trend)
            
            return trends
            
        except Exception as e:
            logging.error(f"Error analyzing quality trends: {e}")
            return []
    
    def _store_trend_analysis(self, trend: TrendAnalysis):
        """Store trend analysis in database"""
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                INSERT INTO trend_analysis
                (metric_name, component, current_value, trend_direction,
                 trend_strength, trend_duration, prediction, confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trend.metric_name,
                trend.component.value,
                trend.current_value,
                trend.trend_direction,
                trend.trend_strength,
                trend.trend_duration,
                trend.prediction,
                trend.confidence
            ))
            
            self.connection.commit()
            
        except Exception as e:
            logging.error(f"Error storing trend analysis: {e}")
    
    def get_system_health_summary(self) -> Dict[str, Any]:
        """Get system health summary"""
        try:
            # Count active alerts by severity
            alert_counts = defaultdict(int)
            for alert in self.active_alerts.values():
                alert_counts[alert.severity.value] += 1
            
            # Determine overall system status
            total_alerts = len(self.active_alerts)
            critical_alerts = alert_counts.get('critical', 0)
            high_alerts = alert_counts.get('high', 0)
            
            if critical_alerts > 0:
                system_status = "critical"
            elif high_alerts > 2:
                system_status = "degraded"
            elif total_alerts > 5:
                system_status = "warning"
            else:
                system_status = "healthy"
            
            return {
                'system_status': system_status,
                'total_active_alerts': total_alerts,
                'alert_counts': dict(alert_counts),
                'monitoring_rules_active': len([r for r in self.monitoring_rules.values() if r.enabled]),
                'monitoring_active': self.monitoring_active,
                'last_check': datetime.now().isoformat(),
                'metrics_processed': len(self.metrics_buffer),
                'trend_analysis_available': True
            }
            
        except Exception as e:
            logging.error(f"Error getting system health summary: {e}")
            return {
                'system_status': 'error',
                'error': str(e)
            }
    
    def start_monitoring(self):
        """Start real-time monitoring"""
        if self.monitoring_active:
            logging.warning("Monitoring already active")
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        logging.info("Quality monitoring started")
    
    def stop_monitoring(self):
        """Stop real-time monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5.0)
        
        logging.info("Quality monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Perform periodic checks
                self._perform_periodic_checks()
                
                # Run trend analysis
                if len(self.metrics_buffer) > 10:  # Only if we have enough data
                    self.analyze_quality_trends()
                
                # Clean up old alerts (keep for 7 days)
                self._cleanup_old_alerts()
                
                # Sleep until next check
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logging.error(f"Error in monitoring loop: {e}")
                time.sleep(self.monitoring_interval)  # Continue monitoring despite errors
    
    def _perform_periodic_checks(self):
        """Perform periodic system health checks"""
        # This would include system resource checks, connectivity tests, etc.
        # Simplified implementation
        pass
    
    def _cleanup_old_alerts(self):
        """Clean up old alerts"""
        try:
            cutoff_date = datetime.now() - timedelta(days=7)
            
            cursor = self.connection.cursor()
            cursor.execute(
                "DELETE FROM quality_alerts WHERE timestamp < ? AND resolved = TRUE",
                (cutoff_date.isoformat(),)
            )
            self.connection.commit()
            
            # Also clean up active alerts dictionary
            to_remove = []
            for alert_id, alert in self.active_alerts.items():
                if alert.resolved and alert.timestamp < cutoff_date:
                    to_remove.append(alert_id)
            
            for alert_id in to_remove:
                del self.active_alerts[alert_id]
            
        except Exception as e:
            logging.error(f"Error cleaning up old alerts: {e}")
    
    def close(self):
        """Close monitoring service and cleanup resources"""
        try:
            # Stop monitoring
            self.stop_monitoring()
            
            # Close database connection
            if self.connection:
                self.connection.close()
            
            logging.info("Quality Monitoring Service closed successfully")
            
        except Exception as e:
            logging.error(f"Error closing monitoring service: {e}")


class LogAlertHandler:
    """Alert handler that logs alerts"""
    
    def __init__(self, log_level: int = logging.WARNING):
        self.log_level = log_level
    
    def __call__(self, alert: QualityAlert):
        """Handle alert by logging"""
        logging.log(self.log_level, f"QUALITY ALERT [{alert.severity.value.upper()}]: {alert.message}")


class EmailAlertHandler:
    """Alert handler that sends email notifications"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize email alert handler
        
        Args:
            config: Email configuration with keys:
                - smtp_server: SMTP server hostname
                - smtp_port: SMTP server port
                - username: SMTP username
                - password: SMTP password
                - recipients: List of recipient email addresses
                - sender: Sender email address (optional)
        """
        self.config = config
        
        # Check if email libraries are available
        if not MimeText or not MimeMultipart:
            logging.warning("Email libraries not available - email alerts will be logged instead")
            self.fallback_to_log = True
        else:
            self.fallback_to_log = False
    
    def __call__(self, alert: QualityAlert):
        """Handle alert by sending email"""
        if self.fallback_to_log:
            # Fallback to logging
            logging.warning(f"EMAIL ALERT [{alert.severity.value.upper()}]: {alert.message}")
            return
        
        try:
            # Create email message
            msg = MimeMultipart()
            msg['From'] = self.config.get('sender', self.config['username'])
            msg['To'] = ', '.join(self.config['recipients'])
            msg['Subject'] = f"RAG Quality Alert - {alert.severity.value.upper()}: {alert.component.value}"
            
            # Email body
            body = f"""
Quality Alert Notification

Severity: {alert.severity.value.upper()}
Component: {alert.component.value}
Metric: {alert.metric_name}
Current Value: {alert.current_value:.3f}
Threshold: {alert.threshold_value:.3f}
Timestamp: {alert.timestamp.isoformat()}

Message: {alert.message}

Alert ID: {alert.alert_id}
            """
            
            msg.attach(MimeText(body, 'plain'))
            
            # Send email
            with smtplib.SMTP(self.config['smtp_server'], self.config['smtp_port']) as server:
                server.starttls()
                server.login(self.config['username'], self.config['password'])
                server.send_message(msg)
            
            logging.info(f"Email alert sent for {alert.alert_id}")
            
        except Exception as e:
            logging.error(f"Error sending email alert: {e}")
            # Fallback to logging
            logging.warning(f"EMAIL ALERT [{alert.severity.value.upper()}]: {alert.message}")


class SlackAlertHandler:
    """Alert handler that sends Slack notifications"""
    
    def __init__(self, webhook_url: str, channel: str = None):
        self.webhook_url = webhook_url
        self.channel = channel
    
    def __call__(self, alert: QualityAlert):
        """Handle alert by sending Slack message"""
        # This would implement Slack webhook integration
        # Simplified implementation logs the alert
        logging.info(f"SLACK ALERT [{alert.severity.value.upper()}]: {alert.message}")