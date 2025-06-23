from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timezone
import json

db = SQLAlchemy()

class Alert(db.Model):
    """Model for storing crypto pre-pump alerts"""
    __tablename__ = 'alerts'
    
    id = db.Column(db.Integer, primary_key=True)
    symbol = db.Column(db.String(20), nullable=False, index=True)
    ppwcs_score = db.Column(db.Float, nullable=False)
    tags = db.Column(db.Text)  # JSON string of tags
    compressed = db.Column(db.Boolean, default=False)
    stage1g_active = db.Column(db.Boolean, default=False)
    alert_message = db.Column(db.Text)
    telegram_sent = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc), index=True)
    
    # Relationships
    gpt_analyses = db.relationship('GPTAnalysis', backref='alert', lazy=True)
    
    def to_dict(self):
        return {
            'id': self.id,
            'symbol': self.symbol,
            'ppwcs_score': self.ppwcs_score,
            'tags': json.loads(self.tags) if self.tags else [],
            'compressed': self.compressed,
            'stage1g_active': self.stage1g_active,
            'telegram_sent': self.telegram_sent,
            'created_at': self.created_at.isoformat(),
            'gpt_analysis': self.gpt_analyses[0].to_dict() if self.gpt_analyses else None
        }

class GPTAnalysis(db.Model):
    """Model for storing GPT analysis results"""
    __tablename__ = 'gpt_analyses'
    
    id = db.Column(db.Integer, primary_key=True)
    alert_id = db.Column(db.Integer, db.ForeignKey('alerts.id'), nullable=False)
    symbol = db.Column(db.String(20), nullable=False, index=True)
    analysis_text = db.Column(db.Text, nullable=False)
    risk_assessment = db.Column(db.String(20))  # low, medium, high
    confidence_level = db.Column(db.Integer)  # 0-100
    price_prediction = db.Column(db.String(20))  # bullish, bearish, neutral
    entry_recommendation = db.Column(db.String(20))  # immediate, wait, avoid
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc), index=True)
    
    def to_dict(self):
        return {
            'id': self.id,
            'symbol': self.symbol,
            'analysis_text': self.analysis_text,
            'risk_assessment': self.risk_assessment,
            'confidence_level': self.confidence_level,
            'price_prediction': self.price_prediction,
            'entry_recommendation': self.entry_recommendation,
            'created_at': self.created_at.isoformat()
        }

class Symbol(db.Model):
    """Model for tracking symbol performance"""
    __tablename__ = 'symbols'
    
    id = db.Column(db.Integer, primary_key=True)
    symbol = db.Column(db.String(20), nullable=False, unique=True, index=True)
    total_alerts = db.Column(db.Integer, default=0)
    avg_ppwcs_score = db.Column(db.Float, default=0.0)
    last_alert_at = db.Column(db.DateTime)
    success_rate = db.Column(db.Float, default=0.0)  # % of successful predictions
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    
    def to_dict(self):
        return {
            'id': self.id,
            'symbol': self.symbol,
            'total_alerts': self.total_alerts,
            'avg_ppwcs_score': self.avg_ppwcs_score,
            'last_alert_at': self.last_alert_at.isoformat() if self.last_alert_at else None,
            'success_rate': self.success_rate,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }

class ScanSession(db.Model):
    """Model for tracking scan sessions"""
    __tablename__ = 'scan_sessions'
    
    id = db.Column(db.Integer, primary_key=True)
    symbols_scanned = db.Column(db.Integer, default=0)
    alerts_generated = db.Column(db.Integer, default=0)
    gpt_analyses_created = db.Column(db.Integer, default=0)
    scan_duration = db.Column(db.Float)  # seconds
    status = db.Column(db.String(20), default='running')  # running, completed, failed
    error_message = db.Column(db.Text)
    started_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    completed_at = db.Column(db.DateTime)
    
    def to_dict(self):
        return {
            'id': self.id,
            'symbols_scanned': self.symbols_scanned,
            'alerts_generated': self.alerts_generated,
            'gpt_analyses_created': self.gpt_analyses_created,
            'scan_duration': self.scan_duration,
            'status': self.status,
            'started_at': self.started_at.isoformat(),
            'completed_at': self.completed_at.isoformat() if self.completed_at else None
        }