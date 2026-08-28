"""SQLAlchemy ORM models for MAYA product entities."""

from __future__ import annotations

from datetime import datetime, timezone

from flask_login import UserMixin
from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.extensions import db
from backend.app.models.enums import (
    AnalysisStatus,
    CasePriority,
    CaseStatus,
    EvidenceStatus,
    UserRole,
)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class User(UserMixin, db.Model):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    username: Mapped[str] = mapped_column(String(80), unique=True, nullable=False, index=True)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    full_name: Mapped[str] = mapped_column(String(255), nullable=False, default="")
    role: Mapped[str] = mapped_column(
        String(32), nullable=False, default=UserRole.INVESTIGATOR.value, index=True
    )
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=_utcnow)
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    cases: Mapped[list[Case]] = relationship("Case", back_populates="owner", lazy="dynamic")
    evidence_uploads: Mapped[list[Evidence]] = relationship(
        "Evidence", back_populates="uploader", lazy="dynamic"
    )
    audit_logs: Mapped[list[AuditLog]] = relationship(
        "AuditLog", back_populates="user", lazy="dynamic"
    )

    def get_id(self) -> str:
        return str(self.id)

    @property
    def is_admin(self) -> bool:
        return self.role == UserRole.ADMIN.value


class Case(db.Model):
    __tablename__ = "cases"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    case_number: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False, default="")
    status: Mapped[str] = mapped_column(
        String(32), nullable=False, default=CaseStatus.OPEN.value, index=True
    )
    priority: Mapped[str] = mapped_column(
        String(32), nullable=False, default=CasePriority.MEDIUM.value
    )
    created_by_user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id"), nullable=False, index=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=_utcnow, onupdate=_utcnow
    )
    closed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    owner: Mapped[User] = relationship("User", back_populates="cases")
    evidence_items: Mapped[list[Evidence]] = relationship(
        "Evidence", back_populates="case", lazy="dynamic", cascade="all, delete-orphan"
    )
    analysis_runs: Mapped[list[AnalysisRun]] = relationship(
        "AnalysisRun", back_populates="case", lazy="dynamic"
    )


class Evidence(db.Model):
    __tablename__ = "evidence"
    __table_args__ = (UniqueConstraint("case_id", "stored_filename", name="uq_case_stored_file"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    case_id: Mapped[int] = mapped_column(Integer, ForeignKey("cases.id"), nullable=False, index=True)
    original_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    stored_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    storage_path: Mapped[str] = mapped_column(String(512), nullable=False)
    media_type: Mapped[str] = mapped_column(String(32), nullable=False, default="image")
    mime_type: Mapped[str] = mapped_column(String(128), nullable=False)
    file_size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    sha256_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    status: Mapped[str] = mapped_column(
        String(32), nullable=False, default=EvidenceStatus.UPLOADED.value, index=True
    )
    analysis_status: Mapped[str] = mapped_column(
        String(32), nullable=False, default=AnalysisStatus.NONE.value
    )
    uploaded_by_user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id"), nullable=False, index=True
    )
    uploaded_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=_utcnow)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    case: Mapped[Case] = relationship("Case", back_populates="evidence_items")
    uploader: Mapped[User] = relationship("User", back_populates="evidence_uploads")
    analysis_runs: Mapped[list[AnalysisRun]] = relationship(
        "AnalysisRun", back_populates="evidence", lazy="dynamic"
    )


class AnalysisRun(db.Model):
    __tablename__ = "analysis_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    evidence_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("evidence.id"), nullable=False, index=True
    )
    case_id: Mapped[int] = mapped_column(Integer, ForeignKey("cases.id"), nullable=False, index=True)
    investigation_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    status: Mapped[str] = mapped_column(
        String(32), nullable=False, default=AnalysisStatus.QUEUED.value, index=True
    )
    prediction: Mapped[str | None] = mapped_column(String(32), nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    model_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    model_version: Mapped[str | None] = mapped_column(String(128), nullable=True)
    dataset_version: Mapped[str | None] = mapped_column(String(128), nullable=True)
    explainer_name: Mapped[str | None] = mapped_column(String(64), nullable=True)
    generate_explanation: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    artifact_dir: Mapped[str | None] = mapped_column(String(512), nullable=True)
    heatmap_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    overlay_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    explanation_json_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    raw_result_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_by_user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id"), nullable=False, index=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=_utcnow)

    evidence: Mapped[Evidence] = relationship("Evidence", back_populates="analysis_runs")
    case: Mapped[Case] = relationship("Case", back_populates="analysis_runs")
    creator: Mapped[User] = relationship("User")


class AuditLog(db.Model):
    __tablename__ = "audit_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("users.id"), nullable=True, index=True
    )
    case_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("cases.id"), nullable=True, index=True
    )
    evidence_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("evidence.id"), nullable=True, index=True
    )
    analysis_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("analysis_runs.id"), nullable=True, index=True
    )
    event_type: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=_utcnow, index=True)
    details_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")

    user: Mapped[User | None] = relationship("User", back_populates="audit_logs")


# Import side-effect for create_all: models must be imported before db.create_all()
__all__ = [
    "AnalysisRun",
    "AuditLog",
    "Case",
    "Evidence",
    "User",
]
