import datetime as dt
from typing import List, Optional

from sqlalchemy import (
    Boolean,
    Column,
    Integer,
    String,
    DateTime,
    ForeignKey,
    Float,
    Text,
    Index,
    JSON,
)
from sqlalchemy.orm import relationship

from .database import Base
from .time_utils import moscow_now


class Bank(Base):
    __tablename__ = "banks"

    id: int = Column(Integer, primary_key=True, index=True)
    name: str = Column(String(255), unique=True, nullable=False)
    slug: str = Column(String(255), unique=True, nullable=True)

    reviews = relationship("Review", back_populates="bank", cascade="all, delete-orphan")
    review_risks = relationship(
        "ReviewRisk", back_populates="bank", cascade="all, delete-orphan"
    )


class Review(Base):
    __tablename__ = "reviews"

    id: int = Column(Integer, primary_key=True, index=True)

    external_id: Optional[str] = Column(String(64), index=True, nullable=True)

    bank_id: int = Column(Integer, ForeignKey("banks.id"), nullable=False, index=True)
    rating: Optional[float] = Column(Float, nullable=True)

    title: Optional[str] = Column(String(500), nullable=True)
    text: str = Column(Text, nullable=False)

    published_at: Optional[dt.datetime] = Column(DateTime, nullable=True, index=True)
    scraped_at: dt.datetime = Column(
        DateTime, default=moscow_now, nullable=False, index=True
    )
    vector_in_qdrant: bool = Column(Boolean, default=False, nullable=False, index=True)
    sentiment: Optional[str] = Column(String(16), nullable=True, index=True)
    sentiment_score: Optional[float] = Column(Float, nullable=True)
    criticality_score: Optional[int] = Column(Integer, nullable=True, index=True)

    bank = relationship("Bank", back_populates="reviews")
    review_risks = relationship(
        "ReviewRisk",
        back_populates="review",
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        Index(
            "ix_reviews_unique_external",
            "external_id",
            "bank_id",
            unique=True,
        ),
    )


class Risk(Base):
    __tablename__ = "risks"

    id: int = Column(Integer, primary_key=True, index=True)
    risk_type: str = Column(String(64), nullable=False, default="unspecified")
    description: str = Column(Text, nullable=False, default="")
    risk_factors: Optional[List[str]] = Column(JSON, nullable=True)
    implications: Optional[List[str]] = Column(JSON, nullable=True)

    review_risks = relationship(
        "ReviewRisk", back_populates="risk", cascade="all, delete-orphan"
    )


class ReviewRisk(Base):
    __tablename__ = "review_risks"

    id: int = Column(Integer, primary_key=True, index=True)
    bank_id: int = Column(Integer, ForeignKey("banks.id"), nullable=False, index=True)
    review_id: int = Column(Integer, ForeignKey("reviews.id"), nullable=False, index=True)
    risk_id: int = Column(Integer, ForeignKey("risks.id"), nullable=False, index=True)
    created_at: dt.datetime = Column(DateTime, default=moscow_now, nullable=False, index=True)
    review_date: Optional[dt.datetime] = Column(DateTime, nullable=True, index=True)

    bank = relationship("Bank", back_populates="review_risks")
    review = relationship("Review", back_populates="review_risks")
    risk = relationship("Risk", back_populates="review_risks")

    __table_args__ = (
        Index("ix_review_risks_review_id_unique", "review_id", unique=True),
    )
