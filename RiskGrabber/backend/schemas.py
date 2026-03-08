import datetime as dt
from typing import List, Optional

from pydantic import BaseModel, ConfigDict


class ReviewBase(BaseModel):
    id: int
    bank_id: int
    rating: Optional[float]
    title: Optional[str]
    text: str
    published_at: Optional[dt.datetime]
    scraped_at: dt.datetime

    model_config = ConfigDict(from_attributes=True)


class BankBase(BaseModel):
    id: int
    name: str
    slug: Optional[str]

    model_config = ConfigDict(from_attributes=True)


class RiskBase(BaseModel):
    """Риск: вид (risk_type), описание, факторы, последствия."""
    id: int
    risk_type: str
    description: str = ""
    risk_factors: Optional[List[str]] = None
    implications: Optional[List[str]] = None

    model_config = ConfigDict(from_attributes=True)


class BankStats(BaseModel):
    bank: BankBase
    total_reviews: int
    positive_reviews: int = 0
    negative_reviews: int = 0
    avg_rating: Optional[float]
    last_review_at: Optional[dt.datetime]
    risks_count: int
    risk_score: Optional[float] = None  # weighted severity score (for period), optional


class TimeBucketStats(BaseModel):
    start: dt.datetime
    end: dt.datetime
    reviews_count: int
    avg_rating: Optional[float]
    risks_count: int


class ReviewsOverTimeBucket(BaseModel):
    """За интервал: дата и количество по сентименту (только pos/neg) и уникальные риски."""
    date: str  # YYYY-MM-DD
    positive: int = 0
    negative: int = 0
    risks_count: int = 0


class ReviewsOverTimeResponse(BaseModel):
    buckets: List["ReviewsOverTimeBucket"]


class SentimentStats(BaseModel):
    positive: int
    negative: int


class RiskSummary(BaseModel):
    """Риск для списка на дашборде: вид (risk_type), описание, факторы; банк и количество отзывов."""
    id: int
    bank: Optional[BankBase]
    risk_type: str
    description: str = ""
    created_at: dt.datetime
    reviews_count: int = 1


class VectorPoint(BaseModel):
    review_id: int
    bank_id: int
    x: float
    y: float
    rating: Optional[float]
    sentiment: Optional[str]


class VectorMapResponse(BaseModel):
    points: List[VectorPoint]


class ReviewDetail(BaseModel):
    id: int
    bank: Optional[BankBase]
    rating: Optional[float]
    title: Optional[str]
    text: str
    published_at: Optional[dt.datetime]
    scraped_at: dt.datetime
    sentiment: Optional[str]
    sentiment_score: Optional[float]
    criticality_score: Optional[int] = None  # 1–5, степень критичности отзыва

    model_config = ConfigDict(from_attributes=True)


class RiskDetailWithReviews(BaseModel):
    """Риск с полным описанием и списком привязанных отзывов."""
    risk: RiskBase
    reviews: List[ReviewDetail]


class DashboardResponse(BaseModel):
    banks: List[BankStats]
    overall: List[TimeBucketStats]
    sentiment: SentimentStats
    risks: List[RiskSummary]
    total_reviews: int = 0
    total_banks: int = 0
    reviews_date_min: Optional[dt.datetime] = None
    reviews_date_max: Optional[dt.datetime] = None
    reviews_with_sentiment: int = 0
    reviews_without_sentiment: int = 0
    negative_reviews_without_risk: int = 0
    negative_reviews_with_risk: int = 0
    bank_scores: Optional[List["BankScoreItem"]] = None


# --- Analytics ---


class RiskCountInBucket(BaseModel):
    risk_id: int
    risk_type: str
    count: int


class RiskTrendBucket(BaseModel):
    start: dt.datetime
    end: dt.datetime
    risks: List[RiskCountInBucket]


class RiskTrendsResponse(BaseModel):
    intervals: List[RiskTrendBucket]
    risk_meta: List[RiskBase]  # unique risks with id, risk_type, description, risk_factors


class HotRiskItem(BaseModel):
    """Один риск в интервале «горячих рисков»: объём и взвешенная метрика (без LLM)."""
    risk_id: int
    risk_type: str
    reviews_count: int
    hot_score: float  # сумма criticality_score по отзывам за интервал (или count * avg)
    avg_criticality: Optional[float] = None


class HotRisksBucket(BaseModel):
    start: dt.datetime
    end: dt.datetime
    hot_risks: List[HotRiskItem]


class HotRisksResponse(BaseModel):
    """Горячие риски по интервалам (день/неделя/месяц/год), опционально по банку."""
    intervals: List[HotRisksBucket]
    risk_meta: List[RiskBase]
    bank_id: Optional[int] = None
    bank_name: Optional[str] = None


class BankRiskTrendsResponse(BaseModel):
    bank_id: int
    bank_name: str
    intervals: List[RiskTrendBucket]
    risk_meta: List[RiskBase]


class RiskMatrixCell(BaseModel):
    bank_id: int
    bank_name: str
    risk_type: str
    avg_criticality: Optional[float] = None  # средняя серьёзность отзывов (1–5) по паре банк × вид риска


class RiskMatrixResponse(BaseModel):
    cells: List[RiskMatrixCell]
    banks: List[BankBase]
    risk_types: List[str]  # порядок видов риска для столбцов (топ по объёму)


class BankScoreItem(BaseModel):
    bank_id: int
    bank_name: str
    risk_score: float  # объединённая метрика: байес × √объём, нормализована в 1–5
    risks_count: int
    total_reviews: int
    avg_severity: Optional[float] = None  # средняя серьёзность (w/n_risk), 1–5


class CriticalityCount(BaseModel):
    score: int
    count: int


class BankCriticalityItem(BaseModel):
    bank_id: int
    bank_name: str
    distribution: List[CriticalityCount]


class ReviewCriticalityResponse(BaseModel):
    """Распределение отзывов по степени критичности (1–5) по банкам."""
    banks: List[BankCriticalityItem]

