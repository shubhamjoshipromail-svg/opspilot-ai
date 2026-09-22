from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from backend.app.database import get_db
from backend.app.schemas.dashboard import DashboardSummary, RecentRoutingPrediction
from backend.app.services.dashboard import get_dashboard_summary, get_recent_predictions


router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])


@router.get("/summary", response_model=DashboardSummary)
def dashboard_summary(db: Session = Depends(get_db)) -> DashboardSummary:
    return get_dashboard_summary(db)


@router.get(
    "/recent-predictions",
    response_model=list[RecentRoutingPrediction],
)
def recent_predictions(
    limit: int = Query(default=20, ge=1, le=100),
    db: Session = Depends(get_db),
) -> list[RecentRoutingPrediction]:
    return get_recent_predictions(db, limit=limit)
