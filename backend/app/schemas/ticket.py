from datetime import datetime

from pydantic import BaseModel

try:
    from pydantic import ConfigDict

    PYDANTIC_V2 = True
except ImportError:  # pragma: no cover - compatibility for Pydantic v1 environments
    ConfigDict = None  # type: ignore[assignment]
    PYDANTIC_V2 = False


class TicketBase(BaseModel):
    external_id: str | None = None
    customer_message: str
    channel: str | None = None
    source: str | None = None
    true_category: str | None = None
    true_priority: str | None = None
    ticket_type: str | None = None
    language: str | None = None
    reference_answer: str | None = None
    tags: str | None = None
    status: str = "new"


class TicketCreate(TicketBase):
    """Payload for creating a ticket."""


class TicketUpdate(BaseModel):
    """Payload for partial ticket updates."""

    external_id: str | None = None
    customer_message: str | None = None
    channel: str | None = None
    source: str | None = None
    true_category: str | None = None
    true_priority: str | None = None
    ticket_type: str | None = None
    language: str | None = None
    reference_answer: str | None = None
    tags: str | None = None
    status: str | None = None


class TicketRead(TicketBase):
    id: int
    created_at: datetime
    updated_at: datetime

    if PYDANTIC_V2:
        model_config = ConfigDict(from_attributes=True)
    else:

        class Config:
            orm_mode = True
