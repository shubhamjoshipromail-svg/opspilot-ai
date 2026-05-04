from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.app.database import get_db
from backend.app.models import Ticket
from backend.app.schemas import TicketCreate, TicketRead, TicketUpdate


router = APIRouter(prefix="/tickets", tags=["tickets"])


def _schema_to_dict(schema: TicketCreate | TicketUpdate, *, exclude_unset: bool = False) -> dict:
    """Support both Pydantic v1 and v2 while keeping route code readable."""
    if hasattr(schema, "model_dump"):
        return schema.model_dump(exclude_unset=exclude_unset)
    return schema.dict(exclude_unset=exclude_unset)


def _get_ticket_or_404(db: Session, ticket_id: int) -> Ticket:
    ticket = db.get(Ticket, ticket_id)
    if ticket is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Ticket {ticket_id} not found",
        )
    return ticket


@router.post("", response_model=TicketRead, status_code=status.HTTP_201_CREATED)
def create_ticket(ticket_in: TicketCreate, db: Session = Depends(get_db)) -> Ticket:
    """Create a ticket and persist it to the configured database."""
    ticket = Ticket(**_schema_to_dict(ticket_in))
    db.add(ticket)
    db.commit()
    db.refresh(ticket)
    return ticket


@router.get("", response_model=list[TicketRead])
def list_tickets(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
) -> list[Ticket]:
    """List tickets with simple offset pagination."""
    return db.query(Ticket).offset(skip).limit(limit).all()


@router.get("/{ticket_id}", response_model=TicketRead)
def get_ticket(ticket_id: int, db: Session = Depends(get_db)) -> Ticket:
    """Return one ticket by database ID."""
    return _get_ticket_or_404(db, ticket_id)


@router.patch("/{ticket_id}", response_model=TicketRead)
def update_ticket(
    ticket_id: int,
    ticket_in: TicketUpdate,
    db: Session = Depends(get_db),
) -> Ticket:
    """Apply a partial update to a ticket."""
    ticket = _get_ticket_or_404(db, ticket_id)
    update_data = _schema_to_dict(ticket_in, exclude_unset=True)

    for field, value in update_data.items():
        setattr(ticket, field, value)

    ticket.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(ticket)
    return ticket


@router.delete("/{ticket_id}")
def delete_ticket(ticket_id: int, db: Session = Depends(get_db)) -> dict[str, bool | int]:
    """Delete a ticket by database ID."""
    ticket = _get_ticket_or_404(db, ticket_id)
    db.delete(ticket)
    db.commit()
    return {"deleted": True, "ticket_id": ticket_id}
