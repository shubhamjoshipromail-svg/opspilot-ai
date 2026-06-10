"""Create the initial OpsPilot product backbone tables."""

from alembic import op
import sqlalchemy as sa


revision = "0001_product_backbone"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    existing_tables = set(inspector.get_table_names())

    if "tickets" not in existing_tables:
        op.create_table(
            "tickets",
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column("external_id", sa.String(), nullable=True),
            sa.Column("customer_message", sa.Text(), nullable=False),
            sa.Column("channel", sa.String(), nullable=True),
            sa.Column("source", sa.String(), nullable=True),
            sa.Column("true_category", sa.String(), nullable=True),
            sa.Column("true_priority", sa.String(), nullable=True),
            sa.Column("ticket_type", sa.String(), nullable=True),
            sa.Column("language", sa.String(), nullable=True),
            sa.Column("reference_answer", sa.Text(), nullable=True),
            sa.Column("tags", sa.Text(), nullable=True),
            sa.Column("status", sa.String(), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        )
        op.create_index("ix_tickets_id", "tickets", ["id"])
        op.create_index("ix_tickets_external_id", "tickets", ["external_id"])
        op.create_index("ix_tickets_status", "tickets", ["status"])

    if "ticket_routing_predictions" not in existing_tables:
        op.create_table(
            "ticket_routing_predictions",
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column(
                "ticket_id",
                sa.Integer(),
                sa.ForeignKey("tickets.id", ondelete="SET NULL"),
                nullable=True,
            ),
            sa.Column("subject", sa.String(length=500), nullable=False),
            sa.Column("body", sa.Text(), nullable=False),
            sa.Column("predicted_queue", sa.String(length=100), nullable=False),
            sa.Column("confidence", sa.Float(), nullable=False),
            sa.Column("threshold", sa.Float(), nullable=False),
            sa.Column("decision", sa.String(length=30), nullable=False),
            sa.Column("probabilities", sa.JSON(), nullable=False),
            sa.Column("model_id", sa.String(length=255), nullable=False),
            sa.Column("latency_ms", sa.Integer(), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        )
        op.create_index(
            "ix_ticket_routing_predictions_id",
            "ticket_routing_predictions",
            ["id"],
        )
        op.create_index(
            "ix_ticket_routing_predictions_ticket_id",
            "ticket_routing_predictions",
            ["ticket_id"],
        )
        op.create_index(
            "ix_ticket_routing_predictions_predicted_queue",
            "ticket_routing_predictions",
            ["predicted_queue"],
        )
        op.create_index(
            "ix_ticket_routing_predictions_decision",
            "ticket_routing_predictions",
            ["decision"],
        )
        op.create_index(
            "ix_ticket_routing_predictions_created_at",
            "ticket_routing_predictions",
            ["created_at"],
        )


def downgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    existing_tables = set(inspector.get_table_names())
    if "ticket_routing_predictions" in existing_tables:
        op.drop_table("ticket_routing_predictions")
