"""Source-to-warehouse column mapping and the pydantic models for ingested rows.

The IBM HI-Small CSV header is:

    Timestamp,From Bank,Account,To Bank,Account,Amount Received,Receiving Currency,
    Amount Paid,Payment Currency,Payment Format,<target>

"Account" appears twice, so columns are mapped by position, not by name. The 11th
field is the target; it is parsed only by the evaluation package and has no entry here.
"""

from datetime import date, datetime
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel

# (1-based field position, source header, warehouse column)
SOURCE_COLUMNS: tuple[tuple[int, str, str], ...] = (
    (1, "Timestamp", "timestamp"),
    (2, "From Bank", "from_bank"),
    (3, "Account", "from_account"),
    (4, "To Bank", "to_bank"),
    (5, "Account", "to_account"),
    (6, "Amount Received", "amount_received"),
    (7, "Receiving Currency", "receiving_currency"),
    (8, "Amount Paid", "amount_paid"),
    (9, "Payment Currency", "payment_currency"),
    (10, "Payment Format", "payment_format"),
)

# Total fields per source line, including the target column (parsed only during evaluation).
SOURCE_FIELD_COUNT = 11

SOURCE_TIMESTAMP_FORMAT = "%Y/%m/%d %H:%M"

Split = Literal["train", "test"]


class Transaction(BaseModel):
    txn_id: str
    row_index: int
    timestamp: datetime
    from_bank: str
    from_account: str
    to_bank: str
    to_account: str
    amount_received: Decimal
    receiving_currency: str
    amount_paid: Decimal
    payment_currency: str
    payment_format: str
    amount_received_usd: Decimal
    amount_paid_usd: Decimal
    split: Split


class Customer(BaseModel):
    """Synthetic profile for one (bank, account). All fields are fake PII."""

    customer_id: str
    bank: str
    account: str
    name: str
    address: str
    date_of_birth: date
    occupation: str
