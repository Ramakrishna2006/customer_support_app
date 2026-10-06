"""
Tools the support agent can use.

A tool is a normal Python function wrapped with @tool. The docstring matters:
the LLM reads it to decide WHEN to call the tool and WHAT to pass in.
In a real company these functions would call your order database or ticketing
system (Zendesk, Freshdesk, Jira). Here they read and write CSV files.
"""
import csv
import re
import shutil
from datetime import datetime

import pandas as pd
from langchain_core.tools import tool

from config import DATA_DIR

# orders_seed.csv is the original sample data (committed to Git).
# orders.csv is the working copy the tools edit; it is created from the seed on first use
# and ignored by Git. Delete it (and tickets.csv) to reset the demo: python reset_data.py
SEED_PATH = DATA_DIR / "orders_seed.csv"
ORDERS_PATH = DATA_DIR / "orders.csv"
TICKETS_PATH = DATA_DIR / "tickets.csv"


def _load_orders() -> pd.DataFrame:
    if not ORDERS_PATH.exists():
        shutil.copy(SEED_PATH, ORDERS_PATH)
    return pd.read_csv(ORDERS_PATH, dtype=str).fillna("")


def _clean_id(order_id: str) -> str:
    """Normalise IDs like 'ord-1002', '#ORD 1002' or '1002' (common in voice transcripts) to 'ORD1002'."""
    cleaned = re.sub(r"[^A-Za-z0-9]", "", str(order_id)).upper()
    return f"ORD{cleaned}" if cleaned.isdigit() else cleaned


@tool
def check_order_status(order_id: str) -> str:
    """Look up an order by the order ID the customer gave (format ORD followed by digits).
    Only use an ID that appears in the customer's message. Returns the product, status,
    order date, expected delivery date, tracking number and shipping address."""
    orders = _load_orders()
    match = orders[orders["order_id"] == _clean_id(order_id)]
    if match.empty:
        return f"No order found with ID '{order_id}'. Ask the customer to double-check the order ID."
    o = match.iloc[0].to_dict()  # a plain dict avoids clashes with pandas method names like .product
    return (
        f"Order {o['order_id']} for {o['customer_name']}: {o['product']} (Rs {o['amount_inr']}). "
        f"Status: {o['status']}. Ordered on {o['order_date']}. "
        f"Expected delivery: {o['expected_delivery'] or 'not applicable'}. "
        f"Tracking number: {o['tracking_number'] or 'not assigned yet'}. "
        f"Shipping address: {o['shipping_address']}."
    )


@tool
def update_delivery_address(order_id: str, new_address: str) -> str:
    """Change the delivery address of an order. Only works while the order status is
    'Processing'. Use this only when the customer has given the full new address."""
    orders = _load_orders()
    oid = _clean_id(order_id)
    idx = orders.index[orders["order_id"] == oid]
    if len(idx) == 0:
        return f"No order found with ID '{order_id}'."
    status = orders.at[idx[0], "status"]
    if status != "Processing":
        return (
            f"Cannot change the address: order {oid} is already '{status}'. "
            "Addresses can only be changed while an order is Processing."
        )
    orders.at[idx[0], "shipping_address"] = new_address.strip()
    orders.to_csv(ORDERS_PATH, index=False)
    return f"Delivery address for {oid} updated to: {new_address.strip()}"


@tool
def search_knowledge_base(query: str) -> str:
    """Search the company help centre for policies: returns, refunds, shipping,
    cancellations, payments, warranty, account/login help and support hours."""
    from knowledge_base import search_faq  # imported here so the tool file loads without an API key

    return search_faq(query, k=2)


@tool
def create_support_ticket(issue_summary: str, priority: str = "medium") -> str:
    """Create a ticket so a human agent follows up. Use for complaints, damaged items,
    payment problems, or anything you cannot solve. priority must be low, medium or high."""
    priority = priority.lower() if priority.lower() in {"low", "medium", "high"} else "medium"
    is_new = not TICKETS_PATH.exists()
    if is_new:
        next_num = 5001
    else:
        with open(TICKETS_PATH, newline="", encoding="utf-8") as f:
            next_num = 5001 + sum(1 for _ in csv.DictReader(f))
    ticket_id = f"TKT{next_num}"
    with open(TICKETS_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["ticket_id", "created_at", "priority", "issue_summary", "status"])
        writer.writerow([ticket_id, datetime.now().isoformat(timespec="seconds"), priority, issue_summary, "open"])
    return f"Ticket {ticket_id} created with {priority} priority. A human agent will contact the customer."


ALL_TOOLS = [check_order_status, update_delivery_address, search_knowledge_base, create_support_ticket]


if __name__ == "__main__":
    # Tools can be called directly, without any LLM, which makes them easy to test.
    print(check_order_status.invoke({"order_id": "ORD1002"}))
    print(check_order_status.invoke({"order_id": "ORD9999"}))
    print(update_delivery_address.invoke({"order_id": "ORD1001", "new_address": "New street"}))
