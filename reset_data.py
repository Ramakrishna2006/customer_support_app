"""Reset the demo data: restore the original orders and delete all support tickets.

    python reset_data.py
"""
import shutil

from config import DATA_DIR

orders, seed, tickets = DATA_DIR / "orders.csv", DATA_DIR / "orders_seed.csv", DATA_DIR / "tickets.csv"

shutil.copy(seed, orders)
print("✅ data/orders.csv restored from data/orders_seed.csv")
if tickets.exists():
    tickets.unlink()
    print("✅ data/tickets.csv deleted")
