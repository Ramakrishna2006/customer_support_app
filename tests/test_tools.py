import tools


def test_order_ids_are_normalised():
    assert tools._clean_id("ORD1002") == "ORD1002"
    assert tools._clean_id("ord-1002") == "ORD1002"
    assert tools._clean_id("#ORD 1002") == "ORD1002"
    assert tools._clean_id("1002") == "ORD1002"  # voice transcripts often drop the letters


def test_working_copy_is_created_from_seed(temp_data):
    assert not (temp_data / "orders.csv").exists()
    tools.check_order_status.invoke({"order_id": "ORD1001"})
    assert (temp_data / "orders.csv").exists()


def test_check_order_status(temp_data):
    result = tools.check_order_status.invoke({"order_id": "ord-1002"})
    assert "Smart Watch" in result and "Status: Shipped" in result and "TRK88457" in result
    assert "bound method" not in result  # regression: row.product used to clash with pandas .prod


def test_unknown_order(temp_data):
    assert "No order found" in tools.check_order_status.invoke({"order_id": "ORD9999"})


def test_address_change_only_while_processing(temp_data):
    refused = tools.update_delivery_address.invoke({"order_id": "ORD1002", "new_address": "X"})
    assert "Cannot change" in refused

    done = tools.update_delivery_address.invoke({"order_id": "ORD1003", "new_address": "21 Park Street, Kolkata"})
    assert "updated" in done
    assert "21 Park Street, Kolkata" in tools.check_order_status.invoke({"order_id": "ORD1003"})


def test_tickets_are_numbered_and_priority_is_validated(temp_data):
    first = tools.create_support_ticket.invoke({"issue_summary": "late", "priority": "HIGH"})
    second = tools.create_support_ticket.invoke({"issue_summary": "late again", "priority": "urgent!!"})
    assert "TKT5001" in first and "high priority" in first
    assert "TKT5002" in second and "medium priority" in second
