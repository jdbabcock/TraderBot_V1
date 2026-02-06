# rl/agent.py
"""
LLM action router for both Simulation and Live Binance clients.
Routes standardized actions to the appropriate execution engine.
"""

ALLOWED_ACTIONS = {
    "PLACE_ORDER",
    "CANCEL_ORDER",
    "CANCEL_REPLACE",
    "AMEND_KEEP_PRIORITY",
    "CANCEL_OPEN_ORDERS",
    "GET_OPEN_ORDERS",
    "GET_ALL_ORDERS",
    "GET_ORDER",
    "CREATE_ORDER_LIST",
    "SOR"
}

def execute_llm_action(client, action: dict):
    """
    Route an LLM action to the provided client (Sim or Live).
    """
    if not isinstance(action, dict):
        return {"status": "ERROR", "message": "action must be a dict"}

    # --- Method/Params Normalization ---
    # (Kept your existing logic for supporting 'method' style inputs)
    if "method" in action and "params" in action:
        method = str(action.get("method", "")).lower()
        params = action.get("params", {}) or {}
        if method == "order.place":
            action = {
                "action": "PLACE_ORDER",
                "symbol": params.get("symbol"),
                "side": params.get("side"),
                "type": params.get("type", "MARKET"),
                "quantity": params.get("quantity"),
                "price": params.get("price")
            }
        elif method == "order.cancel":
            action = {
                "action": "CANCEL_ORDER",
                "symbol": params.get("symbol"),
                "orderId": params.get("origClientOrderId") or params.get("orderId")
            }
        # ... (Include other method translations as needed) ...

    act = str(action.get("action", "")).upper()
    if act not in ALLOWED_ACTIONS:
        return {"status": "ERROR", "message": f"unsupported action: {act}"}

    # --- Validation Helpers ---
    def _validate_common(a):
        if not a.get("symbol"): return "symbol is required"
        return None

    def _validate_qty(a):
        qty = a.get("quantity")
        if qty is None: return "quantity is required"
        if float(qty) <= 0: return "quantity must be > 0"
        return None

    err = _validate_common(action)
    if err: return {"status": "ERROR", "message": err}

    # --- Routing to Client Methods ---
    try:
        if act == "PLACE_ORDER":
            err = _validate_qty(action)
            if err: return {"status": "ERROR", "message": err}
            
            # This calls the .create_order method in either Sim or Live client
            return client.create_order(
                symbol=action.get("symbol"),
                side=action.get("side"),
                type=action.get("type", "MARKET"),
                quantity=action.get("quantity"),
                price=action.get("price")
            )

        if act == "CANCEL_ORDER":
            return client.cancel_order(
                symbol=action.get("symbol"),
                orderId=action.get("orderId")
            )

        if act == "GET_OPEN_ORDERS":
            return client.get_open_orders(symbol=action.get("symbol"))

        # ... (Add other actions like CANCEL_REPLACE if your LiveClient supports them) ...

    except AttributeError as e:
        return {"status": "ERROR", "message": f"Client missing method for {act}: {e}"}
    except Exception as e:
        return {"status": "ERROR", "message": str(e)}

    return {"status": "ERROR", "message": "unhandled action"}