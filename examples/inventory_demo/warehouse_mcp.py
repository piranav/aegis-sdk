"""A small stdio MCP server the demo agent connects to: stock levels for a toy store."""

from mcp.server.mcpserver import MCPServer

STOCK = {"KB-42": 7, "MS-07": 0, "HD-11": 23}
server = MCPServer("warehouse", instructions="Stock levels and reservations for the store.")


@server.tool()
def check_stock(sku: str) -> str:
    """Return how many units of a SKU are on hand."""
    units = STOCK.get(sku.upper())
    return f"{sku}: unknown SKU" if units is None else f"{sku}: {units} units on hand"


@server.tool()
def reserve_stock(sku: str, quantity: int) -> str:
    """Reserve units of a SKU for an order."""
    available = STOCK.get(sku.upper(), 0)
    if quantity > available:
        return f"Cannot reserve {quantity} x {sku}: only {available} available"
    STOCK[sku.upper()] = available - quantity
    return f"Reserved {quantity} x {sku}; {STOCK[sku.upper()]} left"


if __name__ == "__main__":
    server.run()
