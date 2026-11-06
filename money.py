"""Exact monetary arithmetic; rounding is confined to division and display."""

from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN, localcontext


COST_FIELDS = ("input_cost_usd", "cached_input_cost_usd", "output_cost_usd")


def amount(value):
    """Reject binary floats and booleans at accounting boundaries."""
    if not isinstance(value, (str, int, Decimal)) or isinstance(value, bool):
        raise ValueError("Expected an exact decimal amount.")
    try:
        result = Decimal(value)
    except InvalidOperation:
        raise ValueError("Invalid decimal amount.") from None
    if not result.is_finite() or abs(result.adjusted()) > 100:
        raise ValueError("Amount must be finite and within supported bounds.")
    if len(result.as_tuple().digits) > 256 or result.as_tuple().exponent < -256:
        raise ValueError("Amount exceeds supported precision.")
    return result


def add(values):
    values = [amount(v) for v in values]
    if not values:
        return Decimal(0)
    integer_digits = max(max(v.adjusted() + 1, 1) for v in values)
    fractional_digits = max(max(-v.as_tuple().exponent, 0) for v in values)
    with localcontext() as context:
        context.prec = max(28, integer_digits + fractional_digits + len(str(len(values))) + 2)
        return sum(values, Decimal(0))


def multiply(left, right):
    a, b = amount(left), amount(right)
    with localcontext() as context:
        context.prec = max(28, len(a.as_tuple().digits) + len(b.as_tuple().digits) + 2)
        return a * b


def divide(numerator, denominator):
    """Recurring quotients use 28 significant digits and half-even rounding."""
    with localcontext() as context:
        context.prec = 28
        context.rounding = ROUND_HALF_EVEN
        return amount(numerator) / amount(denominator)


def display(value, currency="USD", rate="1"):
    if value is None:
        return None
    symbols = {"USD": "$", "EUR": "€", "INR": "₹", "GBP": "£"}
    if currency not in symbols or amount(rate) <= 0:
        raise ValueError("Invalid display currency or conversion rate.")
    converted = multiply(value, rate)
    with localcontext() as context:
        context.prec = max(28, converted.adjusted() + 12)
        rounded = converted.quantize(Decimal("0.00000001"), rounding=ROUND_HALF_EVEN)
    if converted and not rounded:
        return ("-" if converted < 0 else "") + "<" + symbols[currency] + "0.00000001"
    digits = f"{rounded.copy_abs():,.8f}".rstrip("0")
    if digits.endswith("."):
        digits += "00"
    elif len(digits.rsplit(".", 1)[1]) == 1:
        digits += "0"
    return ("-" if rounded < 0 else "") + symbols[currency] + digits


def cost_display(summary, currency="USD", rate="1"):
    """Use the same formatting for dashboard totals and component breakdowns."""
    return {key: display(summary.get(key), currency, rate)
            for key in ("estimated_cost_usd", "known_cost_subtotal_usd", "average_cost_per_query_usd")}
