from datetime import date, timedelta


def working_days_between(a: date, b: date) -> list[date]:
    """
    Calculate the number of working days (Monday to Friday) between two dates.

    Args:
        a (date): The start date.
        b (date): The end date.

    Returns:
        list(date): List of working days between two dates
    """

    if b <= a:
        return []

    total_days = (b - a).days
    tmp = [ a+timedelta(days=day) for day in range(total_days) ]
    return list(filter(lambda x: x.weekday() < 5, tmp))
