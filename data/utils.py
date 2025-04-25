import statistics

def convert_str_to_list(interval_str: str) -> list[int]:
    """Convert '4/2/2' etc. to [4, 2, 2]"""
    interval_str = interval_str.strip('/')
    return [int(x) for x in interval_str.split('/')]

def get_max(interval_str: str) -> int:
    """Return the maximum value in the interval list."""
    interval = convert_str_to_list(interval_str)
    return max(interval)

def get_median(interval_str: str) -> float:
    """Return the median value in the interval list."""
    interval = convert_str_to_list(interval_str)
    return statistics.median(interval)

def get_average(interval_str: str) -> float:
    """Return the average (mean) value in the interval list."""
    interval = convert_str_to_list(interval_str)
    return statistics.mean(interval)

def categorize_sleep_start_hour(hour):
    return {
        "start1": int(0 <= hour < 4),
        "start2": int(4 <= hour < 8),
        "start3": int(8 <= hour < 12),
        "start4": int(12 <= hour < 16),
        "start5": int(16 <= hour < 20),
        "start6": int(20 <= hour < 24),
    }

def categorize_sleep_end_hour(hour):
    return {
        "end1": int(0 <= hour < 4),
        "end2": int(4 <= hour < 8),
        "end3": int(8 <= hour < 12),
        "end4": int(12 <= hour < 16),
        "end5": int(16 <= hour < 20),
        "end6": int(20 <= hour < 24),
    }