"""
Layer 2: Temporal Focus Layer
Aggregates time-series along the time dimension to identify critical time windows
"""

from .time_agg import TimeAgg, TimeAggAbun

__all__ = ['TimeAgg', 'TimeAggAbun']
