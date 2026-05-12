"""
Pruning module for NetGPT post-training compression.
Implements Magnitude, Wanda, and Pruner-Zero metrics.
"""
from pruning.metrics import METRICS, magnitude_score, wanda_score, pruner_zero_score
from pruning.pruner import NetGPTPruner
