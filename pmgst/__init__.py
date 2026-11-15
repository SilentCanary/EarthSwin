"""PMGST-Net: Physics-guided Multi-scale Spatio-Temporal Graph Transformer Network.

Package map (block numbers = the architecture diagram):
  data/        blocks 1-3   chip format, synthetic generator, dataset, splits, GEE export
  models/      blocks 3-9, 11   encoders, pyramid, fusion, graph, GAT, temporal, physics, memory, heads
  losses.py    block 14
  xai.py       block 13
  infer.py     block 12   risk map, polygons, warning levels
  serve.py     block 15   API
"""

__version__ = "0.1.0"
