"""
CaliforniumWhale AI Package
Temporal GNN Detector with QIRL Agent
"""

from .californium_whale_detect import (
    CaliforniumTGN,
    QIRLAgent,
    californium_whale_detect,
    create_californium_agent,
    prepare_graph_data,
    save_californium_model,
    load_californium_model
)

__all__ = [
    'CaliforniumTGN',
    'QIRLAgent', 
    'californium_whale_detect',
    'create_californium_agent',
    'prepare_graph_data',
    'save_californium_model',
    'load_californium_model'
]