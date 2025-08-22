# consensus/schemas.py
"""
JSON schemas for strict validation
"""

# Single token response schema with agents
SINGLE_SCHEMA = {
    "type": "object",
    "properties": {
        "token_id": {"type": "string"},
        "agents": {
            "type": "object",
            "properties": {
                "Analyzer": {
                    "type": "object",
                    "properties": {
                        "action_probs": {
                            "type": "object",
                            "properties": {
                                "BUY": {"type": "number"},
                                "HOLD": {"type": "number"},
                                "AVOID": {"type": "number"},
                                "ABSTAIN": {"type": "number"}
                            },
                            "required": ["BUY", "HOLD", "AVOID", "ABSTAIN"],
                            "additionalProperties": False
                        },
                        "uncertainty": {
                            "type": "object",
                            "properties": {
                                "epistemic": {"type": "number"},
                                "aleatoric": {"type": "number"}
                            },
                            "required": ["epistemic", "aleatoric"],
                            "additionalProperties": False
                        },
                        "evidence": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "direction": {"type": "string", "enum": ["pro", "con", "neutral"]},
                                    "strength": {"type": "number"}
                                },
                                "required": ["name", "direction", "strength"],
                                "additionalProperties": False
                            },
                            "minItems": 3,
                            "maxItems": 3
                        },
                        "rationale": {"type": "string"},
                        "calibration_hint": {
                            "type": "object",
                            "properties": {
                                "reliability": {"type": "number"},
                                "expected_ttft_mins": {"type": "number"}
                            },
                            "required": ["reliability", "expected_ttft_mins"],
                            "additionalProperties": False
                        }
                    },
                    "required": ["action_probs", "uncertainty", "evidence", "rationale", "calibration_hint"],
                    "additionalProperties": False
                },
                "Reasoner": {
                    "type": "object",
                    "properties": {
                        "action_probs": {
                            "type": "object",
                            "properties": {
                                "BUY": {"type": "number"},
                                "HOLD": {"type": "number"},
                                "AVOID": {"type": "number"},
                                "ABSTAIN": {"type": "number"}
                            },
                            "required": ["BUY", "HOLD", "AVOID", "ABSTAIN"],
                            "additionalProperties": False
                        },
                        "uncertainty": {
                            "type": "object",
                            "properties": {
                                "epistemic": {"type": "number"},
                                "aleatoric": {"type": "number"}
                            },
                            "required": ["epistemic", "aleatoric"],
                            "additionalProperties": False
                        },
                        "evidence": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "direction": {"type": "string", "enum": ["pro", "con", "neutral"]},
                                    "strength": {"type": "number"}
                                },
                                "required": ["name", "direction", "strength"],
                                "additionalProperties": False
                            },
                            "minItems": 3,
                            "maxItems": 3
                        },
                        "rationale": {"type": "string"},
                        "calibration_hint": {
                            "type": "object",
                            "properties": {
                                "reliability": {"type": "number"},
                                "expected_ttft_mins": {"type": "number"}
                            },
                            "required": ["reliability", "expected_ttft_mins"],
                            "additionalProperties": False
                        }
                    },
                    "required": ["action_probs", "uncertainty", "evidence", "rationale", "calibration_hint"],
                    "additionalProperties": False
                },
                "Voter": {
                    "type": "object",
                    "properties": {
                        "action_probs": {
                            "type": "object",
                            "properties": {
                                "BUY": {"type": "number"},
                                "HOLD": {"type": "number"},
                                "AVOID": {"type": "number"},
                                "ABSTAIN": {"type": "number"}
                            },
                            "required": ["BUY", "HOLD", "AVOID", "ABSTAIN"],
                            "additionalProperties": False
                        },
                        "uncertainty": {
                            "type": "object",
                            "properties": {
                                "epistemic": {"type": "number"},
                                "aleatoric": {"type": "number"}
                            },
                            "required": ["epistemic", "aleatoric"],
                            "additionalProperties": False
                        },
                        "evidence": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "direction": {"type": "string", "enum": ["pro", "con", "neutral"]},
                                    "strength": {"type": "number"}
                                },
                                "required": ["name", "direction", "strength"],
                                "additionalProperties": False
                            },
                            "minItems": 3,
                            "maxItems": 3
                        },
                        "rationale": {"type": "string"},
                        "calibration_hint": {
                            "type": "object",
                            "properties": {
                                "reliability": {"type": "number"},
                                "expected_ttft_mins": {"type": "number"}
                            },
                            "required": ["reliability", "expected_ttft_mins"],
                            "additionalProperties": False
                        }
                    },
                    "required": ["action_probs", "uncertainty", "evidence", "rationale", "calibration_hint"],
                    "additionalProperties": False
                },
                "Debater": {
                    "type": "object",
                    "properties": {
                        "action_probs": {
                            "type": "object",
                            "properties": {
                                "BUY": {"type": "number"},
                                "HOLD": {"type": "number"},
                                "AVOID": {"type": "number"},
                                "ABSTAIN": {"type": "number"}
                            },
                            "required": ["BUY", "HOLD", "AVOID", "ABSTAIN"],
                            "additionalProperties": False
                        },
                        "uncertainty": {
                            "type": "object",
                            "properties": {
                                "epistemic": {"type": "number"},
                                "aleatoric": {"type": "number"}
                            },
                            "required": ["epistemic", "aleatoric"],
                            "additionalProperties": False
                        },
                        "evidence": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "direction": {"type": "string", "enum": ["pro", "con", "neutral"]},
                                    "strength": {"type": "number"}
                                },
                                "required": ["name", "direction", "strength"],
                                "additionalProperties": False
                            },
                            "minItems": 3,
                            "maxItems": 3
                        },
                        "rationale": {"type": "string"},
                        "calibration_hint": {
                            "type": "object",
                            "properties": {
                                "reliability": {"type": "number"},
                                "expected_ttft_mins": {"type": "number"}
                            },
                            "required": ["reliability", "expected_ttft_mins"],
                            "additionalProperties": False
                        }
                    },
                    "required": ["action_probs", "uncertainty", "evidence", "rationale", "calibration_hint"],
                    "additionalProperties": False
                }
            },
            "required": ["Analyzer", "Reasoner", "Voter", "Debater"],
            "additionalProperties": False
        }
    },
    "required": ["token_id", "agents"],
    "additionalProperties": False
}

# Batch schema for multiple tokens
BATCH_SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "object",
            "patternProperties": {
                "^[A-Z0-9]+USDT$": {
                    "type": "object",
                    "properties": {
                        "agents": {
                            "type": "object",
                            "properties": {
                                "Analyzer": {
                                    "type": "object",
                                    "properties": {
                                        "action_probs": {
                                            "type": "object",
                                            "properties": {
                                                "BUY": {"type": "number"},
                                                "HOLD": {"type": "number"},
                                                "AVOID": {"type": "number"},
                                                "ABSTAIN": {"type": "number"}
                                            },
                                            "required": ["BUY", "HOLD", "AVOID", "ABSTAIN"],
                                            "additionalProperties": False
                                        },
                                        "uncertainty": {
                                            "type": "object",
                                            "properties": {
                                                "epistemic": {"type": "number"},
                                                "aleatoric": {"type": "number"}
                                            },
                                            "required": ["epistemic", "aleatoric"],
                                            "additionalProperties": False
                                        },
                                        "evidence": {
                                            "type": "array",
                                            "items": {
                                                "type": "object",
                                                "properties": {
                                                    "name": {"type": "string"},
                                                    "direction": {"type": "string", "enum": ["pro", "con", "neutral"]},
                                                    "strength": {"type": "number"}
                                                },
                                                "required": ["name", "direction", "strength"],
                                                "additionalProperties": False
                                            },
                                            "minItems": 3,
                                            "maxItems": 3
                                        },
                                        "rationale": {"type": "string"},
                                        "calibration_hint": {
                                            "type": "object",
                                            "properties": {
                                                "reliability": {"type": "number"},
                                                "expected_ttft_mins": {"type": "number"}
                                            },
                                            "required": ["reliability", "expected_ttft_mins"],
                                            "additionalProperties": False
                                        }
                                    },
                                    "required": ["action_probs", "uncertainty", "evidence", "rationale", "calibration_hint"],
                                    "additionalProperties": False
                                },
                                "Reasoner": {"$ref": "#/properties/items/patternProperties/^[A-Z0-9]+USDT$/properties/agents/properties/Analyzer"},
                                "Voter": {"$ref": "#/properties/items/patternProperties/^[A-Z0-9]+USDT$/properties/agents/properties/Analyzer"},
                                "Debater": {"$ref": "#/properties/items/patternProperties/^[A-Z0-9]+USDT$/properties/agents/properties/Analyzer"}
                            },
                            "required": ["Analyzer", "Reasoner", "Voter", "Debater"],
                            "additionalProperties": False
                        }
                    },
                    "required": ["agents"],
                    "additionalProperties": False
                }
            },
            "additionalProperties": False
        }
    },
    "required": ["items"],
    "additionalProperties": False
}