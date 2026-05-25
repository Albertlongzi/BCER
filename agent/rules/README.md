# Rules

Hard/soft rule checks live here to steer tool calls toward safe, consistent pipelines.
These are used as LLM coaching signals (especially in `autofix_mode=coach`) and
cover both prostate and brain workflows (e.g., modality presence, registration
sanity, ROI prerequisites).
