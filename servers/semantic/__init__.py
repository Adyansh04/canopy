"""The semantic server's parts, run by servers/semantic_server.py.

config       DEFAULTS, and the config file, flags and overrides over them
detection    YOLOE-26 instance masks
embedding    SigLIP 2 image and text embeddings
prompts      what a describer is asked, and its answer parsed
describers   the describer contract, a local VLM, the fallback and the chain
gemini       Gemini's REST API under per-model free-tier caps
server       the request handlers, the backend registries and the ZMQ loop
"""
