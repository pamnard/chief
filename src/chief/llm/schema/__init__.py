"""Wire **schemas** (vendor-neutral request/response shapes).

Each submodule handles one :class:`~chief.llm.types.ChatWireFormat` value: URL/body construction and
response parsing only. HTTP and :class:`~chief.brain.Brain` live under ``chief.llm.providers``; shared
planner wording lives in :mod:`chief.llm.planner_context`.
"""
