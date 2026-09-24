"""LLM prompts for the memory system."""

from __future__ import annotations

EXTRACTION_SYSTEM_PROMPT = """You are a memory extraction assistant.
Analyze a conversation and extract memorable information.
Output ONLY valid JSON. Do not add commentary."""

EXTRACTION_PROMPT = """Analyze this conversation and extract memorable information.

For each memory, output:
- type: one of "semantic", "episodic", "procedural"
  - "semantic" — stable facts, preferences, knowledge (e.g. "项目目录在 ~/project")
  - "episodic" — specific events, decisions, conversations (e.g. "用户昨天问了GRPO的padding问题")
  - "procedural" — learned patterns, workflows, habits (e.g. "用户习惯先看文档再改代码")
- content: concise one-line summary (this is what gets shown in context)
- detail: full detail (optional, for episodic and procedural memories with important specifics)
- category: one of [user_pref, project, technical, decision, event, error, pattern, environment]
- importance: 0.0 to 1.0 (how likely this will be needed in future conversations)

Focus on:
- User preferences and habits (language, coding style, tool preferences)
- Project-specific knowledge (tech stack, architecture decisions, deployment info)
- Key decisions and their rationale
- Errors encountered and how they were resolved
- Repeated patterns (if the user does X multiple times)
- Environment details (OS, tools, versions)

IGNORE:
- Transient task details that won't recur
- Generic programming knowledge the model already has
- Anything already covered by the existing memories listed below

<existing-memories>
{existing_memories}
</existing-memories>

<conversation>
{conversation}
</conversation>

Output format (JSON array) — include AT LEAST ONE example of each type if the
conversation contains a mix of facts, events, and patterns:
[
  {{
    "type": "semantic",
    "content": "用户偏好使用 Python 类型注解",
    "detail": null,
    "category": "user_pref",
    "importance": 0.7
  }},
  {{
    "type": "episodic",
    "content": "用户指出了飞书SDK消息处理的时序冲突问题",
    "detail": "feishu.py第103行每条消息独立创建task并发处理，/forget命令和后续消息可能因调度顺序导致读取旧数据",
    "category": "technical",
    "importance": 0.8
  }},
  {{
    "type": "procedural",
    "content": "用户习惯先查看项目结构再修改代码",
    "detail": null,
    "category": "pattern",
    "importance": 0.6
  }}
]
If nothing memorable, output: []"""

CONSOLIDATION_SYSTEM_PROMPT = """You are a memory consolidation assistant.
Review episodic memories and extract higher-level insights.
Output ONLY valid JSON. Do not add commentary."""

CONSOLIDATION_PROMPT = """Review these episodic memories and consolidate them.

Your tasks:
1. EXTRACT: Find recurring patterns → create semantic memories
2. MERGE: Find similar/overlapping memories → produce merged versions
3. GENERALIZE: Find specific instances of general rules → create abstract versions

<episodic-memories>
{episodic_memories}
</episodic-memories>

<existing-semantic-memories>
{existing_semantic}
</existing-semantic-memories>

Output format (JSON):
{{
  "new_semantic": [
    {{
      "content": "用户的项目使用 fly.io 部署",
      "category": "project",
      "importance": 0.8,
      "source_episodes": ["ep_id_1", "ep_id_2"]
    }}
  ],
  "merge_groups": [
    {{
      "memory_ids": ["id1", "id2"],
      "merged_content": "合并后的内容",
      "merged_detail": "合并后的细节（可选）"
    }}
  ],
  "strengthen": ["id_of_well_confirmed_memory"],
  "forget": ["id_of_irrelevant_memory"]
}}

If nothing to consolidate, output: {{"new_semantic": [], "merge_groups": [], "strengthen": [], "forget": []}}"""

DECAY_SYSTEM_PROMPT = """You are a memory curator. Decide which memories are no longer relevant.
Output ONLY valid JSON."""

DECAY_PROMPT = """Review these low-strength memories and decide which should be forgotten.

Consider:
- Was this a one-time event with no future relevance?
- Has this information been superseded by newer memories?
- Is this too generic to be useful?

<memories>
{memories}
</memories>

Output format (JSON):
{{
  "forget": ["id1", "id2"],
  "keep": ["id3"],
  "reason": "brief explanation"
}}"""


CURATION_SYSTEM_PROMPT = """You are a memory curator for a long-running assistant.
You keep the memory store truthful, non-contradictory and free of anything that can be
read directly from the environment. Output ONLY valid JSON."""


CURATION_PROMPT = """Curate the assistant's long-term memories.

Your tasks:
1. RESOLVE CONFLICTS: several memories about the same topic must not contradict each
   other. Keep the newest version (compare the [date] prefix), supersede the old one.
2. DROP STALE: anything that has changed since it was recorded must be corrected or removed.
3. DROP ENVIRONMENT-DERIVABLE: facts the assistant can read from its own environment
   (active model, config switches, whether a service is running, file layout it can just
   look at, language/runtime versions) must NOT live in memory. Remove them.
   EXCEPTION — KEEP TECHNICAL KNOWLEDGE: API/protocol behaviour, field names, vendor
   quirks, sandbox/tooling gotchas and other things that were *learned the hard way*
   are NOT environment-readable. Keep them, even when they mention a model, a path or
   a version. Only drop a technical memory when the environment facts show it is now
   WRONG (then correct it instead of deleting it).
4. VERIFY: when the environment facts below contradict a memory, the environment wins.
5. MERGE: overlapping/redundant memories → one entry.
Never invent facts. When unsure, keep the memory.

<memories>
{memories}
</memories>

<environment-facts>
{env_facts}
</environment-facts>

Output format (JSON):
{{
  "update": [
    {{"id": "mem_id", "content": "corrected content", "detail": null, "reason": "为什么改"}}
  ],
  "forget": [
    {{"id": "mem_id", "reason": "过时 / 环境可读 / 与事实冲突"}}
  ],
  "merge_groups": [
    {{"memory_ids": ["id1", "id2"], "merged_content": "合并后的内容", "reason": "重复"}}
  ],
  "new_semantic": [
    {{"content": "一条应补上的新事实", "category": "", "importance": 0.6, "reason": "为什么"}}
  ]
}}

If nothing needs changing, output:
{{"update": [], "forget": [], "merge_groups": [], "new_semantic": []}}"""


DECAY_SYSTEM_PROMPT = """You are a memory curator. Decide which memories are no longer relevant.
Output ONLY valid JSON."""
