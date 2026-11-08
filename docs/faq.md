# FAQ

**Does this use an agent framework?**
No. The loop, the parser, the tool registry, the memory manager and the
session store are all implemented in this repository. Only the LLM HTTP call
is delegated, and that is a ~60-line `urllib` client.

**Where do API keys live?**
Only in environment variables (`OPENAI_API_KEY`, `OPENAI_BASE_URL`,
`OPENAI_MODEL`). No key is ever written to the repository or to logs.

**How do I run it without a key?**
Omit the key and the CLI falls back to `MockLLM`, which replays a deterministic
script. All tests run offline this way.
