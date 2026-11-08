# Noname Agent architecture

The runtime is a ReAct loop with four collaborating layers:

1. **Session layer** (`session.py`) -- one JSON document per conversation
   window. Windows are fully isolated; a user can keep several open.
2. **Memory layer** (`memory.py`) -- decides what goes into the prompt each
   turn and compresses old turns into a running summary.
3. **Tool layer** (`tools/`) -- a registry of schema-described tools; the LLM
   chooses which to call.
4. **Runtime layer** (`agent.py`) -- the loop: think, act, observe, repeat.

Key invariants:

- The context for a turn is a pure function of session state plus config.
- Tool errors are returned to the model as `ERROR: ...`, never raised.
- A turn can never crash the process: every failure path yields an answer.
