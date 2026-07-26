"""Memory: what the agent remembers between runs.

Two different things, deliberately kept apart:

* `episodic` — what the agent *earned*. Summaries of its own past runs, stored
  in Chroma and retrieved by similarity to the incoming brief.
* `preferences` — what a *human* told it. Standing style instructions read from
  `config/prefs.yaml`. No model, no vector store, just config.

Conflating them is how a user's explicit instruction ends up competing with a
lesson the agent inferred from a run three weeks ago.
"""
