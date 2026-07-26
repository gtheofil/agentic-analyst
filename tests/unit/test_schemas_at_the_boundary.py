"""Every schema the model is asked to fill, checked against Gemini's converter.

NOTES.md #1 and #2: the most expensive bug in this project was a schema the
provider silently could not accept. Every call died *inside* the SDK, before a
request was sent, and the whole mocked suite stayed green — because mocking a
boundary stops you testing the boundary.

That bug was fixed with one regression test for one schema. A test that covers
one schema protects one schema, so this covers all of them, and the collection
is derived from `state.py` rather than typed out, so a schema added later is
picked up here without anyone remembering to add it.

These reach into a private SDK module on purpose: the conversion is the contract
under test, and there is no public entry point to it that does not also make a
billable request.
"""

import pytest
from google.genai import _transformers
from pydantic import BaseModel

from agentic_analyst.agents.planner import Plan
from agentic_analyst.agents.researcher import NextStep
from agentic_analyst.memory.episodic import RunSummary
from agentic_analyst.state import Critique

# Every model passed as `schema=` to `llm.call` anywhere in the package.
SCHEMAS: list[type[BaseModel]] = [Plan, NextStep, Critique, RunSummary]


@pytest.mark.parametrize("schema", SCHEMAS, ids=lambda s: s.__name__)
def test_schema_is_accepted_by_geminis_converter(schema: type[BaseModel]) -> None:
    _transformers.t_schema(None, schema)  # must not raise


def test_critique_nests_an_object_and_a_list_of_objects() -> None:
    """`Critique` is the most structurally complex schema in the system.

    `scores` is a nested object and `required_fixes` is a list of them, both of
    which the converter has to flatten. The parametrised test above would pass
    on a `Critique` that had quietly lost its structure, so pin the shape too.
    """
    schema = Critique.model_json_schema()
    props = schema["properties"]

    assert props["scores"]["$ref"].endswith("Scores")
    assert props["required_fixes"]["items"]["$ref"].endswith("Fix")


def test_no_schema_uses_a_discriminated_union() -> None:
    """The original bug, generalised.

    `Field(discriminator=...)` serialises to `oneOf` + `discriminator`, neither
    of which Gemini accepts. It is the idiomatic Pydantic thing to reach for and
    it fails at a distance, so catch it on any schema, not just `NextStep`.
    """
    for schema in SCHEMAS:
        rendered = str(schema.model_json_schema())
        assert "discriminator" not in rendered, f"{schema.__name__} uses a discriminator"


def test_scores_bounds_survive_conversion() -> None:
    """The 0-10 range is the only thing stopping a model returning 47/10 and
    sailing over a pass rule defined as a mean."""
    schema = Critique.model_json_schema()
    scores = schema["$defs"]["Scores"]["properties"]

    for dimension in ("groundedness", "structure", "actionability"):
        assert scores[dimension]["minimum"] == 0
        assert scores[dimension]["maximum"] == 10
