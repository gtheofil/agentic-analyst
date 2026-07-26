import pytest
from agentic_analyst.tools.calc import calc, CalcError


def test_basic_arithmetic():
    assert calc("(1200*0.23)/12") == pytest.approx(23.0)


@pytest.mark.parametrize("expr", [
    "__import__('os').system('ls')",  # Call node
    "__import__",                     # bare Name
    "open('/etc/passwd')",            # Call node
    "1 + foo",                        # Name node
    "[1, 2, 3]",                      # List node
    "True + 1",                       # bool literal
])
def test_rejects_hostile(expr):
    with pytest.raises(CalcError):
        calc(expr)