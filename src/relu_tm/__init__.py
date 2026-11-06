"""relu-tm: a fixed width-4 ReLU network that simulates a Turing machine.

>>> from relu_tm import TuringMachine, build_network, run
>>> tm = TuringMachine.from_standard_format("1RB1LB_1LA1RZ")   # BB(2)
>>> result = run(tm)            # emulator vs. network, exact arithmetic
>>> result.steps, result.agree
(6, True)
>>> build_network(tm).width
4
"""

from .automata import DFA, DPDA, EPS
from .encoding import DecodeError, Layout, decode, encode
from .formats import FormatError, MachineFile, load
from .machine import Configuration, Machine, TuringMachine, accepts
from .network import CompiledNetwork, build_network, build_program
from .run import RunResult, run

__all__ = [
    "DFA",
    "DPDA",
    "EPS",
    "CompiledNetwork",
    "Configuration",
    "DecodeError",
    "FormatError",
    "Layout",
    "Machine",
    "MachineFile",
    "RunResult",
    "TuringMachine",
    "accepts",
    "build_network",
    "build_program",
    "decode",
    "encode",
    "load",
    "run",
]

__version__ = "0.1.0"
