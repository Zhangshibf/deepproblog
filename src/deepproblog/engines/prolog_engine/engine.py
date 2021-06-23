from pathlib import Path
from typing import Optional

from pyswip import Prolog

from problog.engine import GenericEngine
from problog.formula import LogicFormula
from problog.logic import Term, Var
from .swi_program import SWIProgram
from ...heuristics import Heuristic


root = Path(__file__).parent


class PrologEvaluationException(Exception):
    """Exception from PrologEngine for unexpected result when evaluating a query."""


class PrologEngine(GenericEngine):
    def __init__(
        self,
        k,
        heuristic: Optional[Heuristic],
        exploration: bool,
        timeout=None,
        ignore_timeout=False,
    ):
        super().__init__()
        self.k = k
        self.heuristic = heuristic
        self.prolog = Prolog()
        self.timeout = timeout
        self.ignore_timeout = ignore_timeout
        self.exploration = exploration
        self.prolog.consult(str(root / "prolog_files" / "engine_heap.pl"))

    def prepare(self, db):
        program = SWIProgram(db, heuristic=self.heuristic)
        return program

    def ground(self, sp, term, target=None, label=None, *args, **kwargs):
        if type(sp) != SWIProgram:
            sp = self.prepare(sp)
        if target is None:
            target = LogicFormula(keep_all=True)
        proofs = self.get_proofs(str(term), sp)
        result = sp.add_proof_trees(proofs, target=target, label=label)
        return result

    def ground_all(self, sp, target=None, queries=None, evidence=None, *args, **kwargs):
        if type(sp) != SWIProgram:
            sp = self.prepare(sp)
        if target is None:
            target = LogicFormula()
        if queries is None:
            queries = [
                q[0].args[0]
                for q in self.ground(
                    sp, Term("query", Var("X")), *args, **kwargs
                ).queries()
            ]
        for q in queries:
            self.ground(sp, q, target, *args, **kwargs)
        return target

    def get_proofs(self, q, program: SWIProgram, profile=0):
        exploration = "true" if self.exploration else "false"
        query_str = "prove({},{},Proofs,{},{})".format(
            q, self.k, self.heuristic.name, exploration
        )
        if self.timeout is not None:
            query_str = "call_with_time_limit({},{})".format(self.timeout, query_str)
        try:
            res = program.query(query_str, profile=profile)
        except TimeoutError:
            if self.ignore_timeout:
                return []
            else:
                raise TimeoutError()
        except OverflowError:
            return []
        if len(res) != 1:
            raise PrologEvaluationException(
                f"Expected exactly 1 result, got {len(res)}"
            )
        return res[0]["Proofs"]
