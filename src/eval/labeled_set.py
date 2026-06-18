"""
My labeled evaluation set. Without it I can't report Precision/Recall/MRR/nDCG,
so I start building it early. I grow this to 50-200 examples and include a few
questions the corpus can't answer, to check that the system refuses instead of
hallucinating.
"""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class LabeledQuery:
    question: str
    relevant_ids: list[str] = field(default_factory=list)  # empty => should refuse
    note: str = ""

    @property
    def should_refuse(self) -> bool:
        return len(self.relevant_ids) == 0


LABELED_SET: list[LabeledQuery] = [
    LabeledQuery(
        question="What did the Court hold about warnings before custodial interrogation?",
        relevant_ids=[],  # TODO: Miranda v. Arizona chunk ids
        note="Expect Miranda v. Arizona.",
    ),
    LabeledQuery(
        question="On what basis did the Court recognize a right to counsel for indigent defendants?",
        relevant_ids=[],  # TODO: Gideon v. Wainwright
        note="Expect Gideon v. Wainwright.",
    ),
    LabeledQuery(
        question="What is the airspeed velocity of an unladen swallow?",
        relevant_ids=[],  # not in any opinion -> must refuse
        note="Negative control.",
    ),
]