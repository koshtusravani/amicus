"""
My labeled evaluation set for retrieval.

I label relevance at the case (cluster) level, not the individual chunk: each
landmark question is answered by one case's holding, and cluster ids stay stable
when I re-tune chunking while chunk ids do not. So the metric asks whether
retrieval surfaced a chunk from the right case near the top.

Questions with no relevant_cases are negative controls for the refusal eval, not
the retrieval metric.
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field

from .. import config


@dataclass
class LabeledQuery:
    question: str
    relevant_cases: list[str] = field(default_factory=list)
    note: str = ""

    @property
    def should_refuse(self) -> bool:
        return len(self.relevant_cases) == 0


LABELED_SET: list[LabeledQuery] = [
    LabeledQuery("What warnings must police give before a custodial interrogation?", ["Miranda"]),
    LabeledQuery("When can a police officer stop and frisk someone without a warrant?", ["Terry v. Ohio"]),
    LabeledQuery("What evidence must the prosecution disclose to the defense?", ["Brady v. Maryland"]),
    LabeledQuery("What test decides whether a defendant had ineffective assistance of counsel?", ["Strickland"]),
    LabeledQuery("What pleading standard did the Court set for antitrust conspiracy claims?", ["Twombly"]),
    LabeledQuery("Does the plausibility pleading standard apply to all civil cases?", ["Iqbal"]),
    LabeledQuery("Can a municipality be held liable under Section 1983 for its official policies?", ["Monell"]),
    LabeledQuery("What is the standard for reviewing whether trial evidence was sufficient to convict?", ["Jackson v. Virginia"]),
    LabeledQuery("How can a plaintiff prove employment discrimination with circumstantial evidence?", ["McDonnell Douglas"]),
    LabeledQuery("What must appointed appellate counsel do if they think an appeal is frivolous?", ["Anders"]),
    LabeledQuery("How should courts read pleadings filed by prisoners representing themselves?", ["Erickson"]),
    LabeledQuery("Who bears the burden on summary judgment when the other side lacks proof?", ["Celotex"]),
    LabeledQuery("What counts as a genuine dispute of material fact at summary judgment?", ["Anderson v. Liberty Lobby"]),
    LabeledQuery("What must an antitrust plaintiff show to survive summary judgment on a conspiracy claim?", ["Matsushita"]),
    LabeledQuery("When is a certificate of appealability needed to appeal a habeas denial?", ["Slack"]),
    LabeledQuery("What is the airspeed velocity of an unladen swallow?", [], "not in corpus"),
    LabeledQuery("What did the Court hold on a constitutional right to same-sex marriage?", [], "Obergefell, not in corpus"),
]


def case_to_cluster() -> dict[str, str]:
    mapping: dict[str, str] = {}
    for path in config.RAW_DIR.glob("*.json"):
        rec = json.loads(path.read_text(encoding="utf-8"))
        mapping[rec["case_name"]] = str(rec["cluster_id"])
    return mapping


def resolve_relevant(q: LabeledQuery, mapping: dict[str, str]) -> set[str]:
    out: set[str] = set()
    for sub in q.relevant_cases:
        for name, cid in mapping.items():
            if sub.lower() in name.lower():
                out.add(cid)
    return out