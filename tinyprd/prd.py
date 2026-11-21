"""PRD-posten: én struktureret produktspecifikation, og hvordan den bliver til tekst.

Hvert træningseksempel er en `PrdRecord`. Den gemmer ikke kun det færdige
dokument, men også *mellemregningerne* (problem, resultat, mål, scope ...),
så modellen lærer hele kæden:

    rodet idé -> intent -> problem -> brugere -> resultat -> succesmål
              -> scope -> non-goals -> tidsbudget -> åbne spørgsmål

Layoutet følger "one-pager"-principperne i docs/PRD_PRINCIPPER.md: problemet
først og uden løsning, målbare succesmål, et afgrænset scope med tydelige
non-goals, et tidsbudget (appetite) og åbne spørgsmål, der inviterer
engineering og design ind i samtalen.

`render()` laver en post om til den tekst, modellen trænes til at skrive.
`parse_sections()` gør det modsatte (bruges af kritikeren og kommandolinjen).

Bemærk: selve PRD-teksten er på engelsk, ligesom de fleste PRD'er i
tech-branchen. Dokumentationen er på dansk.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

# Rækkefølgen af sektioner i hver PRD. En fast rækkefølge giver modellen et
# forudsigeligt skelet, så den kan bruge sine kræfter på *indholdet*.
SECTION_ORDER = (
    "intent",
    "problem",
    "users",
    "outcome",
    "success metrics",
    "scope",
    "non-goals",
    "appetite",
    "open questions",
)

# Sektioner der skrives som punktlister i stedet for én sætning.
LIST_SECTIONS = {"success metrics", "scope", "non-goals", "open questions"}


@dataclass
class PrdRecord:
    """Én produktidé og dens formede specifikation."""

    idea: str  # det rodede input, skrevet som et menneske ville skrive det
    intent: str  # en etiket på 2-5 ord, fx "pull request wait visibility"
    problem: str
    users: str
    outcome: str
    success_metrics: list[str] = field(default_factory=list)
    scope: list[str] = field(default_factory=list)
    non_goals: list[str] = field(default_factory=list)
    appetite: str = ""  # hvor meget tid problemet er værd, fx "two weeks"
    open_questions: list[str] = field(default_factory=list)

    # Bogføring til eksperimenter. Modellen ser aldrig disse felter.
    domain: str = ""
    pattern: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "PrdRecord":
        return cls(**data)

    def render(self) -> str:
        """Skriv posten som den PRD-tekst, modellen skal lære at skrive."""
        values = {
            "intent": self.intent,
            "problem": self.problem,
            "users": self.users,
            "outcome": self.outcome,
            "success metrics": self.success_metrics,
            "scope": self.scope,
            "non-goals": self.non_goals,
            "appetite": self.appetite,
            "open questions": self.open_questions,
        }
        lines: list[str] = []
        for name in SECTION_ORDER:
            lines.append(f"## {name}")
            value = values[name]
            if name in LIST_SECTIONS:
                lines.extend(f"- {item}" for item in value)
            else:
                lines.append(value)
        return "\n".join(lines)


def parse_sections(text: str) -> dict[str, list[str]]:
    """Del PRD-tekst op i {sektionsnavn: linjer}.

    Virker også på modellens output, så den er tilgivende: ukendte
    overskrifter beholdes, tekst før første overskrift ignoreres, og
    punkttegn ("- ") fjernes.
    """
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("##"):
            current = line.lstrip("#").strip()
            sections.setdefault(current, [])
        elif current is not None:
            sections[current].append(line.removeprefix("-").strip())
    return sections
