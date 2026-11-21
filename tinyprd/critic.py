"""PRD-kritikeren: giver en PRD point fra 0 til 10 på syv områder.

Kritikeren er regelbaseret (ingen træning), så man kan læse præcis hvorfor
en PRD får sine point. Reglerne kommer fra principperne i
docs/PRD_PRINCIPPER.md:

    problem      står problemet klart, og uden at foreslå en løsning?
    outcome      er det ønskede resultat formuleret?
    metrics      er succesmålene målbare (median, procent, antal ...)?
    scope        er der en afgrænset liste over hvad der er med?
    non-goals    står det hvad vi *ikke* bygger?
    questions    er der åbne spørgsmål, også et teknisk et?
    brevity      kan det stå på én side?

Den finder også de "klassiske fejl", fx ordet "just" og løsninger
gemt i problemformuleringen.

Den bruges på to måder:
    * som evaluering af modellerne (`tinyprd evaluate`)
    * som værktøj på dine egne PRD'er (`tinyprd critique min_prd.md`)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .prd import parse_sections

# Ord der afslører en løsning. De hører ikke hjemme i problemformuleringen.
SOLUTION_WORDS = {
    "dashboard", "button", "slack", "email", "notification", "notifications", "app",
    "integration", "modal", "chart", "charts", "bot", "ai", "widget", "popup", "page",
}
# Ord der gør et succesmål målbart.
MEASURABLE_WORDS = {
    "median", "percent", "number", "rate", "time", "hours", "days", "count",
    "average", "retention", "per", "within", "share", "ratio",
}
# Ord der tyder på et teknisk spørgsmål til engineering.
TECHNICAL_WORDS = {"data", "api", "engineering", "limits", "prototype", "expose", "technical", "sync"}

MAX_WORDS_ONE_PAGER = 400


@dataclass
class Critique:
    scores: dict[str, float]
    feedback: list[str] = field(default_factory=list)

    @property
    def overall(self) -> float:
        return sum(self.scores.values()) / len(self.scores)

    def report(self) -> str:
        """Pæn tekst-rapport til terminalen."""
        lines = []
        for name, score in self.scores.items():
            bar = "█" * round(score) + "·" * (10 - round(score))
            lines.append(f"  {name:<10} {bar} {score:4.1f}/10")
        lines.append(f"  {'samlet':<10} {'':10} {self.overall:4.1f}/10")
        if self.feedback:
            lines.append("")
            lines.extend(f"  • {item}" for item in self.feedback)
        return "\n".join(lines)


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def _find(sections: dict[str, list[str]], *names: str) -> list[str]:
    """Første sektion der matcher et af navnene (fx 'non-goals' eller 'out of scope')."""
    for wanted in names:
        for title, lines in sections.items():
            if wanted in title.lower():
                return lines
    return []


def critique(text: str) -> Critique:
    """Bedøm en PRD skrevet i '## sektion'-format."""
    sections = parse_sections(text)
    scores: dict[str, float] = {}
    feedback: list[str] = []

    # --- problem: findes, passende længde, ingen løsning gemt i den ---
    problem = " ".join(_find(sections, "problem"))
    problem_words = _words(problem)
    if not problem_words:
        scores["problem"] = 0.0
        feedback.append("Der mangler en problemformulering. Hvad er det vigtigste problem for brugeren?")
    else:
        score = 10.0
        if not 8 <= len(problem_words) <= 45:
            score -= 3
            feedback.append("Problemet bør være én til to stærke sætninger (ca. 8-45 ord).")
        leaked = sorted(SOLUTION_WORDS & set(problem_words))
        if leaked:
            score -= 4
            feedback.append(f"Problemet nævner en løsning ({', '.join(leaked)}). Beskriv problemet, ikke løsningen.")
        scores["problem"] = max(score, 0.0)

    # --- outcome ---
    outcome = _words(" ".join(_find(sections, "outcome", "goal")))
    scores["outcome"] = 10.0 if len(outcome) >= 4 else 0.0
    if not outcome:
        feedback.append("Hvad skal være anderledes for brugeren, når vi er færdige? Skriv et ønsket resultat.")

    # --- success metrics: mindst 2, og de skal være målbare ---
    metrics = _find(sections, "metric", "success")
    measurable = [m for m in metrics if MEASURABLE_WORDS & set(_words(m))]
    scores["metrics"] = min(10.0, 10.0 * len(measurable) / 3)
    if len(measurable) < 2:
        feedback.append("Tilføj målbare succesmål (fx 'median tid til ...' eller 'procent af ...').")

    # --- scope ---
    scope = _find(sections, "scope")
    scores["scope"] = min(10.0, 10.0 * len(scope) / 3)
    if len(scope) < 2:
        feedback.append("Afgræns hvad der er med i første version (mindst to punkter).")

    # --- non-goals ---
    non_goals = _find(sections, "non-goal", "out of scope", "not doing")
    scores["non-goals"] = min(10.0, 5.0 * len(non_goals))
    if not non_goals:
        feedback.append("Skriv hvad I *ikke* bygger (non-goals). Ellers kryber scope.")

    # --- open questions: med spørgsmålstegn, og gerne et teknisk ---
    questions = [q for q in _find(sections, "question") if q.rstrip().endswith("?")]
    score = min(7.0, 3.5 * len(questions))
    if any(TECHNICAL_WORDS & set(_words(q)) for q in questions):
        score += 3.0
    else:
        feedback.append("Stil mindst ét teknisk spørgsmål til engineering tidligt.")
    scores["questions"] = score

    # --- brevity: én side ---
    all_words = _words(text)
    if len(all_words) <= MAX_WORDS_ONE_PAGER:
        scores["brevity"] = 10.0
    else:
        scores["brevity"] = max(0.0, 10.0 - (len(all_words) - MAX_WORDS_ONE_PAGER) / 40)
        feedback.append(f"{len(all_words)} ord er mere end én side. Kort tvinger til klarhed.")

    # --- klassiske fejl ---
    if "just" in all_words:
        feedback.append("Undgå ordet 'just'. Det undervurderer arbejdet for engineering.")
    if not _find(sections, "appetite", "timeline", "time"):
        feedback.append("Hvor meget tid er problemet værd? Tilføj en 'appetite'.")

    return Critique(scores=scores, feedback=feedback)
