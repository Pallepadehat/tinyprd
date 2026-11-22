"""Syntetisk PRD-datasæt: rodede idéer parret med velformede one-pagers.

Hvorfor syntetisk?
    Der findes ikke et offentligt datasæt med "rodet idé -> god PRD". Og hvis
    vi kopierede rigtige PRD'er, ville modellen lære andres formuleringer.
    I stedet er *principperne* for en god one-pager (docs/PRD_PRINCIPPER.md)
    skrevet ind som skabeloner, og generatoren kombinerer dem til tusindvis
    af eksempler.

Sådan virker det
    Et træningseksempel er   DOMÆNE  x  MØNSTER  x  tilfældige valg.

    * Et DOMÆNE er et produktområde med dets mennesker og ting, fx
      code review: developers, pull requests, reviewers, github.
    * Et MØNSTER er en tilbagevendende slags produktproblem, fx "ting sidder
      fast og venter på nogen" eller "det er for svært at dele". Hvert mønster
      ved, hvordan en god PM ville forme det: problem, resultat, succesmål,
      scope, non-goals, tidsbudget og åbne spørgsmål, skrevet med {pladser}
      til domænets ord.
    * Tilfældige valg vælger formuleringer, delmængder af punkter, og den
      løsning personen nævner i sin idé ("maybe it could send slack messages").

    Idéen er bevidst løsnings-præget. PRD'en er bevidst problem-først.
    Det er netop den oversættelse, modellen skal lære.

Test af generalisering
    Nogle (domæne, mønster)-kombinationer vises aldrig under træning.
    Testsættet `test_unseen` spørger: kan modellen forme "deling" for
    *fakturaer*, når den kun har set deling for dashboards, tickets osv.?
    Det er det ærlige mål for, om den har lært struktur og ikke bare har
    lært tekst udenad.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

from .prd import PrdRecord

# --------------------------------------------------------------------------
# Domæner: et produktområdes navneord.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Domain:
    key: str
    area: str  # produktområdet, fx "code review"
    users: str  # primære brugere, flertal: "developers"
    user: str  # ental: "developer"
    obj: str  # det de arbejder med, ental: "pull request"
    objs: str  # flertal: "pull requests"
    peers: str  # de andre involverede: "reviewers"
    tool: str  # hvor arbejdet sker i dag: "github"


DOMAINS = [
    Domain("code", "code review", "developers", "developer", "pull request", "pull requests", "reviewers", "github"),
    Domain("analytics", "analytics", "analysts", "analyst", "dashboard", "dashboards", "coworkers", "the analytics app"),
    Domain("support", "customer support", "support agents", "support agent", "ticket", "tickets", "team leads", "the helpdesk"),
    Domain("billing", "billing", "finance managers", "finance manager", "invoice", "invoices", "approvers", "the billing system"),
    Domain("expenses", "expense management", "employees", "employee", "expense report", "expense reports", "managers", "the expense tool"),
    Domain("deploys", "release engineering", "engineers", "engineer", "deployment", "deployments", "on call engineers", "the ci pipeline"),
    Domain("hiring", "recruiting", "recruiters", "recruiter", "candidate profile", "candidate profiles", "hiring managers", "the applicant tracker"),
    Domain("design", "design", "designers", "designer", "design file", "design files", "product managers", "the design tool"),
    Domain("sales", "sales", "sales reps", "sales rep", "deal", "deals", "sales managers", "the crm"),
    Domain("shop", "ecommerce", "store owners", "store owner", "order", "orders", "warehouse staff", "the store admin"),
    Domain("clinic", "healthcare", "clinic staff", "clinic worker", "appointment", "appointments", "doctors", "the scheduling system"),
    Domain("school", "education", "teachers", "teacher", "assignment", "assignments", "students", "the learning platform"),
    Domain("docs", "documentation", "writers", "writer", "document", "documents", "editors", "the docs editor"),
    Domain("incidents", "incident response", "sre teams", "sre", "incident", "incidents", "stakeholders", "the incident tool"),
    Domain("marketing", "marketing", "marketers", "marketer", "campaign", "campaigns", "brand reviewers", "the marketing suite"),
    Domain("data", "data engineering", "data engineers", "data engineer", "pipeline", "pipelines", "data consumers", "the data platform"),
]

# --------------------------------------------------------------------------
# Mønstre: tilbagevendende produktproblemer, og hvordan en god PM former dem.
# Alle tekster må bruge {users} {user} {obj} {objs} {a_obj} {peers} {tool} {area}.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Pattern:
    key: str
    label: str  # sættes efter objektet og danner intent-linjen
    cores: tuple[str, ...]  # hvordan en person beskriver idéen (efter "something that ...")
    solutions: tuple[str, ...]  # løsninger folk springer til, som udsagnsled
    problem: tuple[str, ...]
    outcome: tuple[str, ...]
    metrics: tuple[str, ...]
    scope: tuple[str, ...]
    non_goals: tuple[str, ...]
    appetite: tuple[str, ...]
    questions: tuple[str, ...]


PATTERNS = [
    Pattern(
        key="waiting",
        label="wait visibility",
        cores=(
            "lets {users} see which {objs} are waiting for {peers}",
            "shows {objs} that are stuck waiting on {peers}",
            "helps {users} notice {objs} nobody has picked up",
            "makes it obvious when {a_obj} has been waiting too long",
        ),
        solutions=("send slack notifications", "show some kind of dashboard", "send a daily digest email", "color code old {objs}"),
        problem=(
            "{users} cannot easily see which {objs} are waiting on {peers}, so work stalls without anyone noticing.",
            "{objs} sit waiting on {peers} for days because nobody can tell which ones are blocked.",
        ),
        outcome=(
            "reduce the time {objs} spend waiting on {peers}.",
            "{objs} get picked up by {peers} quickly instead of stalling.",
        ),
        metrics=(
            "median time {a_obj} waits for {peers}",
            "percent of {objs} picked up within 24 hours",
            "number of {objs} waiting more than three days",
            "median time from creation to completion of {a_obj}",
        ),
        scope=(
            "identify {objs} that are waiting on {peers}",
            "surface waiting {objs} to {users} and {peers}",
            "measure how long each {obj} has been waiting",
            "let {peers} see their own waiting queue",
        ),
        non_goals=("replacing the {obj} workflow in {tool}", "automatically assigning {peers}", "ranking {peers} by speed"),
        appetite=("two weeks for a first version, then check the wait time metrics.", "one week to prototype, six weeks at most."),
        questions=(
            "is the delay caused by busy {peers} or by poor visibility?",
            "which teams feel this delay the most?",
            "what wait time counts as too long?",
        ),
    ),
    Pattern(
        key="sharing",
        label="sharing",
        cores=(
            "lets {users} share {objs} with {peers}",
            "makes it easy to send {a_obj} to {peers}",
            "gives {peers} access to {objs} without extra steps",
        ),
        solutions=("add a share button", "generate a public link", "add a share by email option"),
        problem=(
            "{users} have no simple way to give {peers} access to the {objs} they create, so they copy screenshots or files instead.",
            "sharing {objs} with {peers} takes too many steps, so useful information stays with one person.",
        ),
        outcome=(
            "{users} can share {objs} with {peers} in a few seconds.",
            "{peers} can see the {objs} that matter to them without asking.",
        ),
        metrics=(
            "share actions per active {obj}",
            "percent of {objs} shared at least once",
            "share completion rate",
            "number of {peers} viewing shared {objs} each week",
        ),
        scope=(
            "share {a_obj} with specific {peers}",
            "view a shared {obj}",
            "revoke access to a shared {obj}",
            "see who has access to {a_obj}",
        ),
        non_goals=("full workspace permission management", "real time collaborative editing", "sharing outside {tool}"),
        appetite=("two weeks for a first version.", "three weeks, then review share completion."),
        questions=(
            "can recipients be outside the organization?",
            "should {peers} be able to edit or only view?",
            "how long should shared access last?",
        ),
    ),
    Pattern(
        key="search",
        label="search",
        cores=(
            "helps {users} find old {objs} faster",
            "makes searching for {objs} less painful",
            "lets {users} quickly look up {a_obj}",
        ),
        solutions=("add a search bar", "add filters and tags", "use ai to search"),
        problem=(
            "{users} waste time hunting for {objs} because {tool} only lets them scroll long lists.",
            "{users} often cannot find the {obj} they need, so they recreate work that already exists.",
        ),
        outcome=(
            "{users} find the {obj} they need on the first try.",
            "reduce the time {users} spend looking for {objs}.",
        ),
        metrics=(
            "median time to find {a_obj}",
            "search success rate",
            "percent of searches followed by opening {a_obj}",
            "number of duplicate {objs} created each month",
        ),
        scope=(
            "search {objs} by keyword",
            "filter {objs} by owner and date",
            "show recently viewed {objs}",
            "open {a_obj} directly from results",
        ),
        non_goals=("searching content outside {tool}", "a recommendation engine", "rebuilding the {obj} list view"),
        appetite=("three weeks for a first version.", "two weeks, then review search success."),
        questions=(
            "what do {users} usually remember about the {obj} they want?",
            "how many {objs} does a typical team have?",
            "do {users} need to search archived {objs}?",
        ),
    ),
    Pattern(
        key="automation",
        label="data entry automation",
        cores=(
            "stops {users} from copying {obj} data by hand",
            "automates the boring copy paste around {objs}",
            "syncs {objs} so nobody retypes them",
        ),
        solutions=("build a zapier style integration", "add a csv import", "write a sync bot"),
        problem=(
            "{users} manually copy {obj} details between tools, which is slow and causes mistakes.",
            "retyping {obj} information takes {users} hours each week and errors slip through.",
        ),
        outcome=(
            "{obj} information is entered once and stays correct everywhere.",
            "{users} spend less time on manual data entry for {objs}.",
        ),
        metrics=(
            "hours per week spent on manual {obj} entry",
            "error rate in {obj} records",
            "percent of {objs} created without manual copying",
            "median time until a new {obj} is available in other tools",
        ),
        scope=(
            "import {obj} data automatically",
            "keep {obj} fields in sync",
            "flag sync errors to {users}",
            "show when {a_obj} was last synced",
        ),
        non_goals=("replacing {tool}", "a general purpose integration platform", "two way sync with every tool"),
        appetite=("four weeks for a first version.", "six weeks at most, starting with one source."),
        questions=(
            "which fields are copied most often?",
            "which tool should be the source of truth?",
            "how often do mistakes actually happen today?",
        ),
    ),
    Pattern(
        key="onboarding",
        label="onboarding",
        cores=(
            "helps new {users} get started with {objs}",
            "makes the first week less confusing for new {users}",
            "guides new {users} through setting up their first {obj}",
        ),
        solutions=("add a product tour", "make a video tutorial", "add a setup checklist"),
        problem=(
            "new {users} struggle to create their first {obj}, and many give up before seeing any value.",
            "new {users} do not know where to start in {tool}, so they ask {peers} for help or stop using it.",
        ),
        outcome=(
            "new {users} create their first {obj} without help.",
            "more new {users} reach their first success in {tool}.",
        ),
        metrics=(
            "percent of new {users} who create {a_obj} in their first week",
            "median time to first {obj}",
            "number of help requests from new {users}",
            "week four retention of new {users}",
        ),
        scope=(
            "guide new {users} to their first {obj}",
            "provide sensible defaults for new {objs}",
            "track setup progress",
            "let new {users} skip guidance",
        ),
        non_goals=("redesigning the whole {tool} interface", "training material for {peers}", "paid onboarding services"),
        appetite=("three weeks for a first version.", "two weeks to test one guided path."),
        questions=(
            "where exactly do new {users} drop off today?",
            "what does first success look like for a new {user}?",
            "do new {users} arrive alone or with a team?",
        ),
    ),
    Pattern(
        key="deadlines",
        label="deadline reminders",
        cores=(
            "reminds {users} before {objs} are due",
            "stops {users} from missing deadlines on {objs}",
            "nudges {peers} when {a_obj} is about to be late",
        ),
        solutions=("send push notifications", "add calendar invites", "send text messages"),
        problem=(
            "{users} regularly miss deadlines on {objs} because nothing warns them in time.",
            "late {objs} are only discovered after the deadline has passed, when it is too late to act.",
        ),
        outcome=(
            "fewer {objs} miss their deadline.",
            "{users} act on {objs} before they become late.",
        ),
        metrics=(
            "percent of {objs} completed on time",
            "number of overdue {objs} per week",
            "median days overdue for late {objs}",
            "percent of reminders followed by an action",
        ),
        scope=(
            "set due dates on {objs}",
            "remind owners before a deadline",
            "show overdue {objs}",
            "let {users} snooze a reminder",
        ),
        non_goals=("full project planning", "automatic deadline setting", "penalties for late {objs}"),
        appetite=("two weeks for a first version.", "one week to test reminders with one team."),
        questions=(
            "how far ahead should reminders arrive?",
            "who owns a deadline when several {users} are involved?",
            "how many reminders are too many?",
        ),
    ),
    Pattern(
        key="permissions",
        label="access control",
        cores=(
            "lets admins control who can see {objs}",
            "keeps sensitive {objs} away from the wrong people",
            "adds proper roles for {objs}",
        ),
        solutions=("add role based permissions", "add an admin panel", "add private folders"),
        problem=(
            "everyone in {tool} can see every {obj}, so {users} avoid storing sensitive {objs} there.",
            "{users} cannot restrict access to sensitive {objs}, which creates risk and blocks adoption.",
        ),
        outcome=(
            "{users} trust {tool} with sensitive {objs}.",
            "only the right people can see each {obj}.",
        ),
        metrics=(
            "percent of {objs} with restricted access",
            "number of access related support requests",
            "adoption among teams with sensitive {objs}",
            "median time to grant access to a new person",
        ),
        scope=(
            "mark {a_obj} as private",
            "grant access to specific {peers}",
            "show who can see {a_obj}",
            "log access changes",
        ),
        non_goals=("single sign on", "field level permissions", "compliance certification"),
        appetite=("four weeks for a first version.", "six weeks at most, private items first."),
        questions=(
            "which {objs} are considered sensitive?",
            "who should be allowed to change access?",
            "do we need an audit trail for regulators?",
        ),
    ),
    Pattern(
        key="history",
        label="change history",
        cores=(
            "shows what changed on {a_obj} and who changed it",
            "lets {users} see the history of {objs}",
            "helps {users} undo mistakes on {objs}",
        ),
        solutions=("add a version history panel", "keep a changelog", "add an undo button"),
        problem=(
            "when {a_obj} changes, {users} cannot tell what changed or who changed it, so mistakes are hard to trace.",
            "{users} lose work when {objs} are edited by mistake, because there is no history to recover from.",
        ),
        outcome=(
            "{users} can understand and undo any change to {a_obj}.",
            "mistakes on {objs} are found and fixed quickly.",
        ),
        metrics=(
            "median time to find the cause of a bad change",
            "number of restore actions per week",
            "support tickets about lost {obj} changes",
            "percent of changes with a known author",
        ),
        scope=(
            "record changes to {objs}",
            "show change history for {a_obj}",
            "restore a previous version",
            "show who made each change",
        ),
        non_goals=("real time collaboration", "branching and merging", "long term legal archiving"),
        appetite=("three weeks for a first version.", "four weeks, history first and restore second."),
        questions=(
            "how far back does history need to go?",
            "which fields matter most when tracing a change?",
            "should restoring a version notify {peers}?",
        ),
    ),
    Pattern(
        key="reporting",
        label="reporting",
        cores=(
            "gives {peers} a weekly report on {objs}",
            "lets {users} export {obj} numbers for leadership",
            "summarizes how {objs} are going",
        ),
        solutions=("build charts", "add pdf export", "make a spreadsheet export"),
        problem=(
            "{users} spend hours building reports on {objs} by hand for {peers}.",
            "{peers} have no regular view of how {objs} are going, so problems surface late.",
        ),
        outcome=(
            "{peers} get an accurate view of {objs} without manual work.",
            "reporting on {objs} takes minutes instead of hours.",
        ),
        metrics=(
            "hours spent preparing {obj} reports",
            "number of reports viewed each week",
            "percent of reports generated automatically",
            "median time from period end to report delivery",
        ),
        scope=(
            "summarize key {obj} numbers",
            "schedule a recurring report",
            "export a report",
            "share a report with {peers}",
        ),
        non_goals=("a full business intelligence tool", "a custom report builder", "forecasting"),
        appetite=("three weeks for a first version.", "two weeks for one standard report."),
        questions=(
            "which decisions will these reports drive?",
            "which numbers do {peers} ask for most?",
            "how often do {peers} need an update?",
        ),
    ),
    Pattern(
        key="bulk",
        label="bulk actions",
        cores=(
            "lets {users} update many {objs} at once",
            "saves {users} from clicking through {objs} one by one",
            "adds bulk editing for {objs}",
        ),
        solutions=("add checkboxes to the list", "add a bulk edit modal", "add keyboard shortcuts"),
        problem=(
            "{users} must update {objs} one at a time, which turns routine cleanup into hours of clicking.",
            "changing many {objs} at once is impossible, so {users} postpone cleanup and data goes stale.",
        ),
        outcome=(
            "{users} can handle large batches of {objs} quickly.",
            "routine changes to {objs} take minutes, not hours.",
        ),
        metrics=(
            "median time to update fifty {objs}",
            "number of bulk actions per week",
            "percent of updates done in bulk",
            "number of stale {objs}",
        ),
        scope=(
            "select multiple {objs}",
            "apply a change to selected {objs}",
            "preview changes before applying",
            "undo a bulk change",
        ),
        non_goals=("scripting or automation rules", "bulk actions across workspaces", "an api for bulk changes"),
        appetite=("two weeks for a first version.", "three weeks including undo."),
        questions=(
            "which bulk changes are most common?",
            "how many {objs} does a typical batch include?",
            "what safeguards prevent costly mistakes?",
        ),
    ),
    Pattern(
        key="mobile",
        label="mobile access",
        cores=(
            "lets {users} handle {objs} from their phone",
            "makes {objs} usable on mobile",
            "helps {users} check {objs} while away from their desk",
        ),
        solutions=("build a native app", "make the site responsive", "add a smartwatch app"),
        problem=(
            "{users} cannot act on {objs} away from their desk, so simple tasks wait until they are back.",
            "{tool} is unusable on a phone, so {users} delay quick decisions on {objs}.",
        ),
        outcome=(
            "{users} can handle quick {obj} tasks from anywhere.",
            "simple {obj} decisions no longer wait for a desk.",
        ),
        metrics=(
            "percent of {obj} actions taken on mobile",
            "median response time to new {objs}",
            "weekly mobile active {users}",
            "mobile task completion rate",
        ),
        scope=(
            "view {objs} on a phone",
            "approve or update {a_obj} on mobile",
            "receive mobile alerts for urgent {objs}",
            "find recent {objs} quickly",
        ),
        non_goals=("full feature parity with desktop", "offline editing", "tablet specific layouts"),
        appetite=("four weeks for a first version.", "three weeks for the top two mobile tasks."),
        questions=(
            "which {obj} tasks happen most away from the desk?",
            "do {users} need offline access?",
            "is a responsive site enough before building an app?",
        ),
    ),
    Pattern(
        key="templates",
        label="templates",
        cores=(
            "lets {users} start {objs} from templates",
            "keeps {objs} consistent across the team",
            "stops {users} from rebuilding the same {obj} every time",
        ),
        solutions=("add a template gallery", "add a clone button", "use ai to fill in {objs}"),
        problem=(
            "{users} rebuild similar {objs} from scratch, so quality varies and time is wasted.",
            "every {obj} looks different, which makes it hard for {peers} to review them.",
        ),
        outcome=(
            "{users} create consistent {objs} quickly.",
            "{peers} can review {objs} that follow a shared structure.",
        ),
        metrics=(
            "median time to create {a_obj}",
            "percent of {objs} started from a template",
            "number of review rounds per {obj}",
            "template reuse rate",
        ),
        scope=(
            "save {a_obj} as a template",
            "create {a_obj} from a template",
            "share templates with the team",
            "update a shared template",
        ),
        non_goals=("a public template marketplace", "enforcing templates for everyone", "automatic content generation"),
        appetite=("two weeks for a first version.", "three weeks including shared templates."),
        questions=(
            "which {objs} are rebuilt most often?",
            "who should own team templates?",
            "how strict should templates be?",
        ),
    ),
]

# Tekniske spørgsmål, der inviterer engineering tidligt ind i samtalen.
TECHNICAL_QUESTIONS = (
    "what data does {tool} already expose about {objs}?",
    "are there limits in {tool} that engineering should plan around?",
    "what is the cheapest prototype engineering could build to test this?",
)

# --------------------------------------------------------------------------
# Ekstra sektioner til de længere PRD'er: baggrund, user stories, risici og
# udrulning. De er skrevet én gang pr. mønster (hvordan klarer folk sig i dag,
# og hvad vinder de?) plus fælles skabeloner, der virker for alle mønstre.
# --------------------------------------------------------------------------

# Hvordan folk klarer problemet i dag (bruges i "background").
WORKAROUNDS = {
    "waiting": "{users} ask in chat or check {tool} by hand to find {objs} that are stuck",
    "sharing": "{users} send screenshots or exported files to {peers}",
    "search": "{users} scroll through long lists in {tool} or ask {peers} where things are",
    "automation": "{users} copy the same details into {tool} by hand for every {obj}",
    "onboarding": "new {users} rely on {peers} to show them around {tool}",
    "deadlines": "{users} keep their own notes and calendars to remember when {objs} are due",
    "permissions": "{users} ask an admin every time someone needs access to {a_obj}",
    "history": "{users} ask around to find out who changed {a_obj} and why",
    "reporting": "{users} export data from {tool} and build reports in spreadsheets",
    "bulk": "{users} open and change {objs} one at a time in {tool}",
    "mobile": "{users} wait until they are back at a computer to handle {objs}",
    "templates": "{users} copy an old {obj} and edit it by hand",
}

# Hvad brugerne vinder (bruges i "so that ..." i user stories).
BENEFITS = {
    "waiting": "nothing stalls without anyone noticing",
    "sharing": "{peers} always see the latest version",
    "search": "i find what i need in seconds",
    "automation": "i spend my time on work that needs judgment",
    "onboarding": "i become productive in my first week",
    "deadlines": "no {obj} is late by surprise",
    "permissions": "the right people have access without waiting on an admin",
    "history": "i can understand and undo mistakes",
    "reporting": "decisions are based on current numbers",
    "bulk": "routine cleanup takes minutes instead of hours",
    "mobile": "i can act right away wherever i am",
    "templates": "every {obj} starts from a good structure",
}

WHY_NOW = (
    "it came up in most recent interviews with {area} customers.",
    "requests about it have grown every quarter this year.",
    "it is one of the top complaints in feedback from {users}.",
    "sales hears about it in almost every {area} deal.",
)

RISKS = (
    "{users} may ignore one more signal in {tool}",
    "data in {tool} may be incomplete or out of date",
    "scope may grow beyond the appetite if every edge case is included",
    "{peers} may feel monitored rather than helped",
    "adoption may stay low if it adds steps to the {obj} workflow",
    "it may duplicate something {tool} already offers",
)

ROLLOUT = (
    "start with one {area} team as a pilot, then open it to all {users} after two weeks.",
    "ship behind a feature flag to ten percent of {users} and compare the success metrics.",
    "release to internal {users} first, fix the rough edges, then roll out to customers.",
)

# Hvordan en person begynder at beskrive en idé. "{core}" er mønstrets kerne.
IDEA_OPENERS = (
    "i want to build something that {core}.",
    "what if we built a tool that {core}?",
    "idea: something that {core}.",
    "could we make a feature that {core}?",
    "we keep hearing that {users} want something that {core}.",
    "thinking about a thing for {users} that {core}.",
)
SOLUTION_LEADS = ("maybe it could {solution}.", "it could {solution}.", "we could {solution}.", "probably {solution}.")
IDEA_ENDINGS = ("", "", " not sure yet.", " just a rough idea.", " mostly for {area} teams.")


# --------------------------------------------------------------------------
# Byg ét eksempel.
# --------------------------------------------------------------------------


def with_article(noun: str) -> str:
    """'invoice' -> 'an invoice', 'deal' -> 'a deal' (engelsk ubestemt artikel)."""
    return ("an " if noun[0] in "aeiou" else "a ") + noun


def fill(template: str, domain: Domain) -> str:
    """Udfyld {pladserne} i en skabelon med et domænes ord."""
    return template.format(
        users=domain.users,
        user=domain.user,
        obj=domain.obj,
        objs=domain.objs,
        a_obj=with_article(domain.obj),
        peers=domain.peers,
        tool=domain.tool,
        area=domain.area,
    )


def make_idea(domain: Domain, pattern: Pattern, rng: random.Random) -> tuple[str, list[str]]:
    """Skriv en rodet, løsnings-præget idé. Returnerer (idé, nævnte løsninger)."""
    core = fill(rng.choice(pattern.cores), domain)
    idea = fill(rng.choice(IDEA_OPENERS).replace("{core}", core), domain)

    n_solutions = rng.choice((0, 1, 1, 2))
    mentioned = [fill(s, domain) for s in rng.sample(pattern.solutions, n_solutions)]
    if mentioned:
        lead = rng.choice(SOLUTION_LEADS)
        idea += " " + lead.format(solution=" and ".join(mentioned))

    idea += fill(rng.choice(IDEA_ENDINGS), domain)
    return idea, mentioned


def make_record(domain: Domain, pattern: Pattern, rng: random.Random) -> PrdRecord:
    """Lav ét træningseksempel: (idé, formet PRD)."""
    idea, mentioned_solutions = make_idea(domain, pattern, rng)

    def pick(options: tuple[str, ...], k: int) -> list[str]:
        # Behold punkternes rækkefølge: en fast rækkefølge er lettere at lære.
        chosen = sorted(rng.sample(range(len(options)), k))
        return [fill(options[i], domain) for i in chosen]

    questions = pick(pattern.questions, rng.choice((2, 3)))
    questions.append(fill(rng.choice(TECHNICAL_QUESTIONS), domain))
    if mentioned_solutions:
        # Personen sprang til en løsning. En god PM holder den som en hypotese.
        questions.append(f"is it enough to {mentioned_solutions[0]}, or is that one option among several?")

    scope = pick(pattern.scope, rng.choice((3, 4)))
    # User stories bygges af scope-punkterne: "as a <bruger>, i want to <scope>, so that <gevinst>."
    benefit = fill(BENEFITS[pattern.key], domain)
    stories = [f"as {with_article(domain.user)}, i want to {item}, so that {benefit}." for item in scope[:2]]
    stories.append(f"as one of the {domain.peers}, i want to see the result without extra work in {domain.tool}.")

    background = (
        "today " + fill(WORKAROUNDS[pattern.key], domain) + ". "
        + fill(rng.choice(WHY_NOW), domain)
    )

    return PrdRecord(
        idea=idea,
        intent=f"{domain.obj} {pattern.label}",
        problem=fill(rng.choice(pattern.problem), domain),
        background=background,
        users=f"primary: {domain.users}. secondary: {domain.peers}.",
        user_stories=stories,
        outcome=fill(rng.choice(pattern.outcome), domain),
        success_metrics=pick(pattern.metrics, rng.choice((3, 4))),
        scope=scope,
        non_goals=pick(pattern.non_goals, rng.choice((2, 3))),
        risks=pick(RISKS, 2),
        rollout=fill(rng.choice(ROLLOUT), domain),
        appetite=rng.choice(pattern.appetite),
        open_questions=questions,
        domain=domain.key,
        pattern=pattern.key,
    )


# --------------------------------------------------------------------------
# Byg hele datasættet.
# --------------------------------------------------------------------------


def build_dataset(
    out_dir: Path,
    examples_per_combo: int = 40,
    unseen_fraction: float = 0.08,
    seen_test_per_combo: int = 2,
    seed: int = 0,
) -> dict[str, int]:
    """Skriv train.jsonl, test_seen.jsonl og test_unseen.jsonl til `out_dir`.

    * train        - eksempler fra de fleste (domæne, mønster)-kombinationer
    * test_seen    - nye eksempler af kombinationer, der også er i train
    * test_unseen  - kombinationer, der aldrig er i train (generalisering)
    """
    rng = random.Random(seed)
    combos = [(d, p) for d in DOMAINS for p in PATTERNS]
    rng.shuffle(combos)
    n_unseen = max(1, int(len(combos) * unseen_fraction))
    unseen, seen = combos[:n_unseen], combos[n_unseen:]

    splits: dict[str, list[PrdRecord]] = {"train": [], "test_seen": [], "test_unseen": []}
    for domain, pattern in seen:
        splits["train"] += [make_record(domain, pattern, rng) for _ in range(examples_per_combo)]
        splits["test_seen"] += [make_record(domain, pattern, rng) for _ in range(seen_test_per_combo)]
    for domain, pattern in unseen:
        splits["test_unseen"] += [make_record(domain, pattern, rng) for _ in range(4)]

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, records in splits.items():
        rng.shuffle(records)
        save_records(records, out_dir / f"{name}.jsonl")
    return {name: len(records) for name, records in splits.items()}


def save_records(records: list[PrdRecord], path: Path) -> None:
    with path.open("w") as f:
        for record in records:
            f.write(json.dumps(record.to_dict()) + "\n")


def load_records(path: Path) -> list[PrdRecord]:
    with path.open() as f:
        return [PrdRecord.from_dict(json.loads(line)) for line in f if line.strip()]
