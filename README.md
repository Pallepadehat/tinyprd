<div align="center">

# 🧠📄 TinyPRD

**En lille sprogmodel, der former rodede produktidéer til gode PRD'er.<br/>Den skriver fra groft til fint, ikke ord for ord.**

[![tests](https://github.com/Pallepadehat/tinyprd/actions/workflows/tests.yml/badge.svg)](https://github.com/Pallepadehat/tinyprd/actions/workflows/tests.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c)
![Apple Silicon](https://img.shields.io/badge/k%C3%B8rer%20p%C3%A5-Apple%20Silicon-black)
![License: MIT](https://img.shields.io/badge/licens-MIT-green)

*Skoleprojekt i machine learning: en lille, letlæselig implementering af
[Next-Concept Modeling](https://hazyresearch.stanford.edu/blog/2026-10-12-ncm).*

</div>

---

## 💡 Hvad er det?

Du skriver en rodet idé:

```text
what if we built a tool that lets developers see which pull requests are waiting
for reviewers? maybe it could send slack notifications.
```

TinyPRD former den til en kort PRD, hvor **problemet står først**, **succes kan måles**, og det
står klart, **hvad der ikke er med**. Det her er modellens rigtige output, med små fejl og det hele:

```text
## intent
pull request wait visibility
## problem
pull requests sit waiting on reviewers for days because nobody can tell which ones are blocked.
## users
primary: developers. secondary: reviewers.
## outcome
pull requests get picked up by reviewers quickly instead of stalling.
## success metrics
- median time a pull request waits for reviewers
- percent of requests picked up within 24 hours
- median time from creation to completion of a pull request
## non-goals
- replacing the pull request workflow in github
- automatically assigning
## appetite
two weeks for first version, then check wait time metrics.
## open questions
- is the delay by busy or by poor visibility?
- what data does github already expose about pull?
- is it enough to send slack notifications, or is that one option several?
```

Læg mærke til, at idéen nævnte en løsning (Slack). PRD'en gør den til et **spørgsmål** i stedet
for et krav, og problemet nævner slet ikke Slack. Sådan tænker en god PM, og det har modellen lært.

## 🎨 Det særlige: groft til fint

De fleste sprogmodeller skriver ét ord ad gangen. TinyPRD bruger **Next-Concept Modeling (NCM)**
og skriver i stedet hele PRD'en i **8 lag**, fra en sløret skitse til skarpe detaljer:

```text
niveau 1   (2 koder)   ░░░░░░░░░░░░░░░░░░░░   "en PRD om ventetid"
niveau 4  (32 koder)   ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓   sektioner og nøgleord på plads
niveau 8 (256 koder)   ████████████████████   hvert ord præcist
```

**8 forward passes** i stedet for **256**. Du kan selv se lagene med `--levels`.

➡️ Den fulde forklaring står i **[docs/SAADAN_VIRKER_DET.md](docs/SAADAN_VIRKER_DET.md)**.

## 🔬 Forskningsspørgsmålet

> **Kan generering fra groft til fint gøre en lille sprogmodel god til struktureret
> produkttænkning, og hvordan klarer den sig mod en almindelig ord-for-ord model?**

Resultaterne står i **[docs/EKSPERIMENT.md](docs/EKSPERIMENT.md)**.

| Model | Forward passes pr. PRD | Tid pr. PRD (M3 Pro) | Forstår idéen (intent) | Kritiker-score |
|---|---|---|---|---|
| **TinyPRD (NCM)** | **8** | **0,006 s** | **100 %** | 9,3 / 10 |
| Almindelig ord-for-ord model | 200 | 0,13 s | 100 % | 10,0 / 10 |

*Testsættet med produktidéer, modellen aldrig har set.* **NCM er ca. 25 gange hurtigere** og
forstår idéerne lige så godt, men baseline skriver lidt pænere tekst. Se hele analysen i
[EKSPERIMENT.md](docs/EKSPERIMENT.md).

## 🚀 Kom i gang

Kræver Python 3.10+ og helst en Mac med Apple Silicon (M1/M2/M3). CPU og NVIDIA virker også.

```bash
git clone https://github.com/Pallepadehat/tinyprd.git
cd tinyprd
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

### 1. Lav data og træn (ca. 2 timer på en M3 Pro)

```bash
tinyprd data         # laver ~7.000 syntetiske eksempler i data/
tinyprd train-all    # træner tokenizer → generator → baseline, og gemmer i runs/
```

> Vil du bare se, om alt virker? Kør `tinyprd data --preset smoke && tinyprd train-all --preset smoke`.
> Det tager under et minut, men modellen bliver ikke god.

### 2. Brug modellen

```bash
# Form en idé: problem, succesmål, scope og spørgsmål
prd "what if we built a tool that reminds teachers before assignments are due?"

# Hele PRD'en
prd "what if we built a tool that reminds teachers before assignments are due?" --full

# Se PRD'en komme i fokus, niveau for niveau
tinyprd generate "idea: something that lets analysts share dashboards with coworkers." --levels

# Giv din egen PRD point
tinyprd critique min_prd.md

# Sammenlign NCM med baseline på testsættene
tinyprd evaluate
```

> **Bemærk:** Modellen kender kun ordene fra sit træningsdata (16 produktområder × 12 slags
> problemer, på engelsk). Ukendte ord bliver til `<unk>`, og så bliver resultatet dårligere.

## 🗂️ Projektets opbygning

```text
tinyprd/
├── tinyprd/
│   ├── prd.py                 PRD'ens format
│   ├── synth.py               syntetisk datasæt (domæner × mønstre)
│   ├── vocab.py               tekst ↔ ord-id'er
│   ├── config.py              alle hyperparametre
│   ├── layers.py              transformer-blok + pres/stræk-matricer
│   ├── concept_tokenizer.py   TRIN 1: tekst ↔ hierarki af koder
│   ├── generator.py           TRIN 2: forudsiger koder niveau for niveau
│   ├── sampling.py            generering fra groft til fint
│   ├── baseline.py            almindelig ord-for-ord model (sammenligning)
│   ├── training.py            træningsløkker
│   ├── critic.py              regelbaseret PRD-kritiker
│   ├── evaluate.py            NCM mod baseline
│   └── cli.py                 kommandoerne `tinyprd` og `prd`
├── tests/                     pytest
└── docs/
    ├── SAADAN_VIRKER_DET.md   modellen forklaret fra bunden
    ├── EKSPERIMENT.md         forskningsspørgsmål, metode og resultater
    ├── PRD_PRINCIPPER.md      hvad gør en PRD god?
    └── ORDBOG.md              fagord forklaret
```

Al kode har dokumentation på dansk. Variabel- og funktionsnavne er på engelsk, som det er
normalt i programmering.

## 📚 Dokumentation

| Dokument | Til hvem? |
|---|---|
| [Sådan virker det](docs/SAADAN_VIRKER_DET.md) | Dig der vil forstå modellen |
| [Eksperimentet](docs/EKSPERIMENT.md) | Dig der vil se resultaterne |
| [PRD-principper](docs/PRD_PRINCIPPER.md) | Dig der vil skrive bedre PRD'er |
| [Ordbog](docs/ORDBOG.md) | Når et fagord driller |

## 🙏 Inspiration og kilder

- Kumbong, Khare m.fl. (2026): *Next-Concept Modeling: Fast, Parallel Coarse-to-Fine Text Generation*,
  [blog](https://hazyresearch.stanford.edu/blog/2026-10-12-ncm) · [artikel](https://arxiv.org/abs/2610.14162) · [kode](https://github.com/HazyResearch/ncm)
- Tian m.fl. (2024): *Visual Autoregressive Modeling (VAR)*, [arXiv](https://arxiv.org/abs/2404.02905)
- PRD-principperne er inspireret af Lenny's Podcast og Newsletter og genfortalt med egne ord.

## 📄 Licens

MIT. Se [LICENSE](LICENSE).
