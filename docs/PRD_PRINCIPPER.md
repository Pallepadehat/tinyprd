# PRD-principper

Hvad gør en PRD god? Denne side samler de principper, som TinyPRD er bygget efter.
De bruges tre steder:

1. **Datasættet** (`tinyprd/synth.py`) er skrevet, så hvert eksempel følger principperne.
2. **PRD-formatet** (`tinyprd/prd.py`) har en sektion for hvert princip.
3. **Kritikeren** (`tinyprd/critic.py`) giver point efter principperne.

> **Hvad er en PRD?** PRD står for *Product Requirements Document*. Det er et kort dokument, der
> forklarer *hvorfor* noget skal bygges, og *hvad* der tæller som en succes. Så ved hele teamet
> (produkt, design og engineering), hvad de arbejder hen imod.

Principperne er inspireret af indsigter fra Lenny's Podcast og Lenny's Newsletter. De er
**genfortalt med egne ord** og bruges som *retningslinjer for adfærd*. De er ikke kopieret ind som
træningsdata. Modellen skal lære principperne, ikke gentage andres formuleringer.

---

## De syv principper

### 1. Start med problemet, ikke løsningen
Problemet skal stå øverst, i én eller to stærke sætninger. Det må ikke nævne en løsning.

| ❌ Løsning forklædt som problem | ✅ Rigtigt problem |
|---|---|
| "Vi mangler et dashboard med Slack-notifikationer." | "Udviklere kan ikke se, hvilke pull requests der venter på review, så arbejdet går i stå." |

**I TinyPRD:** Idéerne i datasættet er bevidst fulde af løsninger ("maybe it could send slack
notifications"). PRD'erne er altid problem-først. Kritikeren trækker point, hvis problemet
indeholder løsningsord som *dashboard*, *button* eller *slack*.

### 2. Succes skal kunne måles
Gode succesmål er målbare. Fx "median ventetid på review" eller "procent af pull requests
reviewet inden for 24 timer". De fungerer også som filter: nye feature-ønsker, der ikke flytter
målene, kan vente.

**I TinyPRD:** Sektionen `success metrics`. Kritikeren tjekker for målbare ord som *median*,
*percent*, *number* og *rate*.

### 3. Sæt grænser tidligt
En uklar idé skal snævres ind til et afgrænset koncept, *før* nogen begynder at bygge. Kan man se
slutningen fra begyndelsen?

**I TinyPRD:** Sektionen `scope` (det der er med) og sektionen `appetite` (hvor meget tid
problemet er værd, fx "to uger til en første version").

### 4. Skriv hvad I *ikke* bygger
Uden non-goals vokser projektet hele tiden (*scope creep*).

**I TinyPRD:** Sektionen `non-goals`. Mangler den, giver kritikeren 0 point på det område.

### 5. Kort tvinger til klarhed
Start med én side. Er dokumentet for langt, bliver det ikke læst, og fokus forsvinder.

**I TinyPRD:** Alle PRD'er er under 256 ord. Kritikeren trækker point over ca. 400 ord.

### 6. Undgå at mikrostyre
En PRD skal beskrive problemet og målene, ikke hvordan hver knap skal se ud. Designere og
engineers skal have plads til at finde den bedste løsning. Dokumentet skal *starte* en samtale,
ikke erstatte den.

**I TinyPRD:** Sektionen `open questions` stiller spørgsmål i stedet for at give ordrer. Nævner
idéen en løsning, bliver den til et spørgsmål: *"is it enough to send slack notifications, or is
that one option among several?"*

### 7. Inviter engineering ind tidligt
Spørg om tekniske begrænsninger tidligt. Hvilke data findes allerede? Hvad er den billigste
prototype?

**I TinyPRD:** Hver PRD har mindst ét teknisk åbent spørgsmål.

---

## Klassiske fejl, som kritikeren fanger

| Fejl | Hvorfor er det et problem? |
|---|---|
| Ordet **"just"** ("det er *just* en lille knap") | Det undervurderer arbejdet og skaber urealistiske forventninger. |
| **Løsning i problemformuleringen** | Teamet låses fast på én løsning, før problemet er forstået. |
| **Ingen non-goals** | Scope vokser, og fokus forsvinder. |
| **For langt** | Ingen læser det hele. |
| **Intet tidsbudget** | Ingen ved, hvornår det er tid til at stoppe. |

## Gode spørgsmål at stille sig selv

- Hvad er det *ene* vigtigste problem, vi løser for brugeren?
- Hvilke tal viser, at vi har succes?
- Hvad er helt sikkert *ikke* med i denne version?
- Har vi spurgt engineering om de tekniske begrænsninger?
- Hvor meget tid vil vi bruge, før vi går videre?
- Er dokumentet kort nok til, at hele teamet faktisk læser det?

## PRD-formatet i TinyPRD

```text
## intent            hvad handler det om, på 2-5 ord
## problem           1-2 sætninger, uden løsning
## users             primære og sekundære brugere
## outcome           hvad er anderledes, når vi er færdige?
## success metrics   3 målbare mål
## scope             3 ting der er med
## non-goals         2 ting der IKKE er med
## appetite          tidsbudget
## open questions    3-4 spørgsmål, mindst ét teknisk
```
