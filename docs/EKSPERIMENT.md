# Eksperimentet

## Forskningsspørgsmål

> **Kan generering fra groft til fint (NCM) gøre en lille sprogmodel god til at forme produktidéer
> til strukturerede PRD'er, og hvordan klarer den sig mod en almindelig ord-for-ord model?**

Vi ser på tre ting:
1. **Hastighed:** hvor mange forward passes og hvor meget tid koster én PRD?
2. **Struktur:** forstår modellen, *hvad* idéen handler om, og følger den PRD-formatet?
3. **Kvalitet:** hvor gode er formuleringerne?

---

## Opsætning

### Data
Syntetisk datasæt fra [`synth.py`](../tinyprd/synth.py):
**16 produktområder × 12 slags problemer = 192 kombinationer.**

| Del | Antal | Beskrivelse |
|---|---|---|
| `train` | 7.080 | 177 kombinationer × 40 eksempler |
| `test_seen` | 354 | nye eksempler af kombinationer fra train |
| `test_unseen` | 60 | **15 kombinationer, modellen aldrig har set** |

`test_unseen` er den vigtigste test. Modellen har fx set "deling" for dashboards og "påmindelser"
for opgaver, men aldrig "påmindelser for deployments". Den skal altså *kombinere* det, den har lært.

### Modeller

| | Koncept-tokenizer | NCM-generator | Baseline (autoregressiv) |
|---|---|---|---|
| Parametre | 1,9M | 6,1M | 5,2M |
| Lag × bredde | 2+2 × 128 | 6 × 256 | 6 × 256 |
| Træning | 4.000 skridt, batch 64 | 3.000 skridt, batch 32 | 3.000 skridt, batch 32 |
| Træningstid (M3 Pro) | ca. 14 min. | ca. 40 min. | ca. 60 min. |

Generatoren og baseline ser **præcis lige mange eksempler** (ca. 14 epochs).

### Mål

| Mål | Betydning |
|---|---|
| **critic** | Kritikerens score 0-10 ([critic.py](../tinyprd/critic.py)): problem, mål, scope, non-goals, spørgsmål, længde |
| **word_f1** | Ord-overlap med facit (0-1). *Loftet er under 1*, for der findes flere rigtige formuleringer af samme PRD. |
| **intent** | Andel hvor intent-linjen ("dashboard sharing") er helt rigtig. Den viser, om modellen har *forstået* idéen. |
| **nfe** | Forward passes pr. PRD |
| **sek/prd** | Målt tid pr. PRD på en M3 Pro (batch 32) |

---

## Resultater

60 eksempler pr. testsæt. Tal fra `tinyprd evaluate`.

### Kendte kombinationer (`test_seen`)

| Model | critic | word_f1 | intent | nfe | sek/prd |
|---|---|---|---|---|---|
| **NCM** (med oprydning) | 9,02 | 0,780 | **100 %** | **8** | **0,0065** |
| NCM (råt output) | 9,02 | 0,766 | 100 % | 8 | 0,0058 |
| Baseline | **9,89** | **0,797** | 100 % | 232 | 0,1661 |

### Nye kombinationer (`test_unseen`)

| Model | critic | word_f1 | intent | nfe | sek/prd |
|---|---|---|---|---|---|
| **NCM** (med oprydning) | 9,27 | 0,766 | **100 %** | **8** | **0,0057** |
| NCM (råt output) | 9,29 | 0,751 | 100 % | 8 | 0,0059 |
| Baseline | **9,98** | **0,801** | 100 % | 200 | 0,1294 |

### Tokenizeren: hvor meget bærer hvert niveau?

Andel af ordene, der gendannes korrekt, hvis man *kun* bruger de første k niveauer:

| Niveau (koder) | 1 (2) | 2 (8) | 3 (16) | 4 (32) | 5 (64) | 6 (128) | 7 (192) | 8 (256) |
|---|---|---|---|---|---|---|---|---|
| test_seen | 36 % | 42 % | 49 % | 59 % | 87 % | 98 % | 100 % | 100 % |
| test_unseen | 26 % | 32 % | 38 % | 51 % | 81 % | 95 % | 99 % | 100 % |

**Kun 2 koder gendanner over en tredjedel af en PRD på 256 ord.** Nested dropout virker: de
grove niveauer bærer den vigtigste information.

---

## Se det ske: en PRD kommer i fokus

Idé: *"idea: something that lets analysts share dashboards with coworkers."*
(`tinyprd generate "..." --levels`, forkortet)

**Efter niveau 1, kun 2 koder:** intent og problem er allerede på plads, resten er en tåge af
nøgleord.

```text
## intent
dashboard sharing
## problem
analysts have no simple way to give coworkers access to dashboards they create, so copy screenshots or files instead.
:.: coworkers. ##
can see share dashboards with in few. success - per active per -
of shared week
scope
a with specific coworkers a
```

**Efter niveau 4, 32 koder:** sektionerne tager form, og mange ord mangler stadig.

```text
## outcome
analysts can dashboards coworkers in to them without asking. ## success metrics
- actions per dashboard
percent dashboards at least once - number coworkers viewing each
scope - a dashboard access revoke - access to a dashboard ## non-goals
full workspace permission management -
```

**Efter niveau 8, 256 koder:** færdig PRD. Kritiker-score 9,5/10.

```text
## outcome
coworkers can see the dashboards that matter to them without asking.
## success metrics
share actions per active dashboard
- percent of dashboards shared at least once
- number of coworkers viewing shared dashboards each week ## scope
- share a dashboard specific coworkers - view a shared dashboard
- revoke has access to a dashboard
## non-goals
- full workspace permission management
- sharing outside the analytics app
## appetite
three weeks, then review share completion.
```

Det interessante: **starten af dokumentet bliver skarp først.** Intent og problem står stort set
færdige efter niveau 1. Det passer med, at de groveste koder bærer "hvad handler det om".

---

## Hvad betyder det?

### ✅ NCM er ca. 25 gange hurtigere
8 forward passes i stedet for 200-230. Målt i tid: ca. 0,006 sekunder mod 0,13-0,17 sekunder pr.
PRD.

> ⚠️ **Fair sammenligning?** Baseline bruger ingen KV-cache (for at holde koden enkel), så den
> er langsommere, end den behøvede at være. Antallet af forward passes (nfe) er det mest fair
> mål, og der er forskellen 25-29 gange.

### ✅ NCM forstår idéerne, også nye kombinationer
Intent er 100 % rigtig på *begge* testsæt. Modellen har altså lært at kombinere produktområde og
problemtype, ikke bare lært eksempler udenad. Kritikerens score er endda en anelse højere på de
nye kombinationer (9,27) end på de kendte (9,02).

### ❌ Baseline skriver pænere tekst
Baseline får næsten fuld score (9,9-10), mens NCM ligger på 9,0-9,3. NCM's fejl er typiske for
parallel generering:
- **Gentagne ord** ("dashboards dashboards"), fordi nabopositioner vælger ord samtidig.
  Oprydningen (`collapse_repeats`) løfter word_f1 fra 0,766 til 0,780.
- **Manglende ord** ("share a dashboard ~~with~~ specific coworkers").
- **Linjeskift det forkerte sted** ("... each week ## scope").

### Konklusion
På denne opgave er **NCM markant hurtigere og lige så god til at forstå idéen**, men en **lille
smule dårligere til ordvalg**. Det samme mønster ses i den originale NCM-artikel: stor
hastighedsgevinst, med kvalitet der er tæt på, men ikke altid helt på højde med, de bedste
alternativer.

---

## Lærdomme undervejs

### 1. Attention-dropout gør træning 30 gange langsommere på en Mac
Den første generator-træning nåede under 100 skridt på 15 minutter. Efter en måling viste det sig,
at PyTorchs attention på Apples GPU (MPS) **ikke har en hurtig udgave med dropout**:

| | Tid pr. skridt |
|---|---|
| Med dropout på attention-vægtene | 46,6 s |
| Uden (dropout på outputtet i stedet) | 1,4 s |

Løsning: dropout lægges på attention-*outputtet* ([`layers.py`](../tinyprd/layers.py)).
Derudover er en fælles maske for hele batchet ca. 5 gange hurtigere end en maske pr. eksempel.

### 2. Blok-diagonal attention kan køres blok for blok
Når blokkene ikke kan se hinanden, kan de køres hver for sig med præcis samme resultat, og det er
billigere. Se [SAADAN_VIRKER_DET.md](SAADAN_VIRKER_DET.md#42-træn-alle-niveauer-i-ét-skridt).

### 3. Mål altid før du optimerer
Begge problemer blev fundet ved at time hver del for sig (forward, backward, attention med og uden
maske) i stedet for at gætte.

---

## Begrænsninger

- **Syntetisk data:** opgaven er lettere end virkeligheden. Baseline når næsten perfekt score,
  fordi datasættet bygger på skabeloner.
- **Lille ordforråd:** modellen kender ca. 750 ord. Idéer uden for de 16 områder giver `<unk>`.
- **Kun 60 eksempler pr. testsæt** og ét træningsforsøg (seed 0). Små forskelle kan være tilfældige.
- **Kritikeren er regelbaseret:** den tjekker struktur og nøgleord, ikke om teksten giver mening.

## Næste skridt

1. **Træn generatoren længere.** Loss faldt stadig ved skridt 3.000 (fra 0,96 til 0,92 over de
   sidste 600 skridt).
2. **KV-cache i baseline** for en helt fair tidsmåling.
3. **Trænet kritiker:** en lille model, der giver point og spørgsmål, så writer og kritiker kan
   forbedre hinanden.
4. **Interaktiv formning:** `prd` stiller spørgsmål, brugeren svarer, og svarene kommer med i
   prompten.
5. **Semantiske niveauer:** lad niveauerne svare til *idé → problem → mål → scope → PRD*, i stedet
   for at modellen selv lærer, hvad hvert niveau betyder.
6. **Rigtige PRD'er:** test på håndskrevne idéer uden for skabelonerne.

## Gentag eksperimentet

```bash
tinyprd data
tinyprd train-all
tinyprd evaluate
```

Alt er seedet (seed 0), men GPU-beregninger kan give små forskelle fra kørsel til kørsel.
