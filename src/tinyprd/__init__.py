"""TinyPRD: en lille Next-Concept Model, der former rodede produktidéer til PRD'er.

Pakken er delt op sådan:

    data/        PRD-formatet, syntetisk datasæt og ordforråd
    models/      koncept-tokenizer, NCM-generator og baseline
    web/         webinterfacet (`tinyprd serve`)

    config.py    alle hyperparametre
    runtime.py   enhed, seeds, checkpoints
    sampling.py  generering fra groft til fint
    training.py  træningsløkkerne
    critic.py    regelbaseret PRD-kritiker
    evaluate.py  NCM mod baseline
    cli.py       kommandoerne `tinyprd` og `prd`
"""
