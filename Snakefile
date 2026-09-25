PY = ".venv/bin/python"
PP = "PYTHONPATH=src"


rule all:
    input:
        "results/summary.json",
        "results/curves.png",


rule fetch:
    output:
        "data/raw/gb1_flip.csv.zip",
    shell:
        "{PP} {PY} -c \"from al_loop.data import fetch_raw; "
        "from al_loop.config import load_config; "
        "fetch_raw(load_config()['dataset']['url'], '{output}')\""


rule prepare:
    input:
        rules.fetch.output,
    output:
        "data/processed/gb1.parquet",
    shell:
        "{PP} {PY} -m al_loop.data {input} {output}"


rule run:
    input:
        rules.prepare.output,
    output:
        records="data/processed/records.parquet",
        picks="data/processed/active_picks.parquet",
    shell:
        "{PP} {PY} -m al_loop.loop {input} {output.records} {output.picks}"


rule report:
    input:
        records=rules.run.output.records,
        landscape=rules.prepare.output,
    output:
        "results/summary.json",
        "results/curves.png",
    shell:
        "{PP} {PY} -m al_loop.evaluate "
        "{input.records} {input.landscape} {output[0]} {output[1]}"
