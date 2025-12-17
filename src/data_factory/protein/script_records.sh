python src/data_factory/protein/uniprot_fitler.py data/uniprot/test/uniprotkb_test25.json
python src/data_factory/protein/map_fetch_pdb3d.py data/uniprot/test/idmapping.json
python src/data_factory/protein/inspect_cif.py data/uniprot/test/pdb_structures/2MIO.cif
python src/data_factory/protein/instruction_construction.py data/uniprot/test/uniprotkb_test25.jsonl

python src/data_factory/protein/instruction_construction.py add-structure \
    data/uniprot/test/uniprotkb_test25_instructions.jsonl \
    data/uniprot/test/pdb_structures/ \
    data/uniprot/test/idmapping.json \
    --ca