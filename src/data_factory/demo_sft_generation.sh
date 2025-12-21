#!/bin/bash
# Quick demo script for molecule SFT data generation

echo "================================================"
echo "Molecule SFT Data Generation Demo"
echo "================================================"
echo ""

# Check environment
if [ -z "$OPENAI_API_KEY" ]; then
    echo "⚠️  OPENAI_API_KEY not set. Run: source local_env.sh"
    exit 1
fi

echo "✓ Environment ready"
echo ""

# Show input files
echo "=== Input Files ==="
echo "Structural data:"
ls -lh data/encoder/test/molecule.parquet
echo ""
echo "SMILES strings (first 5):"
head -5 data/encoder/test/raw/molecule.txt
echo ""

# Generate small test dataset
echo "=== Generating Test Dataset (3 molecules) ==="
python src/data_factory/generate_molecule_sft_data.py \
    --max-molecules 3 \
    --sequential \
    --output data/molecule_sft_demo.parquet

echo ""

# Show output
echo "=== Output File ==="
ls -lh data/molecule_sft_demo.parquet
echo ""

# Test the output
echo "=== Testing Generated Data ==="
python -c "
from src.data_loader.multimodal_sft_dataset import MultiModalSFTDataset
import sys

dataset = MultiModalSFTDataset(
    dataset_path='data/molecule_sft_demo.parquet',
    use_combined_parquet=True
)
print(f'Loaded {len(dataset)} examples')
print('')

example = dataset[0]
print('Example 0:')
print(f'  SMILES: {example.get(\"smiles\", \"N/A\")}')
print(f'  Messages: {len(example[\"messages\"])} messages')
print(f'  Graph nodes: {example[\"graph_data\"][\"value\"][\"node_feat\"].shape[0]}')
print('')

print('User message preview:')
print(example['messages'][1]['content'][:150] + '...')
print('')

print('Assistant response preview:')
print(example['messages'][2]['content'][:150] + '...')
"

echo ""
echo "================================================"
echo "✓ Demo complete!"
echo "================================================"
echo ""
echo "To generate full dataset (49 molecules):"
echo "  python src/data_factory/generate_molecule_sft_data.py \\"
echo "      --output data/molecule_sft_train.parquet"
echo ""
echo "To load in your code:"
echo "  from src.data_loader.multimodal_sft_dataset import MultiModalSFTDataset"
echo "  dataset = MultiModalSFTDataset("
echo "      dataset_path='data/molecule_sft_train.parquet',"
echo "      use_combined_parquet=True"
echo "  )"

