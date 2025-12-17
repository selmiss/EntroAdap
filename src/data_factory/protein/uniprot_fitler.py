import json


def process_uniprot_json(input_path: str, output_path: str = None):
    """
    Process UniProt JSON file and extract protein data to JSONL format.
    
    Args:
        input_path: Path to input JSON file
        output_path: Path to output JSONL file (defaults to input_path with .jsonl extension)
    """
    if output_path is None:
        output_path = input_path.rsplit('.', 1)[0] + '.jsonl'
    
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    with open(output_path, 'w') as out_f:
        for protein in data.get('results', []):
            try:
                uniprot_id = protein.get('primaryAccession')
                if not uniprot_id:
                    continue
                
                comments = []
                for comment in protein.get('comments', []):
                    comment_type = comment.get('commentType', '')
                    
                    if 'texts' in comment:
                        for text_obj in comment['texts']:
                            if 'value' in text_obj:
                                comments.append({
                                    'type': comment_type,
                                    'value': text_obj['value']
                                })
                    
                    if 'note' in comment and 'texts' in comment['note']:
                        for text_obj in comment['note']['texts']:
                            if 'value' in text_obj:
                                comments.append({
                                    'type': f"{comment_type} (note)",
                                    'value': text_obj['value']
                                })
                
                sequence = protein.get('sequence', {}).get('value', '')
                
                protein_data = {
                    'uniprot_id': uniprot_id,
                    'comments': comments,
                    'sequence': sequence
                }
                
                out_f.write(json.dumps(protein_data) + '\n')
                
            except Exception as e:
                print(f"Error processing protein {protein.get('primaryAccession', 'unknown')}: {e}")
                continue
    
    print(f"Processed {len(data.get('results', []))} proteins. Output saved to {output_path}")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python uniprot_fitler.py <input_json_path> [output_jsonl_path]")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    process_uniprot_json(input_path, output_path)
