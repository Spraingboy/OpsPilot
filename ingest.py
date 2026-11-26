"""Examples:
python ingest.py --tenant a --group employee --id A-VPN-01 --file sample_docs/a_vpn.md
python ingest.py --tenant a --group it --id A-SEC-01 --file sample_docs/a_security.md
python ingest.py --tenant b --group employee --id B-VPN-01 --file sample_docs/b_vpn.md
"""
import argparse
from pathlib import Path
from rag import init_schema, load_document, upsert_document

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tenant", required=True)
    parser.add_argument("--group", required=True)
    parser.add_argument("--id", required=True)
    parser.add_argument("--file", required=True)
    args = parser.parse_args()
    file = Path(args.file)
    content = load_document(file)
    init_schema()
    count = upsert_document(
        tenant=args.tenant, doc_id=args.id, title=file.stem,
        allowed_group=args.group, content=content,
    )
    print(f"Indexed {args.id}: {count} chunks")

if __name__ == "__main__":
    main()
