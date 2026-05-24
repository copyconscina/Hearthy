#!/bin/bash
echo "Downloading journals..."
gdown --folder "https://drive.google.com/drive/folders/1xweGz8XNANh8oga23jeReMsbtd-QXdeC" -O app/data/journals

echo "Building knowledge base..."
python scripts/load_knowledge.py --folder app/data/journals --output app/data/knowledge_base.json

echo "Done!"
