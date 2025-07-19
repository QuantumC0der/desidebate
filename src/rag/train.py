#!/usr/bin/env python3
"""
RAG training/indexing script
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.rag.build_index import build_simple_index, build_chroma_index
import argparse

def main():
    """Main training function for RAG"""
    parser = argparse.ArgumentParser(description="Train RAG system")
    parser.add_argument("--type", choices=["simple", "chroma"], default="simple",
                       help="Type of RAG system to train")
    parser.add_argument("--data", default="data/raw/pairs.jsonl",
                       help="Input data file")
    parser.add_argument("--output", help="Output directory")
    
    args = parser.parse_args()
    
    print("RAG Training Script")
    print("=" * 30)
    
    if args.type == "simple":
        print("Building simple RAG index...")
        output = args.output or "src/rag/data/rag/simple_index.json"
        success = build_simple_index(args.data, output)
        
    elif args.type == "chroma":
        print("Building Chroma vector index...")
        output = args.output or "data/chroma/social_debate"
        success = build_chroma_index(args.data, output)
    
    if success:
        print("\nRAG training completed successfully!")
        print(f"Index saved to: {output}")
        return 0
    else:
        print("\nRAG training failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())