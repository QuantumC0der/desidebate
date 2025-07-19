#!/usr/bin/env python3
"""
RAG index builder
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict
import sys

def build_simple_index(data_path: str = "data/raw/pairs.jsonl", 
                      output_path: str = "src/rag/data/rag/simple_index.json"):
    """Build simple RAG index from data"""
    
    print(f"Building simple index from {data_path}")
    
    documents = []
    data_file = Path(data_path)
    
    if not data_file.exists():
        print(f"Data file not found: {data_path}")
        print("Creating sample index with default data...")
        
        # Create sample documents
        sample_docs = [
            {
                "text": "Artificial intelligence regulation is necessary to prevent potential misuse and ensure ethical development. Government oversight can establish standards for AI safety, transparency, and accountability.",
                "metadata": {
                    "topic": "AI regulation",
                    "stance": "pro",
                    "quality": 0.8,
                    "source": "sample"
                }
            },
            {
                "text": "Government regulation of AI could stifle innovation and slow technological progress. The private sector is better positioned to self-regulate and adapt quickly to emerging challenges.",
                "metadata": {
                    "topic": "AI regulation", 
                    "stance": "con",
                    "quality": 0.7,
                    "source": "sample"
                }
            },
            {
                "text": "Universal Basic Income could provide economic security and reduce poverty, allowing people to pursue education, entrepreneurship, or care work without fear of destitution.",
                "metadata": {
                    "topic": "Universal Basic Income",
                    "stance": "pro", 
                    "quality": 0.9,
                    "source": "sample"
                }
            },
            {
                "text": "UBI may reduce work incentives and create dependency on government support. It could also be extremely expensive and require significant tax increases.",
                "metadata": {
                    "topic": "Universal Basic Income",
                    "stance": "con",
                    "quality": 0.8,
                    "source": "sample"
                }
            },
            {
                "text": "Social media platforms have democratized information sharing and enabled global communication, connecting people across geographical and cultural boundaries.",
                "metadata": {
                    "topic": "Social media impact",
                    "stance": "pro",
                    "quality": 0.7,
                    "source": "sample"
                }
            },
            {
                "text": "Social media has contributed to misinformation spread, mental health issues, and political polarization, creating echo chambers that divide society.",
                "metadata": {
                    "topic": "Social media impact",
                    "stance": "con",
                    "quality": 0.8,
                    "source": "sample"
                }
            },
            {
                "text": "Climate change requires immediate global action through renewable energy transition, carbon pricing, and international cooperation to prevent catastrophic environmental damage.",
                "metadata": {
                    "topic": "Climate action",
                    "stance": "pro",
                    "quality": 0.9,
                    "source": "sample"
                }
            },
            {
                "text": "Economic costs of rapid climate action could harm developing nations and working families. A gradual transition with technological innovation may be more sustainable.",
                "metadata": {
                    "topic": "Climate action",
                    "stance": "con",
                    "quality": 0.7,
                    "source": "sample"
                }
            }
        ]
        
        documents = sample_docs
        
    else:
        # Process real data file
        print("Processing data file...")
        processed = 0
        
        try:
            with open(data_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    try:
                        pair = json.loads(line.strip())
                        
                        # Extract submission
                        submission = pair.get('submission', {})
                        title = submission.get('title', '')
                        selftext = submission.get('selftext', '')
                        
                        if title:
                            documents.append({
                                "text": title + (" " + selftext if selftext else ""),
                                "metadata": {
                                    "type": "submission",
                                    "score": submission.get('score', 0),
                                    "source": "reddit_cmv",
                                    "id": submission.get('id', f'sub_{line_num}')
                                }
                            })
                        
                        # Extract delta comment if exists
                        delta_comment = pair.get('delta_comment', {})
                        if delta_comment and delta_comment.get('body'):
                            documents.append({
                                "text": delta_comment['body'],
                                "metadata": {
                                    "type": "delta_comment",
                                    "score": delta_comment.get('score', 0),
                                    "source": "reddit_cmv",
                                    "id": delta_comment.get('id', f'delta_{line_num}'),
                                    "persuasion_success": True
                                }
                            })
                        
                        processed += 1
                        if processed % 1000 == 0:
                            print(f"Processed {processed} pairs...")
                            
                        # Limit for demo purposes
                        if processed >= 5000:
                            break
                            
                    except json.JSONDecodeError:
                        continue
                    except Exception as e:
                        print(f"Error processing line {line_num}: {e}")
                        continue
                        
        except Exception as e:
            print(f"Error reading data file: {e}")
            return False
    
    # Save index
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(documents, f, indent=2, ensure_ascii=False)
    
    print(f"Built index with {len(documents)} documents")
    print(f"Saved to: {output_path}")
    
    return True

def build_chroma_index(data_path: str = "data/raw/pairs.jsonl",
                      output_path: str = "data/chroma/social_debate"):
    """Build Chroma vector index"""
    
    try:
        import chromadb
        from langchain_community.vectorstores import Chroma
        from langchain_openai import OpenAIEmbeddings
        import os
        
        # Check API key
        if not os.getenv('OPENAI_API_KEY'):
            print("Error: OPENAI_API_KEY not set")
            return False
            
        print(f"Building Chroma index from {data_path}")
        
        # Initialize embeddings
        embeddings = OpenAIEmbeddings(model='text-embedding-3-small')
        
        # Load documents
        documents = []
        metadata = []
        
        data_file = Path(data_path)
        if not data_file.exists():
            print("Data file not found, using sample data...")
            
            sample_texts = [
                "AI regulation is necessary for safety and ethics",
                "Government oversight could stifle AI innovation", 
                "UBI provides economic security and reduces poverty",
                "UBI may reduce work incentives and increase costs",
                "Social media democratizes information sharing",
                "Social media spreads misinformation and polarization"
            ]
            
            documents = sample_texts
            metadata = [{"source": "sample", "id": i} for i in range(len(sample_texts))]
            
        else:
            # Process real data
            with open(data_file, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    try:
                        pair = json.loads(line.strip())
                        submission = pair.get('submission', {})
                        
                        if submission.get('title'):
                            text = submission['title']
                            if submission.get('selftext'):
                                text += " " + submission['selftext']
                            
                            documents.append(text)
                            metadata.append({
                                "type": "submission",
                                "score": submission.get('score', 0),
                                "id": submission.get('id', f'sub_{i}')
                            })
                        
                        if i >= 1000:  # Limit for demo
                            break
                            
                    except:
                        continue
        
        # Create Chroma index
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        vectorstore = Chroma.from_texts(
            texts=documents,
            embedding=embeddings,
            metadatas=metadata,
            persist_directory=str(output_dir)
        )
        
        print(f"Built Chroma index with {len(documents)} documents")
        print(f"Saved to: {output_path}")
        
        return True
        
    except ImportError as e:
        print(f"Missing dependencies for Chroma: {e}")
        print("Install with: pip install chromadb langchain-openai")
        return False
    except Exception as e:
        print(f"Error building Chroma index: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Build RAG index")
    parser.add_argument("--type", choices=["simple", "chroma"], default="simple",
                       help="Type of index to build")
    parser.add_argument("--data", default="data/raw/pairs.jsonl",
                       help="Input data file")
    parser.add_argument("--output", help="Output path")
    
    args = parser.parse_args()
    
    if args.type == "simple":
        output = args.output or "src/rag/data/rag/simple_index.json"
        success = build_simple_index(args.data, output)
    elif args.type == "chroma":
        output = args.output or "data/chroma/social_debate"
        success = build_chroma_index(args.data, output)
    
    if success:
        print("Index building completed successfully!")
        return 0
    else:
        print("Index building failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())