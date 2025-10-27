import argparse
import os
import random
import time
import math
import json
import pandas as pd
from openai import OpenAI
from eval import get_simcse, get_simcse_llm_param

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Get API key from environment variable or use default
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')

def parse_args():
    parser = argparse.ArgumentParser(description="ChatGPT RNA Inference")
    parser.add_argument("--api-key", help="OpenAI API key", default=OPENAI_API_KEY)
    parser.add_argument("--model", help="ChatGPT model to use", default="gpt-4o")
    parser.add_argument("--variant", choices=["name_only", "name_and_sequence"], 
                       default="name_only", help="Whether to provide only RNA name or both name and sequence")
    parser.add_argument("--start-index", type=int, default=4200, help="Starting index for evaluation")
    parser.add_argument("--end-index", type=int, default=4210, help="Ending index for evaluation")
    args = parser.parse_args()
    return args

def setup_chatgpt_client(api_key):
    """Initialize OpenAI client"""
    client = OpenAI(api_key=api_key)
    return client

def generate_rna_description_chatgpt(client, model, name, sequence=None, variant="name_only"):
    """
    Generate RNA functional description using ChatGPT
    
    Args:
        client: OpenAI client
        model: ChatGPT model name
        name: RNA name
        sequence: RNA sequence (optional)
        variant: "name_only" or "name_and_sequence"
    
    Returns:
        str: Generated description
    """
    
    if variant == "name_only":
        system_prompt = """You are a bioinformatics expert specializing in RNA analysis. 
        Given an RNA name, provide a comprehensive functional description of what this RNA does, 
        its role in biological processes, and any known associations with diseases or cellular functions. 
        Be detailed and scientific in your response."""
        
        user_prompt = f"Give me a functional description of this RNA named {name}."
        
    else:  # name_and_sequence
        system_prompt = """You are a bioinformatics expert specializing in RNA analysis. 
        Given an RNA name and its sequence, provide a comprehensive functional description of what this RNA does, 
        its role in biological processes, and any known associations with diseases or cellular functions. 
        Consider both the name and sequence information in your analysis. Be detailed and scientific in your response."""
        
        user_prompt = f"Give me a functional description of this RNA named {name} with sequence: {sequence}"
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=1500,
            temperature=0.3,
            top_p=0.9
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        print(f"Error generating description for {name}: {e}")
        return f"Error: Could not generate description for {name}"

def main():
    print("Starting ChatGPT RNA Inference...")
    
    # Parse arguments
    args = parse_args()
    
    # Setup ChatGPT client
    client = setup_chatgpt_client(args.api_key)
    
    # Create results directory
    directory_name = "results"
    if not os.path.exists(directory_name):
        try:
            os.mkdir(directory_name)
        except Exception as e:
            print(f"An error occurred when creating results folder: {e}")
    
    # Load data
    df = pd.read_csv("rna_summary_2d.csv")
    ids = df['id'].values.tolist()[args.start_index:]
    names = df['name'].values.tolist()[args.start_index:]
    sequence = df['Sequence'].values.tolist()[args.start_index:]
    labels = df['summary_no_citation'].values.tolist()[args.start_index:]
    
    func_text = []
    
    print(f"Processing {len(ids)} RNA samples using variant: {args.variant}")
    print(f"Using model: {args.model}")
    
    for i, (id, name, seq, lab) in enumerate(zip(ids, names, sequence, labels)):
        print(f"Processing {i+1}/{len(ids)}: {name}")
        
        # Truncate sequence if too long
        if len(seq) > 1000:
            seq = seq[:1000]
        
        # Generate description using ChatGPT
        if args.variant == "name_only":
            user_message = f"Give me a functional description of this RNA named {name}."
            llm_message = generate_rna_description_chatgpt(client, args.model, name, variant="name_only")
        else:
            user_message = f"Give me a functional description of this RNA named {name} with sequence: {seq}"
            llm_message = generate_rna_description_chatgpt(client, args.model, name, seq, variant="name_and_sequence")
        
        # Create entry for evaluation
        entry = {
            "seq": seq, 
            "query": user_message, 
            "correct_func": lab, 
            "predict_func": llm_message,
            "variant": args.variant,
            "model": args.model
        }
        func_text.append(entry)
        
        print("Uniprot ID:", id)
        print("Correct summary:", lab)
        print(f"Predicted summary: {llm_message}")
        print('='*80)
        
        # Add small delay to avoid rate limiting
        time.sleep(0.1)
    
    print("******************")
    
    # Calculate evaluation metrics
    simcse_path = "princeton-nlp/sup-simcse-roberta-large"
    scores = get_simcse(simcse_path, func_text)
    
    # Save results
    output_filename = f"results/chatgpt_{args.variant}_{args.model.replace('-', '_')}.json"
    
    # Convert numpy types to Python types for JSON serialization
    def convert_numpy_types(obj):
        if isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        elif hasattr(obj, 'item'):  # numpy types
            return obj.item()
        else:
            return obj
    
    scores_serializable = convert_numpy_types(scores)
    
    with open(output_filename, "w") as outfile:
        json.dump(scores_serializable, outfile, indent=4)
    
    print(f"Results saved to: {output_filename}")
    print("ChatGPT RNA Inference completed!")

if __name__ == "__main__":
    main()
