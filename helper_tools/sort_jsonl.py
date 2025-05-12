import os
import jsonlines
from tqdm import tqdm
from huggingface_hub import snapshot_download
import shutil
import gzip
from dotenv import load_dotenv
import git

repo = git.Repo(search_parent_directories=True)
load_dotenv(repo.working_dir + "/.env", override=True)

def sort_jsonl_file(input_file, output_file=None):
    """
    Sort a JSONL file by docid and write it back.
    
    Args:
        input_file (str): Path to the input JSONL file
        output_file (str, optional): Path to the output JSONL file. If None, overwrites input file.
    """
    if output_file is None:
        output_file = input_file
    
    # Read all entries
    print(f"Reading entries from {input_file}...")
    entries = []
    with jsonlines.open(input_file) as reader:
        for entry in tqdm(reader):
            entries.append(entry)
    
    # Sort by docid
    print("Sorting entries by docid...")
    entries.sort(key=lambda x: x.get('id', x.get('docid')))
    
    # Write back sorted entries
    print(f"Writing sorted entries to {output_file}...")
    with jsonlines.open(output_file, mode='w') as writer:
        for entry in tqdm(entries):
            writer.write(entry)
    
    print("Sorting complete!")

def sort_synthie_text_splits():
    """
    Sort all splits from sdg_text_davinci_003 dataset.
    """
    # Download dataset if not already present
    file_path = snapshot_download("martinjosifoski/SynthIE", repo_type="dataset", 
                                 local_dir=os.getenv("DATASET_DIR") + "/synthIE")
    
    base_path = f'{file_path}/sdg_text_davinci_003'
    splits = ['test', 'val']
    
    for split in splits:
        gz_filename = f'{base_path}/{split}.jsonl.gz'
        filename = f'{base_path}/{split}.jsonl'
        
        # Decompress if needed
        if not os.path.exists(filename):
            print(f"Decompressing {gz_filename}...")
            with gzip.open(gz_filename, 'rb') as f_in:
                with open(filename, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        
        # Sort the file
        print(f"\nProcessing {split} split...")
        sort_jsonl_file(filename)
        
        # Recompress the sorted file
        print(f"Recompressing {filename}...")
        with open(filename, 'rb') as f_in:
            with gzip.open(gz_filename, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        print(f"Completed processing {split} split!")

if __name__ == "__main__":
    sort_synthie_text_splits() 