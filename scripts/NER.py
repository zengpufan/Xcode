import os
import random
from datasets import load_dataset, DatasetDict  
from tqdm import tqdm
import spacy
from multiprocessing import Pool
import json
import torch
import multiprocessing
import string
from typing import List, Dict, Any, Optional

LANG_CODE_SPACY_MODEL_MAPPING = {
    # 'en': 'en_core_web_trf',
    'en': 'en_core_web_lg',
    # 'zh': 'zh_core_web_trf',
    'zh': 'zh_core_web_lg',
    'de': 'de_core_news_lg',
    'fr': 'fr_core_news_lg',    
    'es': 'es_core_news_lg',
    'it': 'it_core_news_lg',
    'nl': 'nl_core_news_lg',
    'pt': 'pt_core_news_lg',
    'ja': 'ja_core_news_lg',
}

class LanguageEmbeddingPreprocessor:
    """
    A preprocessor for extracting named entities from multilingual Wikipedia datasets.
    
    This class handles loading Wikipedia datasets, processing them with spaCy NER models,
    and filtering/validating the extracted entities based on language-specific criteria.
    """
    
    def __init__(self, lang_code: str, device: int, cache_dir: Optional[str] = None, output_dir: Optional[str] = None):
        """
        Initialize the preprocessor with language-specific settings.
        
        Args:
            lang_code: Language code (e.g., 'en', 'zh', 'de')
            device: GPU device ID for spaCy processing
            cache_dir: Directory to cache datasets
            output_dir: Directory to save processed results
            
        Raises:
            ValueError: If language code is not supported
        """
        self.lang_code = lang_code
        self.device = device
        self.cache_dir = cache_dir
        self.output_dir = output_dir

        # Validate and load spaCy model
        if self.lang_code not in LANG_CODE_SPACY_MODEL_MAPPING:
            raise ValueError(f"Unsupported language code: {self.lang_code}")
            
        model_name = LANG_CODE_SPACY_MODEL_MAPPING[self.lang_code]
        
        # Configure GPU for transformer models
        if "trf" in model_name:
            print(f"CUDA available: {torch.cuda.is_available()}")
            try:
                spacy.require_gpu(self.device)
                print(f"Using GPU device: {self.device}")
            except Exception as e:
                print(f"Warning: Could not configure GPU, using CPU: {e}")
        
        # Load spaCy model
        try:
            self.nlp = spacy.load(model_name)
            print(f"NER labels for {lang_code}: {self.nlp.pipe_labels.get('ner', [])}")
        except OSError as e:
            raise ValueError(f"Could not load spaCy model '{model_name}': {e}")
        
        # Create output directory
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
    
    @staticmethod
    def get_dataset_length(lang_code: str, cache_dir: Optional[str] = None, 
                          dataset_name: str = "wikimedia/wikipedia", date: str = "20231101") -> int:
        """
        Get the total length of a Wikipedia dataset for a specific language.
        
        Args:
            lang_code: Language code
            cache_dir: Directory to cache the dataset
            dataset_name: Name of the dataset
            date: Dataset version date
            
        Returns:
            Number of samples in the dataset
        """
        try:
            data = load_dataset(dataset_name, f"{date}.{lang_code}", cache_dir=cache_dir)
            return len(data['train'])
        except Exception as e:
            print(f"Error loading dataset for {lang_code}: {e}")
            return 0
    
    def _get_valid_entities(self) -> List[str]:
        """Get list of valid entity types based on language."""
        if self.lang_code in ['zh', 'ja', 'en']:
            return ['DATE', 'FAC', 'GPE', 'LOC', 'ORG', 'WORK_OF_ART', 'NORP']
        else:
            return ['LOC', 'ORG', 'MISC']
    
    def _is_title_valid(self, title: str) -> bool:
        """
        Check if title contains invalid characters (digits or punctuation).
        
        Args:
            title: The title to validate
            
        Returns:
            True if title is valid, False otherwise
        """
        return not any(char.isdigit() or char in string.punctuation for char in title)
    
    def _has_valid_entities(self, doc, title: str, valid_entities: List[str]) -> bool:
        """
        Check if document contains valid entities that relate to the title.
        
        Args:
            doc: Processed spaCy document
            title: The title to check against
            valid_entities: List of valid entity types
            
        Returns:
            True if valid entities are found, False otherwise
        """
        # if title has digits return false
        if any(char.isdigit() for char in title):
            return False
        return any(
            ((ent.text.strip() in title) or (title in ent.text.strip())) and 
            ent.label_ in valid_entities
            for ent in doc.ents
            if ent.text.strip()  # Skip empty entities
        )
    
    def load_data(self, dataset_name: str = "wikimedia/wikipedia", date: str = "20231101", 
                  start_idx: int = 0, end_idx: int = 1) -> None:
        """
        Load and process Wikipedia dataset data.
        
        Args:
            dataset_name: Name of the dataset
            date: Dataset version date
            start_idx: Starting index for data selection
            end_idx: Ending index for data selection
            
        Raises:
            ValueError: If indices are invalid
        """
        # Validate indices
        if start_idx < 0 or end_idx <= start_idx:
            raise ValueError(f"Invalid indices: start_idx={start_idx}, end_idx={end_idx}")
        
        print(f"Loading dataset {dataset_name} for language {self.lang_code} from index {start_idx} to {end_idx}")
        
        try:
            data = load_dataset(dataset_name, f"{date}.{self.lang_code}", cache_dir=self.cache_dir)
            dataset_length = len(data['train'])
            
            # Adjust end_idx if it exceeds dataset length
            if end_idx > dataset_length:
                end_idx = dataset_length
                print(f"Warning: end_idx exceeds dataset length, adjusting to {end_idx}")
            
            self.data = data['train'].select(range(start_idx, end_idx))
            self.titles = self.data['title']
            self.texts = self.data['text']
            
            # Extract first paragraph from each text
            self.texts = [text.split('\n')[0] if text else "" for text in self.texts]
            print(f"Loaded {len(self.data)} samples.")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset: {e}")
        
        # Initialize result containers
        self.selected_titles = []
        self.selected_texts = []
        self.selected_indices = []
        self.selected_entities = []

        self.filtered_titles = []
        self.filtered_texts = []
        self.filtered_indices = []
        self.filtered_entities = []
        
        # Get valid entity types for this language
        valid_entities = self._get_valid_entities()
        
        # Process each sample
        for idx, title in enumerate(tqdm(self.titles, desc="Processing titles")):
            try:
                text = self.texts[idx]
                
                # Skip empty titles or texts
                if not title.strip() or not text.strip():
                    continue
                
                # Process with spaCy
                combined_text = text.strip()
                doc = self.nlp(combined_text)
                
                # Validate title
                has_invalid_label = not self._is_title_valid(title)
                
                # Check for valid entities if title is still valid
                if not has_invalid_label:
                    has_invalid_label = has_invalid_label or not self._has_valid_entities(doc, title, valid_entities)
                
                # Extract entities
                entities = [(ent.text.strip(), ent.label_) for ent in doc.ents ]
                
                # Categorize sample
                if not has_invalid_label and entities:
                    self.selected_titles.append(title)
                    self.selected_indices.append(start_idx + idx)
                    self.selected_entities.append(entities)
                else:
                    self.filtered_titles.append(title)
                    self.filtered_indices.append(start_idx + idx)
                    self.filtered_entities.append(entities)
                    
            except Exception as e:
                print(f"Warning: Error processing sample {idx}: {e}")
                continue
        
        print(f"Selected {len(self.selected_titles)} valid samples.")
        print(f"Filtered {len(self.filtered_titles)} invalid samples.")
    
    def save_data(self, start_idx: int) -> None:
        """
        Save processed data to JSON files.
        
        Args:
            start_idx: Starting index used for filename generation
        """
        if not self.output_dir:
            print("Warning: No output directory specified, skipping save")
            return
        
        selected_file_name = os.path.join(self.output_dir, f"selected_data_{start_idx}.json")
        filtered_file_name = os.path.join(self.output_dir, f"filtered_data_{start_idx}.json")
        
        # Save selected data
        try:
            with open(selected_file_name, 'w', encoding='utf-8') as f:
                for title, idx, entities in zip(self.selected_titles, self.selected_indices, self.selected_entities):
                    json.dump({
                        'title': title, 
                        'index': idx, 
                        'entities': entities,
                        'lang_code': self.lang_code
                    }, f, ensure_ascii=False)
                    f.write('\n')
            print(f"Saved {len(self.selected_titles)} selected samples to {selected_file_name}")
        except IOError as e:
            print(f"Error saving selected data: {e}")
        
        # Save filtered data
        try:
            with open(filtered_file_name, 'w', encoding='utf-8') as f:
                for title, idx, entities in zip(self.filtered_titles, self.filtered_indices, self.filtered_entities):
                    json.dump({
                        'title': title, 
                        'index': idx, 
                        'entities': entities,
                        'lang_code': self.lang_code
                    }, f, ensure_ascii=False)
                    f.write('\n')
            print(f"Saved {len(self.filtered_titles)} filtered samples to {filtered_file_name}")
        except IOError as e:
            print(f"Error saving filtered data: {e}")
    
    @staticmethod
    def merge_data_files(lang_code: str, output_dir: str) -> None:
        """
        Merge chunked data files into single files.
        
        Args:
            lang_code: Language code for filename generation
            output_dir: Directory containing chunked files
        """
        selected_output_file = os.path.join(output_dir, f"{lang_code}_selected_data.json")
        filtered_output_file = os.path.join(output_dir, f"{lang_code}_filtered_data.json")
        
        if not os.path.exists(output_dir):
            print(f"Warning: Output directory {output_dir} does not exist")
            return
        
        # Get all files once to avoid multiple directory scans
        try:
            all_files = os.listdir(output_dir)
        except OSError as e:
            print(f"Error reading directory {output_dir}: {e}")
            return
        
        selected_files = sorted([f for f in all_files if f.startswith("selected_data_") and f.endswith(".json")])
        filtered_files = sorted([f for f in all_files if f.startswith("filtered_data_") and f.endswith(".json")])
        
        # Merge selected files
        selected_count = 0
        with open(selected_output_file, 'w', encoding='utf-8') as outfile:
            for fname in selected_files:
                file_path = os.path.join(output_dir, fname)
                try:
                    with open(file_path, 'r', encoding='utf-8') as infile:
                        for line in infile:
                            if line.strip():  # Skip empty lines
                                outfile.write(line)
                                selected_count += 1
                except IOError as e:
                    print(f"Warning: Could not read file {fname}: {e}")
        
        # Merge filtered files
        filtered_count = 0
        with open(filtered_output_file, 'w', encoding='utf-8') as outfile:
            for fname in filtered_files:
                file_path = os.path.join(output_dir, fname)
                try:
                    with open(file_path, 'r', encoding='utf-8') as infile:
                        for line in infile:
                            if line.strip():  # Skip empty lines
                                outfile.write(line)
                                filtered_count += 1
                except IOError as e:
                    print(f"Warning: Could not read file {fname}: {e}")
        
        print(f"Merged {selected_count} selected samples and {filtered_count} filtered samples for {lang_code}")


def process_data_chunk(args: tuple) -> None:
    """
    Process a chunk of data for multiprocessing.
    
    Args:
        args: Tuple containing (lang_code, start_idx, end_idx, device, cache_dir, output_dir)
    """
    lang_code, start_idx, end_idx, device, cache_dir, output_dir = args
    
    try:
        preprocessor = LanguageEmbeddingPreprocessor(lang_code, device, cache_dir, output_dir)
        preprocessor.load_data(start_idx=start_idx, end_idx=end_idx)
        preprocessor.save_data(start_idx)
        print(f"Completed processing chunk {start_idx}-{end_idx} for {lang_code}")
    except Exception as e:
        print(f"Error processing chunk {start_idx}-{end_idx} for {lang_code}: {e}")


def run_multiprocessing(device_list: List[int], process_num: int, lang_list: List[str], 
                       cache_dir: str, output_base_dir: str, chunk_size: int) -> None:
    """
    Run multiprocessing data processing for multiple languages.
    
    Args:
        device_list: List of GPU device IDs
        process_num: Number of processes to use
        lang_list: List of language codes to process
        cache_dir: Directory for dataset caching
        output_base_dir: Base directory for output files
        chunk_size: Size of data chunks for processing
    """
    os.makedirs(output_base_dir, exist_ok=True)
    
    for lang_code in lang_list:
        print(f"Starting processing for language: {lang_code}")
        os.makedirs(os.path.join(output_base_dir, lang_code), exist_ok=True)
        # Get dataset length
        print(cache_dir)
        # if lang_code=='en':
        #     total_len=6000000
        # else:
        total_len = LanguageEmbeddingPreprocessor.get_dataset_length(
            lang_code, cache_dir, dataset_name="wikimedia/wikipedia", date="20231101"
        )
        total_len=500000
        if total_len == 0:
            print(f"Skipping {lang_code}: no data available")
            continue
        
        # Create chunks
        chunks = [
            (lang_code, i, min(i + chunk_size, total_len), 
             device_list[idx % len(device_list)], cache_dir, 
             os.path.join(output_base_dir, lang_code))
            for idx, i in enumerate(range(0, total_len, chunk_size))
        ]
        
        print(f"Processing {total_len} samples in {len(chunks)} chunks for {lang_code}")
        
        # Process chunks with multiprocessing
        try:
            with Pool(processes=process_num) as pool:
                pool.map(process_data_chunk, chunks)
        except Exception as e:
            print(f"Error in multiprocessing for {lang_code}: {e}")
            continue
        
        # Merge chunked files
        try:
            LanguageEmbeddingPreprocessor.merge_data_files(
                lang_code, os.path.join(output_base_dir, lang_code)
            )
            print(f"Completed processing for language: {lang_code}")
        except Exception as e:
            print(f"Error merging files for {lang_code}: {e}")


if __name__ == "__main__":
    # Set multiprocessing start method for compatibility
    multiprocessing.set_start_method('spawn')
    
    # Configuration
    device_list = [0]  # GPU devices
    lang_list = ['zh','en','de','fr','es','ja']  # Languages to process
    lang_list=['ja']
    cache_dir = ""
    output_base_dir = ""
    chunk_size = 10000  # Samples per chunk
    process_num = 33  # Number of parallel processes
    
    print("Starting NER data processing pipeline")
    print(f"Languages: {lang_list}")
    print(f"Chunk size: {chunk_size}")
    print(f"Processes: {process_num}")
    print(f"Output directory: {output_base_dir}")
    
    # Run the processing pipeline
    run_multiprocessing(
        device_list=device_list,
        process_num=process_num,
        lang_list=lang_list,
        cache_dir=cache_dir,
        output_base_dir=output_base_dir,
        chunk_size=chunk_size
    )
    
    print("NER data processing pipeline completed")