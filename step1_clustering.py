import os
import numpy as np
import json
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from cuml import KMeans as cuKMeans
from typing import List, Dict, Any, Optional

class LanguageEmbeddingProcessor:
    """
    Processor for language-specific embeddings and clustering.
    
    This class loads filtered Wikipedia data, computes embeddings using sentence transformers,
    and performs clustering analysis on the embeddings.
    """
    
    def __init__(self,
                 lang_code: str, 
                 device: str = 'cuda', 
                 cache_dir: Optional[str] = None, 
                 output_dir: Optional[str] = None, 
                 max_elements: int = 10000,
                 model: Optional[SentenceTransformer] = None):
        """
        Initialize the processor.
        
        Args:
            lang_code: Language code (e.g., 'en', 'de', 'fr')
            device: Device for computation ('cuda' or 'cpu')
            cache_dir: Directory for dataset caching
            output_dir: Directory for output files
            max_elements: Maximum number of elements to process
            model: Pre-loaded sentence transformer model
        """
        self.lang_code = lang_code
        self.device = device
        self.cache_dir = cache_dir
        self.output_dir = output_dir
        self.max_elements = max_elements
        
        # Initialize model
        if model is not None:
            self.model = model
        else:
            try:
                self.model = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
            except Exception as e:
                raise RuntimeError(f"Failed to load sentence transformer model: {e}")
        
        # Create output directory
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)

    def load_data(self, dataset_name: str = "wikimedia/wikipedia", date: str = "20231101") -> None:
        """
        Load and filter dataset data.
        
        Args:
            dataset_name: Name of the dataset
            date: Dataset version date
            
        Raises:
            FileNotFoundError: If filtered data file doesn't exist
            ValueError: If data is invalid
        """
        print(f"Loading dataset {dataset_name} for language {self.lang_code}...")
        
        try:
            # Load main dataset
            print("Loading main Wikipedia dataset...")
            data = load_dataset(dataset_name, f"{date}.{self.lang_code}", cache_dir=self.cache_dir)
            # dataset_size = min(len(data['train']), self.max_elements) if self.max_elements else len(data['train'])
            self.data = data['train']
            self.titles = self.data['title']
            self.texts = self.data['text']
            self.texts = [text.split('\n')[0] if text else "" for text in self.texts]
            print(f"✓ Loaded {len(self.data)} samples from main dataset.")

            # Load filtered data
            print("Loading filtered NER data...")
            filtered_file_path = os.path.join("ner_processed_data_v2", self.lang_code, f"{self.lang_code}_selected_data.json")
            if not os.path.exists(filtered_file_path):
                raise FileNotFoundError(f"Filtered data file not found: {filtered_file_path}")
            
            with open(filtered_file_path, 'r', encoding='utf-8') as f:
                filtered_data = []
                lines = f.readlines()
                for line in tqdm(lines, desc="Processing filtered data"):
                    try:
                        item = json.loads(line.strip())
                        filtered_data.append(item)
                    except json.JSONDecodeError:
                        continue
            
            if not filtered_data:
                raise ValueError(f"No valid filtered data found for language {self.lang_code}")
            
            selected_indices = [item['index'] for item in tqdm(filtered_data, desc="Processing filtered data")]
            # Validate indices
            valid_indices = []
            for idx in selected_indices:
                if 0 <= idx < len(self.titles):
                    valid_indices.append(idx)
                else:
                    print(f"Warning: Invalid index {idx} for dataset size {len(self.titles)}")
            
            # Filter data using valid indices
            self.selected_titles = [self.titles[i] for i in tqdm(valid_indices, desc="Filtering titles")]
            self.selected_texts = [self.texts[i] for i in tqdm(valid_indices,desc="Filtering texts")]
            self.selected_indices = valid_indices
            
            print(f"Selected {len(self.selected_texts)} samples after filtering.")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load data for {self.lang_code}: {e}")

    def compute_embeddings(self, batch_size: int = 64) -> None:
        """
        Compute embeddings for selected texts.
        
        Args:
            batch_size: Batch size for embedding computation
            
        Raises:
            RuntimeError: If embedding computation fails
        """
        if not hasattr(self, 'selected_texts') or not self.selected_texts:
            raise ValueError("No data loaded. Call load_data() first.")
        
        print("Computing embeddings...")
        
        try:
            # Prepare input texts with proper formatting
            input_texts = []
            for title, text in zip(self.selected_titles, self.selected_texts):
                # Clean and format text
                clean_title = title.strip() if title else ""
                clean_text = text.strip() if text else ""
                if clean_title or clean_text:
                    input_texts.append(f"<title> {clean_title} <text> {clean_text}")
            
            if not input_texts:
                raise ValueError("No valid texts to encode")
            
            # Compute embeddings with enhanced progress bar
            print(f"Computing embeddings for {len(input_texts)} texts...")
            self.embeddings = self.model.encode(
                input_texts, 
                batch_size=batch_size, 
                show_progress_bar=True,
                # batch_progress_bar=True,
                convert_to_numpy=True
            )
            
            print(f"Computed embeddings for {len(self.embeddings)} samples.")
            print(f"Embedding shape: {self.embeddings.shape}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to compute embeddings: {e}")

    def cluster_embeddings(self, n_clusters: int = 100) -> None:
        """
        Perform clustering on embeddings.
        
        Args:
            n_clusters: Number of clusters (if None, will be auto-calculated)
            
        Raises:
            RuntimeError: If clustering fails
        """
        if not hasattr(self, 'embeddings') or self.embeddings is None:
            raise ValueError("No embeddings computed. Call compute_embeddings() first.")
        
        # Calculate optimal number of clusters if not provided
        n_clusters = max(2, len(self.embeddings) // 10)
        print(f"Using specified number of clusters: {n_clusters}")
        
        # Ensure we don't have more clusters than samples
        n_clusters = min(n_clusters, len(self.embeddings))
        
        try:
            print(f"Clustering {len(self.embeddings)} embeddings into {n_clusters} clusters...")
            
            # Perform clustering
            kmeans = cuKMeans(n_clusters=n_clusters, random_state=42)
            self.cluster_labels = kmeans.fit_predict(self.embeddings)
            print("Clustering completed.")

            # Group indices by cluster for efficient processing
            cluster_indices = {}
            for idx, label in enumerate(self.cluster_labels):
                if label not in cluster_indices:
                    cluster_indices[label] = []
                cluster_indices[label].append(idx)

            # Prepare cluster data
            cluster_data = []
            for cluster_id in range(n_clusters):
                indices = cluster_indices.get(cluster_id, [])
                cluster_size = len(indices)
                print(f"Cluster {cluster_id}: {cluster_size} samples")
                
                # Extract data for this cluster
                cluster_info = {
                    "cluster_id": cluster_id,
                    "cluster_size": cluster_size,
                    "titles": [self.selected_titles[idx] for idx in indices],
                    "indices": [self.selected_indices[idx] for idx in indices]
                }
                cluster_data.append(cluster_info)
            
            # Save cluster data
            if self.output_dir:
                cluster_file_name = os.path.join(self.output_dir, f"{self.lang_code}_clusters.json")
                with open(cluster_file_name, 'w', encoding='utf-8') as f:
                    json.dump(cluster_data, f, ensure_ascii=False, indent=4)
                print(f"Cluster data saved to: {cluster_file_name}")
            
            print(f"Clustering summary: {len(cluster_data)} clusters created")
            
        except Exception as e:
            raise RuntimeError(f"Failed to perform clustering: {e}")
    

def run() -> None:
    """
    Main function to run the language embedding processing pipeline.
    """
    # Configuration
    # lang_list = ['zh', 'de', 'fr']
    lang_list = ['ja']
    cache_dir = './dataset_cache'
    output_base_dir = './output_single_lang_clusters'
    device = "cuda:0"
    batch_size = 128
    max_elements = None  # Set to a number to limit processing, None for all data
    
    print("Starting language embedding processing pipeline")
    print(f"Languages: {lang_list}")
    print(f"Device: {device}")
    print(f"Output directory: {output_base_dir}")
    
    # Create output directory
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Initialize model once for efficiency
    try:
        model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2', device=device)
        print("Sentence transformer model loaded successfully")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return
    
    # Process each language
    for lang_code in lang_list:
        print(f"\n{'='*50}")
        print(f"Processing language: {lang_code}")
        print(f"{'='*50}")
        
        try:
            processor = LanguageEmbeddingProcessor(
                lang_code=lang_code,
                device=device,
                cache_dir=cache_dir,
                output_dir=output_base_dir,
                max_elements=max_elements,
                model=model
            )
            
            # Load data
            processor.load_data()
            
            # Compute embeddings
            processor.compute_embeddings(batch_size=batch_size)
            
            # Perform clustering
            processor.cluster_embeddings(n_clusters=100)
            
            print(f"Successfully completed processing for {lang_code}")
            
        except Exception as e:
            print(f"Error processing {lang_code}: {e}")
            continue
    
    print(f"\n{'='*50}")
    print("Language embedding processing pipeline completed")
    print(f"{'='*50}")

if __name__ == "__main__":
    run()


