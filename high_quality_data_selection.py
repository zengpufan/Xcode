import json
import torch
import os
import logging
import warnings
import numpy as np
from datasets import load_dataset
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from scipy.stats import entropy
from multiprocessing import Process

# Suppress warnings
warnings.filterwarnings(action='ignore')
logging.basicConfig(level=logging.ERROR)

class WikiDataCluster:
    def __init__(self, json_file_path, lang_code, model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2", device=None):
        self.json_file_path = json_file_path
        self.lang_code = lang_code
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"[{lang_code}] Device: {self.device}")
        
        try:
            self.model = SentenceTransformer(model_name, device=self.device)
        except Exception:
            # Fallback to CPU
            self.device = "cpu"
            self.model = SentenceTransformer(model_name, device=self.device)
        
        self.data = self.load_json()
        
        # Load specific wikipedia snapshot
        self.dataset = load_dataset("wikimedia/wikipedia", f"")["train"]

    def load_json(self):
        with open(self.json_file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
        
    def get_embedding(self, texts, batch_size=64):
        if not texts: return np.array([])
        with torch.no_grad():
            return self.model.encode(texts, batch_size=batch_size, normalize_embeddings=True, convert_to_numpy=True)
    
    def calculate_avg_distance(self, embeddings, k=5):
        """Calculate average distance to k-nearest neighbors (Local Dispersion)"""
        if len(embeddings) < k:
            return np.zeros(len(embeddings))
        neighbors = NearestNeighbors(n_neighbors=k, metric="cosine").fit(embeddings)
        distances, _ = neighbors.kneighbors(embeddings)
        return np.mean(distances, axis=1)
    
    def split_into_paragraphs(self, text):
        return [p.strip() for p in text.split('\n') if p.strip()]
    
    def calculate_semantic_entropy(self, paragraph_embeddings):
        """Calculate semantic entropy based on self-similarity distribution"""
        normalized = paragraph_embeddings / (np.linalg.norm(paragraph_embeddings, axis=1, keepdims=True) + 1e-8)
        similarity_matrix = np.dot(normalized, normalized.T)
        prob_dist = similarity_matrix / (np.sum(similarity_matrix, axis=1, keepdims=True) + 1e-8)
        prob_dist = np.clip(prob_dist, 1e-8, 1.0)
        return np.mean([entropy(row) for row in prob_dist])
    
    def cluster(self, top_k_output=10):
        processed_count = 0
        
        for cluster in tqdm(self.data, desc=f"[{self.lang_code}] Processing"):
            try:
                if "titles" not in cluster or "indices" not in cluster: continue
                
                indices = cluster["indices"]
                titles = cluster["titles"]
                texts = []
                valid_titles = []
                
                # 1. Retrieve texts (Indices validation)
                for ptr, i in enumerate(indices):
                    try:
                        if 0 <= i < len(self.dataset):
                            text = self.dataset[i]["text"]
                            if text and text.strip():
                                texts.append(text)
                                valid_titles.append(titles[ptr])
                    except Exception: continue
                
                if not texts: continue
                
                # 2. Compute Embeddings
                embeddings = self.get_embedding(texts)
                if len(embeddings) == 0: continue
                
                processed_count += 1
                
                # 3. Calculate Density (Distance)
                avg_distances = self.calculate_avg_distance(embeddings)
                
                # 4. Filter: Keep entries denser than the median (Section 3.2)
                median_dist = np.median(avg_distances)
                density_mask = avg_distances <= median_dist
                
                # Fallback if mask is empty
                if np.sum(density_mask) == 0:
                     density_mask = np.ones(len(avg_distances), dtype=bool)

                filtered_texts = [t for j, t in enumerate(texts) if density_mask[j]]
                filtered_titles = [t for j, t in enumerate(valid_titles) if density_mask[j]]
                filtered_densities = [1.0/(d+1e-8) for j, d in enumerate(avg_distances) if density_mask[j]]
                
                # 5. Calculate Entropy for filtered candidates
                candidates = []
                for j, text in enumerate(filtered_texts):
                    try:
                        paras = self.split_into_paragraphs(text)
                        if len(paras) > 1:
                            para_embeds = self.get_embedding(paras, batch_size=32)
                            sem_entropy = self.calculate_semantic_entropy(para_embeds)
                        else:
                            sem_entropy = 0.0
                        
                        candidates.append({
                            "title": filtered_titles[j],
                            "density": float(filtered_densities[j]),
                            "semantic_entropy": float(sem_entropy)
                        })
                    except Exception: continue

                # 6. Sort by Entropy and select Top-K
                candidates.sort(key=lambda x: x["semantic_entropy"], reverse=True)
                top_results = candidates[:top_k_output]

                cluster["top_density_texts"] = top_results
                cluster["avg_semantic_entropy"] = np.mean([r["semantic_entropy"] for r in top_results]) if top_results else 0.0
                
            except Exception as e:
                print(f"[{self.lang_code}] Error: {e}")
                continue
        
        return self.data
    
    def save_clusters(self, output_file_top, output_file_res):
        # Helper to serialize float32
        def convert_float32(obj):
            if isinstance(obj, dict): return {k: convert_float32(v) for k, v in obj.items()}
            elif isinstance(obj, list): return [convert_float32(i) for i in obj]
            elif isinstance(obj, np.float32): return float(obj)
            return obj
            
        converted_data = convert_float32(self.data)
        
        # Sort clusters by average entropy
        sorted_clusters = sorted(converted_data, key=lambda x: x.get("avg_semantic_entropy", 0), reverse=True)

        top_fraction = int(len(sorted_clusters) * 0.2)
        with open(output_file_top, 'w', encoding='utf-8') as f:
            json.dump(sorted_clusters[:top_fraction], f, ensure_ascii=False, indent=4)

        with open(output_file_res, "w", encoding="utf-8") as f:
            json.dump(sorted_clusters[top_fraction:], f, ensure_ascii=False, indent=4)

def process_lang(lang_code, base_input_dir, output_dir):
    try:
        json_file_path = os.path.join(base_input_dir, f"")
        cluster = WikiDataCluster(json_file_path, lang_code)
        
        cluster.cluster(top_k_output=10)
        
        output_top = os.path.join(output_dir, f"")
        output_res = os.path.join(output_dir, f"")
        cluster.save_clusters(output_top, output_res)
        print(f"[{lang_code}]  Done")
    except Exception as e:
        print(f"[{lang_code}]  Failed: {e}")

if __name__ == "__main__":
    lang_code_list = ["ja", "es", "de", "zh", "en"]
    base_input_dir = ""  
    output_dir = ""  
    
    os.makedirs(output_dir, exist_ok=True)
    
    processes = []
    for lang_code in lang_code_list:
        p = Process(target=process_lang, args=(lang_code, base_input_dir, output_dir))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()