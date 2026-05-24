import numpy as np
import os
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import json
from cuml import KMeans as cuKMeans
from sklearn.preprocessing import StandardScaler
from scipy import stats 

class LanguageEmbeddingProcessor:  
    def __init__(self,
                 lang_code_list: list,  
                 device: str = 'cuda',
                 cache_dir: str = None,
                 output_dir: str = None,
                 max_elements: int = 10000,
                 model: SentenceTransformer = None):
        self.lang_code_list = lang_code_list 
        self.device = device
        self.cache_dir = cache_dir
        self.output_dir = output_dir
        self.max_elements = max_elements

        self.model = model if model is not None else SentenceTransformer(
            'sentence-transformers/paraphrase-multilingual-mpnet-base-v2', device=self.device
        )
      
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.multi_lang_data = {
            "lang_code": [],
            "title": [],
            "text": [],
            "embedding": []  
        }

    def load_data(self, lang_code: str, dataset_name: str = "wikimedia/wikipedia", date: str = "20231101"):


        try:
            
            data = load_dataset(
                dataset_name,
                f"{date}.{lang_code}",
                cache_dir=self.cache_dir,
                
            )
        except Exception as e:
            return None
        indices_list=[]
        with open(f"",'r',encoding='utf-8') as f:
            cluster_data=json.load(f)
            for item in cluster_data:
                indices_list.extend(item['indices'])

        data=data['train'].select(indices_list)
        
        clean_data = []
        for idx, (title, text) in enumerate(tqdm(zip(data['title'], data['text']), 
                                               total=len(data))):
            if not text or not title:
                continue
            
            clean_text = text.split('\n')[0].strip()
            if len(clean_text) < 10:  
                continue
            clean_data.append({
                "lang_code": lang_code,
                "title": title.strip(),
                "text": clean_text,
                "index": idx
            })

        return clean_data

    def remove_embedding_outliers(self, embeddings, z_threshold=3):
  
        
        z_scores = np.abs(stats.zscore(embeddings))
        
        max_z_scores = np.max(z_scores, axis=1)
        
        normal_indices = np.where(max_z_scores <= z_threshold)[0]
        filtered_embeddings = embeddings[normal_indices]

        return filtered_embeddings, normal_indices

    def compute_embeddings(self, text_data):

        
        input_texts = [f"<title> {item['title']} <text> {item['text']}" for item in text_data]
        
        embeddings = self.model.encode(
            input_texts,
            batch_size=64,
            show_progress_bar=True,
            convert_to_numpy=True,
            device=self.device
        )
        
        
        
        
        filtered_embeddings=embeddings
        filtered_text_data=text_data
        return filtered_embeddings, filtered_text_data

    def compute_cluster(self, embeddings, text_data, k=100):

        
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(embeddings)

        
        kmeans = cuKMeans(n_clusters=k, random_state=42, verbose=0)
        cluster_labels = kmeans.fit_predict(embeddings_scaled)

        
        cluster_results = []
        for cluster_id in tqdm(range(k)):
            
            cluster_sample_indices = np.where(cluster_labels == cluster_id)[0]
            cluster_samples = [text_data[idx] for idx in cluster_sample_indices]
            sample_count = len(cluster_samples)

            
            if sample_count < 5:
                continue

            
            lang_distribution = {}
            for sample in cluster_samples:
                lang = sample['lang_code']
                lang_distribution[lang] = lang_distribution.get(lang, 0) + 1

            
            cluster_results.append({
                "cluster_id": int(cluster_id),
                "sample_count": sample_count,
                "lang_distribution": lang_distribution,
                "samples": cluster_samples  
            })

     
        with open(os.path.join(self.output_dir, "multi_lang_cluster_raw.json"), 'w', encoding='utf-8') as f:
            json.dump(cluster_results, f, ensure_ascii=False, indent=2)


        return cluster_results

    def filter_single_lang_cluster(self, cluster_results, max_distribution_ratio=0.8):

        single_lang_clusters = []
        for cluster in tqdm(cluster_results):
            lang_dist = cluster['lang_distribution']
            total_samples = cluster['sample_count']
            if total_samples == 0:
                continue

            
            max_lang = max(lang_dist, key=lang_dist.get)
            max_lang_ratio = lang_dist[max_lang] / total_samples

            
            if max_lang_ratio >= max_distribution_ratio:
                cluster['dominant_lang'] = max_lang
                cluster['dominant_lang_ratio'] = round(max_lang_ratio, 3)
                single_lang_clusters.append(cluster)


        with open(os.path.join(self.output_dir, "single_lang_clusters.json"), 'w', encoding='utf-8') as f:
            json.dump(single_lang_clusters, f, ensure_ascii=False, indent=2)

        lang_cluster_count = {}
        for cluster in single_lang_clusters:
            lang = cluster['dominant_lang']
            lang_cluster_count[lang] = lang_cluster_count.get(lang, 0) + 1
        for lang, count in lang_cluster_count.items():
            print(f"{lang} : {count}")

        return single_lang_clusters

    def process(self, k=100, single_lang_ratio=0.8):

        
        all_clean_data = []
        for lang_code in tqdm(self.lang_code_list):
            lang_data = self.load_data(lang_code=lang_code)
            if lang_data is not None:
                all_clean_data.extend(lang_data)
    

        
        embeddings, filtered_text_data = self.compute_embeddings(all_clean_data)

        
        cluster_results = self.compute_cluster(embeddings, filtered_text_data, k=k)

        
        self.filter_single_lang_cluster(cluster_results, max_distribution_ratio=single_lang_ratio)

  

def run():
    
    lang_code_list = ['en','es','ja','zh','fr','de']  
    cache_dir = ''  
    output_dir = ''  
    max_elements = 10000  
    cluster_k = 1000  
    single_lang_ratio = 0.8  

    
    processor = LanguageEmbeddingProcessor(
        lang_code_list=lang_code_list,
        device='cuda' ,  
        cache_dir=cache_dir,
        output_dir=output_dir,
        max_elements=max_elements
    )

    
    processor.process(k=cluster_k, single_lang_ratio=single_lang_ratio)

if __name__ == "__main__":
    
    import warnings
    warnings.filterwarnings('ignore')
    run()
