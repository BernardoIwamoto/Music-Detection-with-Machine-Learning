import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd

class MusicCoverIdentifier:
    def __init__(self):
        # Modelo de linguagem para embeddings avançados
        self.embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        
        # Dados das músicas (substituir por API real posteriormente)
        self.original_songs = {}
        self.covers = {}
        
        # Inicializar com alguns exemplos
        self._load_sample_data()
    
    def _load_sample_data(self):
        """Carrega dados de exemplo de uma API ou dataset público"""
        try:
            # Exemplo: buscar dados da Genius API (implementação real precisaria de chave API)
            # response = requests.get(f"https://api.genius.com/songs/...")
            # data = response.json()
            
            # Dados de exemplo (substituir por chamada real à API)
            self.original_songs = {
                "Bohemian Rhapsody": "Is this the real life is this just fantasy",
                "Imagine": "Imagine there's no heaven it's easy if you try",
                "Yesterday": "Yesterday all my troubles seemed so far away"
            }
            
            self.covers = {
                "Bohemian Rhapsody (Cover)": "Is this the real life is this just fantasy",
                "Imagine (Jazz Version)": "Imagine there no heaven it easy if you try"
            }
            
        except Exception as e:
            print(f"Erro ao carregar dados: {e}")
            # Continuar com os dados de exemplo
    
    def _preprocess_text(self, text):
        """Pré-processamento básico do texto"""
        text = text.lower()
        text = ''.join([c for c in text if c.isalpha() or c == ' '])
        return ' '.join(text.split())
    
    def find_similar_songs(self, lyrics, threshold=0.7):
        """Encontra músicas similares usando embeddings e similaridade de cosseno"""
        processed_lyrics = self._preprocess_text(lyrics)
        
        # Método 1: TF-IDF tradicional (rápido para comparações simples)
        all_songs = list(self.original_songs.items()) + list(self.covers.items())
        song_titles = [title for title, _ in all_songs]
        song_lyrics = [self._preprocess_text(lyr) for _, lyr in all_songs]
        
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([processed_lyrics] + song_lyrics)
        
        # Calcular similaridades
        similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
        
        # Método 2: Usar Sentence Transformers para embeddings semânticos
        embeddings = self.embedding_model.encode([processed_lyrics] + song_lyrics)
        semantic_similarities = cosine_similarity(embeddings[0:1], embeddings[1:]).flatten()
        
        # Combinar os resultados (pesos podem ser ajustados)
        combined_scores = 0.4 * similarities + 0.6 * semantic_similarities
        
        # Encontrar a melhor correspondência
        best_match_idx = np.argmax(combined_scores)
        best_score = combined_scores[best_match_idx]
        best_match = song_titles[best_match_idx]
        
        # Determinar se é um cover ou original
        is_cover = best_match in self.covers
        original_song = None
        
        if is_cover:
            # Se for cover, encontrar a música original correspondente
            for orig_title, orig_lyrics in self.original_songs.items():
                orig_embedding = self.embedding_model.encode([self._preprocess_text(orig_lyrics)])[0]
                cover_embedding = self.embedding_model.encode([self._preprocess_text(self.covers[best_match])])[0]
                sim = cosine_similarity([orig_embedding], [cover_embedding])[0][0]
                if sim > 0.8:  # Threshold para considerar como cover
                    original_song = orig_title
                    break
        
        return {
            'best_match': best_match,
            'score': best_score,
            'is_cover': is_cover,
            'original_song': original_song if is_cover else None,
            'above_threshold': best_score > threshold
        }
    
    def add_new_song(self, title, lyrics, is_cover=False, original_title=None):
        """Adiciona uma nova música à base de dados"""
        processed_lyrics = self._preprocess_text(lyrics)
        
        if is_cover and original_title:
            self.covers[title] = processed_lyrics
            print(f"✅ Cover '{title}' adicionado à base de dados, associado a '{original_title}'")
        else:
            self.original_songs[title] = processed_lyrics
            print(f"✅ Música original '{title}' adicionada à base de dados")
    
    def search_online_songs(self, query):
        """Busca músicas online (implementação simulada)"""
        print(f"🔍 Buscando músicas online para: '{query}'")
        # Implementação real usaria uma API como Genius, Spotify, etc.
        return {
            "results": [
                {"title": f"{query} (Official)", "lyrics": "Sample lyrics for official version"},
                {"title": f"{query} (Cover)", "lyrics": "Sample lyrics for cover version"}
            ]
        }


def main():
    print("🎵 Identificador de Covers Musicais com Machine Learning 🎵")
    identifier = MusicCoverIdentifier()
    
    while True:
        print("\n1. Identificar uma música")
        print("2. Buscar músicas online")
        print("3. Sair")
        choice = input("Escolha uma opção: ")
        
        if choice == '1':
            print("\nDigite a letra da música (digite 'FIM' em uma linha separada para terminar):")
            lyrics_lines = []
            while True:
                line = input()
                if line.strip().upper() == 'FIM':
                    break
                lyrics_lines.append(line)
            
            lyrics = '\n'.join(lyrics_lines)
            if not lyrics.strip():
                print("Letra vazia. Tente novamente.")
                continue
            
            result = identifier.find_similar_songs(lyrics)
            
            if result['above_threshold']:
                if result['is_cover']:
                    print(f"\n🎶 Esta música parece ser um cover de: {result['original_song']}")
                    print(f"📊 Similaridade: {result['score']:.2%}")
                    
                    title = input("Digite um nome para este cover: ")
                    identifier.add_new_song(title, lyrics, is_cover=True, original_title=result['original_song'])
                else:
                    print(f"\n🎵 Música similar encontrada: {result['best_match']}")
                    print(f"📊 Similaridade: {result['score']:.2%}")
                    
                    title = input("Digite um nome para esta música original: ")
                    identifier.add_new_song(title, lyrics)
            else:
                print("\n🎉 Esta música parece ser nova e original!")
                title = input("Digite um nome para esta música original: ")
                identifier.add_new_song(title, lyrics)
        
        elif choice == '2':
            query = input("Digite o nome da música para buscar online: ")
            results = identifier.search_online_songs(query)
            
            print("\nResultados encontrados:")
            for i, song in enumerate(results['results'], 1):
                print(f"{i}. {song['title']}")
            
            song_choice = input("Escolha uma música para ver detalhes (ou 0 para voltar): ")
            if song_choice.isdigit() and 0 < int(song_choice) <= len(results['results']):
                selected = results['results'][int(song_choice)-1]
                print(f"\n🎤 {selected['title']}")
                print(f"\n{selected['lyrics']}\n")
                
                add = input("Deseja adicionar esta música à base de dados? (s/n): ").lower()
                if add == 's':
                    is_cover = "(Cover)" in selected['title']
                    original = None
                    if is_cover:
                        original = input("Digite o título da música original: ")
                    identifier.add_new_song(
                        selected['title'], 
                        selected['lyrics'], 
                        is_cover=is_cover, 
                        original_title=original
                    )
        
        elif choice == '3':
            print("Até logo! 👋")
            break
        else:
            print("Opção inválida. Tente novamente.")


if __name__ == '__main__':
    main()