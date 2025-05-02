# 🎵 Music Cover Identifier with Machine Learning

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Sentence%20Transformers-orange.svg)
![NLP](https://img.shields.io/badge/NLP-TF--IDF%20%2B%20Embeddings-green.svg)

## 📌 Visão Geral

O **Music Cover Identifier** é um sistema inteligente que utiliza técnicas de NLP e Machine Learning para identificar automaticamente se uma letra de música é original ou uma versão cover, além de encontrar a música original correspondente no caso de covers.

O projeto combina:
- **Embeddings semânticos** com Sentence Transformers
- **TF-IDF tradicional** para comparações rápidas
- **Similaridade de cosseno** em espaços vetoriais de alta dimensão

## ✨ Features Principais

- 🔍 Identificação automática de covers musicais
- 📊 Análise de similaridade entre letras de música
- 🌐 Busca simulada de músicas online (pronta para integração com APIs reais)
- 🧠 Modelos de machine learning state-of-the-art
- 💾 Sistema de armazenamento em memória (facilmente expansível para banco de dados)

## 🛠️ Como Instalar e Executar

### Pré-requisitos

- Python 3.8 ou superior
- pip (gerenciador de pacotes do Python)

### Instalação

1. Clone o repositório:
```bash
git clone https://github.com/seu-usuario/music-cover-identifier.git
cd music-cover-identifier
```

2. É recomendado que crie um ambiente virtual:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate    # Windows
```

3. Instale as dependências:
```bash
pip install -r requirements.txt
```

Se der errado dessa forma, faça:
```bash
pip install sentence-transformers scikit-learn pandas numpy requests
```

4. Para executar:
```bash
python music_cover_identifier.py
```

## 🎮 Como Usar

O sistema oferece um menu interativo com 3 opções principais. Para iniciar, execute:

```bash
python music_cover_identifier.py
```

## 1. Identificar uma música

Selecione a opção 1 no menu
Digite a letra da música linha por linha
Para finalizar, digite 'FIM' em uma linha separada
O sistema irá analisar e mostrar:
- Se é uma música original ou cover
- O grau de similaridade com outras músicas
- A música original correspondente (se for cover)

## 2. Buscar músicas online

Selecione a opção 2 no menu
Digite o nome da música que deseja buscar
Escolha entre os resultados simulados
Visualize os detalhes da letra

## 3. Sair do programa

Selecione a opção 3 para encerrar
Ou pressione Ctrl+C no terminal
Opção para adicionar à base de dados

## Comandos adicionais

Para ver o menu de ajuda durante a execução:
Pressione 'h' + Enter em qualquer momento

Para limpar a tela durante a execução:
Pressione 'c' + Enter

Para visualizar as músicas cadastradas:
Pressione 'l' + Enter

Para sair rapidamente:
Pressione 'q' + Enter

## 📚 Documentação das Principais

FunçõesMusicCoverIdentifier (Classe Principal)
## __init__()
- Inicializa o modelo de embeddings e carrega dados de exemplo

- Configura os dicionários para armazenar músicas originais e covers

## _preprocess_text(text)
Pré-processamento do texto:

- Conversão para minúsculas

- Remoção de caracteres especiais

- Normalização de espaços

## find_similar_songs(lyrics, threshold=0.7)
Método principal de identificação:

- Pré-processa a letra inserida

- Calcula similaridade usando TF-IDF e embeddings

- Combina os resultados

- Retorna o melhor match com metadados

## add_new_song(title, lyrics, is_cover=False, original_title=None)
Adiciona novas músicas à base de dados:

- Armazena em memória (original_songs ou covers)

- Aplica pré-processamento automático

## search_online_songs(query)
- Simula busca online (pronta para integração com APIs reais)

- Retorna resultados formatados

## 🤝 Como Contribuir
1. Faça um fork do projeto

2. Crie uma branch para sua feature (git checkout -b feature/AmazingFeature)

3; Commit suas mudanças (git commit -m 'Add some AmazingFeature')

4. Push para a branch (git push origin feature/AmazingFeature)

5. Abra um Pull Request

## 📄 Licença
Distribuído sob a licença MIT. Veja LICENSE para mais informações.
