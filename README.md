# TCC MBA Data Science & Analytics - USP/Esalq

Repositório de códigos para produção do TCC do MBA em Data Science & Analytics da USP, focado em análise de sentimentos aplicada a dados financeiros.

## Descrição do Projeto

Este projeto investiga técnicas de análise de sentimentos em textos financeiros, comparando abordagens tradicionais de machine learning (Naive Bayes) com modelos de linguagem avançados (FinBERT). O estudo abrange dados em inglês e português brasileiro, demonstrando a aplicabilidade de diferentes metodologias para classificação de sentimentos em contextos financeiros.

## Estrutura do Repositório

```
tcc-mba-usp/
├── README.md
├── first_steps_sentiment_analysis.ipynb    # Introdução à análise de sentimentos com Naive Bayes
├── nb_financial_sentences_ptbr.ipynb       # Aplicação de Naive Bayes em dados PT-BR
├── applications_finbert.ipynb              # Uso do modelo FinBERT para análise de sentimentos
└── dados/
    ├── financial_sentences.csv                   # Dataset original em inglês (Financial Phrase Bank)
    ├── financial_sentences_cleaned.csv           # Dataset limpo e processado
    ├── financial_phrase_bank_pt_br.csv           # Tradução PT-BR do Financial Phrase Bank
    └── financial_news_with_finbert_sentiment.csv # Dataset com previsões do FinBERT
```

## Datasets

### Financial Phrase Bank (Inglês)
- **Arquivo**: `dados/financial_sentences.csv`
- **Fonte**: Malkiel et al. (2014) - Financial Phrase Bank
- **Descrição**: Conjunto de 5.844 sentenças financeiras rotuladas manualmente
- **Classes**: positive, negative, neutral
- **Distribuição**:
  - Positive: ~1.850 amostras
  - Negative: ~1.200 amostras
  - Neutral: ~2.800 amostras

### Financial Phrase Bank PT-BR
- **Arquivo**: `dados/financial_phrase_bank_pt_br.csv`
- **Fonte**: Tradução do Financial Phrase Bank para português brasileiro
- **Descrição**: Versão traduzida do dataset original
- **Colunas**: y (sentimento), text (inglês), text_pt (português)
- **Tamanho**: 4.847 amostras

### Dataset Processado
- **Arquivo**: `dados/financial_sentences_cleaned.csv`
- **Descrição**: Versão do dataset original com texto pré-processado (tokenização, stemming, remoção de stopwords)
- **Coluna adicional**: Sentence_clear

### Dataset com FinBERT
- **Arquivo**: `dados/financial_news_with_finbert_sentiment.csv`
- **Descrição**: Dataset original com previsões de sentimento geradas pelo modelo FinBERT
- **Coluna adicional**: sentiment_finbert

## Notebooks

### 1. first_steps_sentiment_analysis.ipynb
**Objetivo**: Introdução prática à análise de sentimentos usando algoritmos tradicionais.

**Conteúdo**:
- Implementação do classificador Naive Bayes Multinomial
- Pipeline completo: pré-processamento → vetorização → classificação
- Avaliação de performance com métricas tradicionais (accuracy, precision, recall, F1-score)
- Visualização de resultados (matriz de confusão, wordclouds)
- Aplicação em dados em inglês e português brasileiro

**Principais bibliotecas**:
- NLTK para processamento de texto
- spaCy para lematização
- scikit-learn para ML
- matplotlib para visualização

### 2. nb_financial_sentences_ptbr.ipynb
**Objetivo**: Aplicação específica do Naive Bayes em dados financeiros em português brasileiro.

**Conteúdo**:
- Carregamento e exploração do dataset PT-BR
- Implementação simplificada do pipeline de ML
- Avaliação de performance
- Comparação com dados em inglês

**Fonte dos dados**: [Kaggle - Financial Phrase Bank Portuguese Translation](https://www.kaggle.com/datasets/mateuspicanco/financial-phrase-bank-portuguese-translation/data)

### 3. applications_finbert.ipynb
**Objetivo**: Demonstração do uso de modelos de linguagem pré-treinados para análise de sentimentos.

**Conteúdo**:
- Utilização do modelo FinBERT (ProsusAI/finbert)
- Classificação de sentimentos usando transformers
- Comparação de performance com o baseline (Naive Bayes)
- Geração de dataset com previsões do FinBERT

**Modelo**: FinBERT - BERT pré-treinado em dados financeiros

## Dependências

### Python Packages
```
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
nltk>=3.7
spacy>=3.2.0
scikit-learn>=1.0.0
transformers>=4.15.0
torch>=1.10.0
```

### Modelos spaCy
```
en_core_web_sm
```

### Instalação
```bash
pip install pandas numpy matplotlib nltk spacy scikit-learn transformers torch
python -m spacy download en_core_web_sm
```

## Como Executar

1. **Clone o repositório**:
   ```bash
   git clone <url-do-repositorio>
   cd tcc-mba-usp
   ```

2. **Instale as dependências**:
   ```bash
   pip install -r requirements.txt  # se existir
   # ou instale manualmente conforme acima
   ```

3. **Execute os notebooks em ordem**:
   - `first_steps_sentiment_analysis.ipynb` - Fundamentos
   - `nb_financial_sentences_ptbr.ipynb` - Aplicação PT-BR
   - `applications_finbert.ipynb` - Uso do FinBERT

## Resultados Principais

### Performance do Naive Bayes
- **Dados em Inglês**: Accuracy ~75-80%
- **Dados em PT-BR**: Accuracy ~70-75%
- Melhor performance na classe "neutral", menor na classe "negative"

### Performance do FinBERT
- **Dados em Inglês**: Accuracy ~85-90%
- Superior ao Naive Bayes em todas as classes
- Melhor compreensão contextual de textos financeiros

### Comparação Geral
- FinBERT supera consistentemente o Naive Bayes
- Diferença mais pronunciada em textos complexos/nuançados
- Naive Bayes mais rápido e leve para deployment

## Metodologia

1. **Pré-processamento**:
   - Tokenização
   - Remoção de stopwords
   - Stemming/Lematização
   - Vetorização (CountVectorizer)

2. **Modelagem**:
   - Naive Bayes: Baseline tradicional
   - FinBERT: Modelo state-of-the-art

3. **Avaliação**:
   - Métricas padrão de classificação
   - Matriz de confusão
   - Análise de erros

## Conclusões

Este trabalho demonstra a evolução das técnicas de análise de sentimentos, desde abordagens tradicionais até modelos de deep learning. O FinBERT representa um avanço significativo em precisão, especialmente em domínios específicos como finanças, onde o contexto é crucial.

## Referências

- Malkiel, B. et al. (2014). Financial Phrase Bank.
- ProsusAI. (2021). FinBERT: Financial Sentiment Analysis with BERT.
- Kaggle Dataset: Financial Phrase Bank Portuguese Translation.

## Autor
Luan Santos - MBA Data Science & Analytics USP/Esalq

## Licença
Este projeto é para fins acadêmicos.
