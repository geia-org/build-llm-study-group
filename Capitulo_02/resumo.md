# Resumo do Capítulo 2 - Trabalhando com Dados de Texto

Este capítulo explora o processo de preparação de dados textuais para o treinamento de Modelos de Linguagem de Grande Porte (LLMs), com foco em LLMs baseados na arquitetura _transformer_ do tipo GPT. Abaixo estão os principais pontos abordados:

## Preparação de Texto
- **Tokenização**: O texto bruto é dividido em tokens (palavras, subpalavras ou caracteres especiais) para processamento. Um exemplo prático usa o conto "O Veredito" de Edith Wharton, tokenizado com expressões regulares em Python.
- **Vocabulário e IDs de Token**: Tokens são mapeados para IDs inteiros únicos por meio de um vocabulário construído a partir do conjunto de dados. Um tokenizador simples (`SimpleTokenizerV1`) é implementado para codificar e decodificar texto.

## Tokens Especiais
- Tokens como `<|unk|>` (para palavras desconhecidas) e `<|endoftext|>` (para separar textos não relacionados) são adicionados ao vocabulário (`SimpleTokenizerV2`) para lidar com contextos específicos e palavras fora do vocabulário.

## Codificação de Pares de Bytes (BPE)
- O BPE, utilizado em modelos como GPT-2 e GPT-3, divide palavras desconhecidas em subpalavras ou caracteres, eliminando a necessidade de tokens `<|unk|>`. A biblioteca `tiktoken` é usada para implementar o BPE de forma eficiente.

## Amostragem de Dados
- Uma abordagem de **janela deslizante** é empregada para gerar pares entrada-alvo para a tarefa de previsão da próxima palavra. Um carregador de dados (`GPTDatasetV1` e `DataLoader` do PyTorch) é implementado para processar texto em lotes, retornando tensores de entrada e alvo.

## Embeddings
- **Embeddings de Tokens**: IDs de token são convertidos em vetores contínuos usando uma camada de _embedding_ do PyTorch, inicializada com valores aleatórios e otimizada durante o treinamento.
- **Embeddings Posicionais**: Para compensar a falta de noção de ordem no mecanismo de autoatenção dos LLMs, embeddings posicionais absolutos são adicionados aos embeddings de tokens, codificando a posição de cada token na sequência.

## Resumo Geral
O capítulo detalha as etapas essenciais para transformar texto bruto em representações vetoriais adequadas para LLMs:
1. Tokenização e conversão em IDs de token.
2. Uso de tokens especiais para contextos específicos.
3. Aplicação de BPE para lidar com vocabulários dinâmicos.
4. Geração de pares entrada-alvo com janela deslizante.
5. Criação de embeddings de tokens e posicionais para alimentar as camadas do LLM.

Os conceitos são ilustrados com exemplos práticos em Python, utilizando bibliotecas como `re`, `tiktoken`, e `PyTorch`, preparando o terreno para a implementação de LLMs nos capítulos subsequentes.