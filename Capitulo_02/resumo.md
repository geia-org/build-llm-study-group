# Resumo do Capítulo 2 - Trabalhando com Dados de Texto

Este capítulo explora o processo de preparação de dados textuais para o treinamento de Modelos de Linguagem de Grande Porte (LLMs), com foco em LLMs baseados na arquitetura _transformer_ do tipo GPT. Abaixo estão os principais pontos abordados:

## Preparação de Texto
- **Tokenização**: O texto bruto é dividido em tokens (palavras, subpalavras ou caracteres especiais) para processamento. Um exemplo prático usa o conto "O Veredito" de Edith Wharton, tokenizado com expressões regulares em Python.
- **Vocabulário e IDs de Token**: Tokens são mapeados para IDs inteiros únicos por meio de um vocabulário construído a partir do conjunto de dados. Um tokenizador simples (`SimpleTokenizerV1`) é implementado para codificar e decodificar texto.

```Python
class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab #1
        self.int_to_str = {i:s for s,i in vocab.items()} #2

    def encode(self, text): #3
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids): #4
        text = " ".join([self.int_to_str[i] for i in ids]) 
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text) #5
        return text
```

## Tokens Especiais
- Tokens como `<|unk|>` (para palavras desconhecidas) e `<|endoftext|>` (para separar textos não relacionados) são adicionados ao vocabulário (`SimpleTokenizerV2`) para lidar com contextos específicos e palavras fora do vocabulário.

```Python
class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = { i:s for s,i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        preprocessed = [item if item in self.str_to_int #1
                        else "<|unk|>" for item in preprocessed]

        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])

        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text) #2
        return text
```

## Codificação de Pares de Bytes (BPE)
- O BPE, utilizado em modelos como GPT-2 e GPT-3, divide palavras desconhecidas em subpalavras ou caracteres, eliminando a necessidade de tokens `<|unk|>`. A biblioteca `tiktoken` é usada para implementar o BPE de forma eficiente.

## Amostragem de Dados
- Uma abordagem de **janela deslizante** é empregada para gerar pares entrada-alvo para a tarefa de previsão da próxima palavra. Um carregador de dados (`GPTDatasetV1` e `DataLoader` do PyTorch) é implementado para processar texto em lotes, retornando tensores de entrada e alvo.

```Python
import torch
from torch.utils.data import Dataset, DataLoader
class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt) #1

        for i in range(0, len(token_ids) - max_length, stride): #2
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):    #3
        return len(self.input_ids)

    def __getitem__(self, idx):         #4
        return self.input_ids[idx], self.target_ids[idx]
```

```Python
def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True,
                         num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2") #1
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride) #2
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last, #3
        num_workers=num_workers #4
    )

    return dataloader
```

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