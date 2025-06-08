# Cap 2 - Trabalhando com dados de texto

Este capítulo abrange

* Preparando texto para treinamento de modelo de linguagem grande
* Dividir texto em tokens de palavras e subpalavras
* Codificação de pares de bytes como uma forma mais avançada de tokenizar texto
* Amostragem de exemplos de treinamento com uma abordagem de janela deslizante
* Convertendo tokens em vetores que alimentam um grande modelo de linguagem

Até agora, abordamos a estrutura geral de modelos de linguagem de grande porte (LLMs) e aprendemos que eles são pré-treinados em grandes quantidades de texto. Especificamente, nosso foco foi em LLMs somente decodificadores baseados na arquitetura do _transformer_, que fundamenta os modelos usados no ChatGPT e outros LLMs populares do tipo GPT.

Durante a fase de pré-treinamento, os LLMs processam texto, uma palavra de cada vez. Treinar LLMs com milhões a bilhões de parâmetros usando uma tarefa de previsão da próxima palavra produz modelos com capacidades impressionantes. Esses modelos podem então ser ajustados para seguir instruções gerais ou executar tarefas específicas. Mas antes de implementar e treinar os LLMs, precisamos preparar o conjunto de dados de treinamento, conforme ilustrado na figura 1.

![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781633437166/files/Images/2-1.png)

Figura 1 Os três estágios principais da codificação de um LLM. Este capítulo se concentra na etapa 1 do estágio 1: implementação do pipeline de amostra de dados.

Você aprenderá a preparar texto de entrada para o treinamento de LLMs. Isso envolve dividir o texto em tokens individuais de palavras e subpalavras, que podem então ser codificados em representações vetoriais para o LLM. Você também aprenderá sobre esquemas avançados de tokenização, como a codificação de pares de bytes, utilizada em LLMs populares como o GPT. Por fim, implementaremos uma estratégia de amostragem e carregamento de dados para produzir os pares de entrada-saída necessários para o treinamento de LLMs.

## Compreendendo os embeddings de palavras

Modelos de redes neurais profundas, incluindo LLMs, não conseguem processar texto bruto diretamente. Como o texto é categórico, ele não é compatível com as operações matemáticas usadas para implementar e treinar redes neurais. Portanto, precisamos de uma maneira de representar palavras como vetores de valores contínuos.

Nota: Leitores não familiarizados com vetores e tensores em um contexto computacional podem aprender mais no apêndice A, seção A.2.

O conceito de conversão de dados em formato vetorial é frequentemente chamado de _embedding_ (em português: incorporação). Usando uma camada de rede neural específica ou outro modelo de rede neural pré-treinado, podemos criar _embedding_ com diferentes tipos de dados — por exemplo, vídeo, áudio e texto, conforme ilustrado na Figura 2. No entanto, é importante observar que diferentes formatos de dados exigem modelos de _embedding_ distintos. Por exemplo, um modelo de _embedding_ projetado para texto não seria adequado para criar _embedding_ com dados de áudio ou vídeo.

![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781633437166/files/Images/2-2.png)

Figura 2 Modelos de aprendizado profundo não conseguem processar formatos de dados como vídeo, áudio e texto em sua forma bruta. Portanto, usamos um modelo de _embedding_ para transformar esses dados brutos em uma representação vetorial densa que as arquiteturas de aprendizado profundo podem facilmente compreender e processar. Especificamente, esta figura ilustra o processo de conversão de dados brutos em um vetor numérico tridimensional.

Em essência, uma _embedding_ é um mapeamento de objetos discretos, como palavras, imagens ou até mesmo documentos inteiros, para pontos em um espaço vetorial contínuo — o objetivo principal das incorporações é converter dados não numéricos em um formato que as redes neurais possam processar.

Embora a _word embeddings_ seja a forma mais comum de _embedding_ de texto, também existem incorporações para frases, parágrafos ou documentos inteiros. A _sentence embeddings_, _paragraphs embeddings_ ou _documents  embeddings_ é uma escolha popular para geração de texto com _retrieval-augmented generation_.A _retrieval-augmented generation_ combina geração (como a produção de texto) com recuperação (como a busca em uma base de conhecimento externa) para extrair informações relevantes durante a geração de texto, uma técnica que está além do escopo deste livro. Como nosso objetivo é treinar LLMs semelhantes ao GPT, que aprendem a gerar texto palavra por palavra, vamos nos concentrar na _word embeddings_.

Diversos algoritmos e frameworks foram desenvolvidos para gerar embeddings de palavras. Um dos exemplos mais antigos e populares é a abordagem _Word2Vec_ A arquitetura de rede neural treinada pelo Word2Vec gera embeddings de palavras prevendo o contexto de uma palavra, considerando a palavra-alvo, ou vice-versa. A ideia principal por trás do Word2Vec é que palavras que aparecem em contextos semelhantes tendem a ter significados semelhantes. Consequentemente, quando projetadas em embeddings de palavras bidimensionais para fins de visualização, termos semelhantes são agrupados, como mostrado na figura 3.

![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781633437166/files/Images/2-3.png)

Figura 3 Se os embeddings de palavras forem bidimensionais, podemos plotá-los em um gráfico de dispersão bidimensional para fins de visualização, como mostrado aqui. Ao usar técnicas de embedding de palavras, como Word2Vec, palavras correspondentes a conceitos semelhantes frequentemente aparecem próximas umas das outras no espaço de embedding. Por exemplo, diferentes tipos de pássaros aparecem mais próximos uns dos outros no espaço de embedding do que em países e cidades.

Embeddings de palavras podem ter dimensões variadas, de um a milhares. Uma dimensionalidade maior pode capturar relações mais sutis, mas à custa da eficiência computacional.

Embora possamos usar modelos pré-treinados, como o Word2Vec, para gerar embeddings para modelos de aprendizado de máquina, os LLMs geralmente produzem seus próprios embeddings, que fazem parte da camada de entrada e são atualizados durante o treinamento. A vantagem de otimizar os embeddings como parte do treinamento do LLM, em vez de usar o Word2Vec, é que eles são otimizados para a tarefa e os dados específicos em questão. Implementaremos essas camadas de embedding posteriormente neste capítulo. (Os LLMs também podem criar embeddings de saída contextualizados, como discutiremos no Capítulo 3.)

Infelizmente, embeddings de alta dimensão apresentam um desafio para a visualização porque nossa percepção sensorial e representações gráficas comuns são inerentemente limitadas a três dimensões ou menos, razão pela qual a figura 3 mostra embeddings bidimensionais em um gráfico de dispersão bidimensional. No entanto, ao trabalhar com LLMs, normalmente usamos embeddings com uma dimensionalidade muito maior. Tanto para GPT-2 quanto para GPT-3, o tamanho do embedding (frequentemente chamado de dimensionalidade dos estados ocultos do modelo) varia com base na variante e no tamanho específicos do modelo. É uma compensação entre desempenho e eficiência. Os menores modelos GPT-2 (parâmetros 117M e 125M) usam um tamanho de embedding de 768 dimensões para fornecer exemplos concretos. O maior modelo GPT-3 (parâmetros 175B) usa um tamanho de embedding de 12.288 dimensões.

A seguir, veremos as etapas necessárias para preparar os embeddings usados por um LLM, que incluem dividir o texto em palavras, converter palavras em tokens e transformar tokens em vetores de _embedding_.

## Tokenização de texto

Vamos discutir como dividimos o texto de entrada em tokens individuais, uma etapa de pré-processamento necessária para criar embeddings para um LLM. Esses tokens são palavras individuais ou caracteres especiais, incluindo caracteres de pontuação, conforme mostrado na Figura 4.

![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781633437166/files/Images/2-4.png "figura")

Figura 4 Uma visão das etapas de processamento de texto no contexto de um LLM. Aqui, dividimos um texto de entrada em tokens individuais, que são palavras ou caracteres especiais, como caracteres de pontuação.

O texto que tokenizaremos para o treinamento em LLM é "O Veredito", um conto de Edith Wharton, que foi lançado em domínio público e, portanto, pode ser usado para tarefas de treinamento em LLM. O texto está disponível no Wikisource em <https://en.wikisource.org/wiki/The_Verdict>, e você pode copiá-lo e colá-lo em um arquivo de texto. Eu o copiei para um arquivo com o nome `"the-verdict.txt"`.

Alternativamente, você pode encontrar este arquivo (`"the-verdict.txt"`) no repositório GitHub deste livro em <https://mng.bz/Adng>. Você pode baixar o arquivo com o seguinte código Python:

```Python
import urllib.request
url = ("""https://raw.githubusercontent.com/rasbt/
		LLMs-from-scratch/main/ch02/01_main-chapter-code
        /the-verdict.txt""")
file_path = "the-verdict.txt"
urllib.request.urlretrieve(url, file_path)
```

Em seguida, podemos carregar o arquivo `the-verdict.txt` usando os utilitários de leitura de arquivos padrão do Python.

Listagem 1 Leitura de um conto como exemplo de texto em Python

```Python
with open("the-verdict.txt", "r", encoding="utf-8") as f:
	raw_text = f.read()
print("Total number of character:", len(raw_text))
print(raw_text[:99])
```

O comando ``print`` imprime o número total de caracteres seguido pelos primeiros 99 caracteres deste arquivo para fins ilustrativos:

```Python
Total number of character: 20479
I HAD always thought Jack Gisburn rather a cheap genius-
-though a good fellow enough--so it was no
```

Nosso objetivo é tokenizar esse conto de 20.479 caracteres em palavras individuais e caracteres especiais que podemos então transformar em incorporações para treinamento de LLM.

Observação: É comum processar milhões de artigos e centenas de milhares de livros — muitos gigabytes de texto — ao trabalhar com LLMs. No entanto, para fins educacionais, é suficiente trabalhar com amostras de texto menores, como um único livro, para ilustrar as principais ideias por trás das etapas de processamento de texto e possibilitar a execução em um tempo razoável em hardware de consumo.

Como podemos dividir este texto da melhor forma para obter uma lista de tokens? Para isso, faremos uma pequena excursão e usaremos a biblioteca de expressões regulares do Python `re` para fins ilustrativos. (Você não precisa aprender ou memorizar nenhuma sintaxe de expressão regular, pois mais tarde faremos a transição para um tokenizador pré-construído.)

Usando um texto de exemplo simples, podemos usar o comando `re.split` com a seguinte sintaxe para dividir um texto em caracteres de espaço em branco:

```Python
import re
text = "Hello, world. This, is a test."
result = re.split(r'(\s)', text)
print(result)
```

O resultado é uma lista de palavras individuais, espaços em branco e caracteres de pontuação:

```Python
['Hello,', ' ', 'world.', ' ', 'This,', ' ', 'is', ' ', 'a', 
 ' ', 'test.']
```

Este esquema simples de tokenização funciona principalmente para separar o texto de exemplo em palavras individuais; no entanto, algumas palavras ainda estão conectadas a caracteres de pontuação que queremos ter como entradas separadas na lista. Também evitamos usar todo o texto em minúsculas, pois o uso de letras maiúsculas ajuda os LLMs a distinguir entre nomes próprios e substantivos comuns, a entender a estrutura das frases e a aprender a gerar texto com letras maiúsculas corretamente.

Vamos modificar as divisões da expressão regular em espaços em branco (`\s`), vírgulas e pontos (`[,.]`):

```Python
result = re.split(r'([,.]|\s)', text)
print(result)
```

Podemos ver que as palavras e os caracteres de pontuação agora são entradas de lista separadas, exatamente como queríamos:

```Python
['Hello', ',', '', ' ', 'world', '.', '', ' ', 'This', ',', 
 '', ' ', 'is', ' ', 'a', ' ', 'test', '.', '']
```

Um pequeno problema remanescente é que a lista ainda inclui caracteres de espaço em branco. Opcionalmente, podemos remover esses caracteres redundantes com segurança da seguinte maneira:

```Python
result = [item for item in result if item.strip()]
print(result)
```

A saída resultante sem espaços em branco se parece com o seguinte:

```Python
['Hello', ',', 'world', '.', 'This', ',', 'is', 'a', 'test', '.']
```

Observação: Ao desenvolver um tokenizador simples, a codificação dos espaços em branco como caracteres separados ou sua simples remoção depende da aplicação e de seus requisitos. A remoção dos espaços em branco reduz os requisitos de memória e computação. No entanto, manter os espaços em branco pode ser útil se treinarmos modelos que sejam sensíveis à estrutura exata do texto (por exemplo, código Python, que é sensível a recuo e espaçamento). Aqui, removemos os espaços em branco para simplificar e tornar mais concisas as saídas tokenizadas. Posteriormente, adotaremos um esquema de tokenização que inclua espaços em branco.

O esquema de tokenização que criamos aqui funciona bem no texto de exemplo simples. Vamos modificá-lo um pouco mais para que ele também possa lidar com outros tipos de pontuação, como pontos de interrogação, aspas e os travessões que vimos anteriormente nos primeiros 100 caracteres do conto de Edith Wharton, além de caracteres especiais adicionais:

```Python
text = "Hello, world. Is this-- a test?"
result = re.split(r'([,.:;?_!"()\']|--|\s)', text)
result = [item.strip() for item in result if item.strip()]
print(result)
```

A saída resultante é:

```Python
['Hello', ',', 'world', '.', 'Is', 'this', '--', 'a', 'test', '?']
```

Como podemos ver com base nos resultados resumidos na figura 5, nosso esquema de tokenização agora pode manipular os vários caracteres especiais no texto com sucesso.

![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781633437166/files/Images/2-5.png)

Figura 5 O esquema de tokenização que implementamos até agora divide o texto em palavras individuais e caracteres de pontuação. Neste exemplo específico, o texto de exemplo é dividido em 10 tokens individuais.

Agora que temos um tokenizador básico funcionando, vamos aplicá-lo ao conto inteiro de Edith Wharton:

```Python
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for 
                	item in preprocessed if item.strip()]
print(len(preprocessed))
```

Esta instrução print retorna `4690`, que é o número de tokens neste texto (sem espaços em branco). Vamos imprimir os primeiros 30 tokens para uma verificação visual rápida:

```Python
print(preprocessed[:30])
```

A saída resultante mostra que nosso tokenizador parece estar lidando bem com o texto, já que todas as palavras e caracteres especiais estão claramente separados:

```Python
['I', 'HAD', 'always', 'thought', 'Jack', 'Gisburn', 'rather', 'a',
'cheap', 'genius', '--', 'though', 'a', 'good', 'fellow', 'enough',
'--', 'so', 'it', 'was', 'no', 'great', 'surprise', 'to', 'me', 'to',
'hear', 'that', ',', 'in']
```

## Convertendo tokens em IDs de token

Em seguida, vamos converter esses tokens de uma string Python para uma representação inteira para produzir os IDs dos tokens. Essa conversão é uma etapa intermediária antes de converter os IDs dos tokens em vetores de _embedding_.

Para mapear os tokens gerados anteriormente em IDs de token, precisamos primeiro construir um vocabulário. Esse vocabulário define como mapeamos cada palavra única e caractere especial para um inteiro único, conforme mostrado na Figura 6.

![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781633437166/files/Images/2-6.png)

Figura 6 Construímos um vocabulário tokenizando todo o texto de um conjunto de dados de treinamento em tokens individuais. Esses tokens individuais são então classificados em ordem alfabética e os tokens duplicados são removidos. Os tokens únicos são então agregados em um vocabulário que define um mapeamento de cada token único para um valor inteiro único. O vocabulário representado é propositalmente pequeno e não contém pontuação ou caracteres especiais para simplificar.

Agora que tokenizamos o conto de Edith Wharton e o atribuímos a uma variável Python chamada `preprocessed`, vamos criar uma lista de todos os tokens exclusivos e classificá-los em ordem alfabética para determinar o tamanho do vocabulário:

```Python
all_words = sorted(set(preprocessed))
vocab_size = len(all_words)
print(vocab_size)
```

Depois de determinar que o tamanho do vocabulário é 1.130 por meio desse código, criamos o vocabulário e imprimimos suas primeiras 51 entradas para fins ilustrativos.

Listagem 2 Criando um vocabulário

```Python
vocab = {token:integer for integer,token in enumerate(all_words)}
for i, item in enumerate(vocab.items()):
    print(item)
    if i >= 50:
        break
```

A saída é

```Python
('!', 0)
('"', 1)
("'", 2)
...
('Her', 49)
('Hermia', 50)
```

Como podemos ver, o dicionário contém tokens individuais associados a rótulos inteiros exclusivos. Nosso próximo objetivo é aplicar esse vocabulário para converter novos textos em IDs de token (figura 7).

![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781633437166/files/Images/2-7.png)

Figura 7 Começando com uma nova amostra de texto, tokenizamos o texto e usamos o vocabulário para converter os tokens de texto em IDs de token. O vocabulário é construído a partir de todo o conjunto de treinamento e pode ser aplicado ao próprio conjunto de treinamento e a quaisquer novas amostras de texto. O vocabulário representado não contém pontuação ou caracteres especiais para simplificar.

Quando queremos converter as saídas de um LLM de números de volta para texto, precisamos de uma maneira de transformar os IDs de token em texto. Para isso, podemos criar uma versão inversa do vocabulário que mapeia os IDs de token de volta para os tokens de texto correspondentes.

Vamos implementar uma classe tokenizadora completa em Python com um método `encode` que divide o texto em tokens e realiza o mapeamento de string para inteiro para produzir IDs de token por meio do vocabulário. Além disso, implementaremos um método `decode` que realiza o mapeamento reverso de inteiro para string para converter os IDs de token novamente em texto. A listagem a seguir mostra o código para esta implementação do tokenizador.

Listagem 3 Implementando um tokenizador de texto simples

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

\#1 Armazena o vocabulário como um atributo de classe para acesso nos métodos de codificação e decodificação\
\#2 Cria um vocabulário inverso que mapeia os IDs de token de volta aos tokens de texto originais\
\#3 Processa o texto de entrada em IDs de token\
\#4 Converte os IDs de token de volta em texto\
\#5 Remove espaços antes da pontuação especificada

Usando a classe `SimpleTokenizerV1` Python, agora podemos instanciar novos objetos tokenizadores por meio de um vocabulário existente, que podemos usar para codificar e decodificar texto, conforme ilustrado na figura 8.

![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781633437166/files/Images/2-8.png)

Figura 8 As implementações do Tokenizer compartilham dois métodos comuns: um método de codificação e um método de decodificação. O método de codificação recebe o texto de amostra, divide-o em tokens individuais e converte os tokens em IDs de token por meio do vocabulário. O método de decodificação recebe os IDs de token, converte-os novamente em tokens de texto e concatena os tokens de texto em texto natural.

Vamos instanciar um novo objeto tokenizador da classe `SimpleTokenizerV1` e tokenizar uma passagem do conto de Edith Wharton para testá-lo na prática:

```Python
tokenizer = SimpleTokenizerV1(vocab)
text = """"It's the last he painted, you know, 
       Mrs. Gisburn said with pardonable pride."""
ids = tokenizer.encode(text)
print(ids)
```

O código anterior imprime os seguintes IDs de token:

```Python
[1, 56, 2, 850, 988, 602, 533, 746, 5, 1126, 596, 5, 1, 67, 7, 
 38, 851, 1108, 754, 793, 7]
```

Em seguida, vamos ver se podemos transformar esses IDs de token novamente em texto usando o método de decodificação:

```Python
print(tokenizer.decode(ids))
```

Isso gera:

```Python
""" It' s the last he painted, you know, Mrs. Gisburn said with 
pardonable pride."""
```

Com base nessa saída, podemos ver que o método de decodificação converteu com sucesso os IDs do token de volta ao texto original.

Até aqui, tudo bem. Implementamos um tokenizador capaz de tokenizar e destokenizar texto com base em um trecho do conjunto de treinamento. Vamos agora aplicá-lo a uma nova amostra de texto não contida no conjunto de treinamento:

```Python
text = "Hello, do you like tea?"
print(tokenizer.encode(text))
```

A execução deste código resultará no seguinte erro:

```Python
KeyError: 'Hello'
```

O problema é que a palavra "Hello" não foi usada no conto "The Verdict". Portanto, ela não está presente no vocabulário. Isso destaca a necessidade de considerar conjuntos de treinamento amplos e diversificados para ampliar o vocabulário ao trabalhar em LLMs.

Em seguida, testaremos o tokenizador mais detalhadamente em textos que contêm palavras desconhecidas e discutiremos tokens especiais adicionais que podem ser usados para fornecer mais contexto para um LLM durante o treinamento.

## Adicionando tokens de contexto especiais

Precisamos modificar o tokenizador para lidar com palavras desconhecidas. Também precisamos abordar o uso e a adição de tokens de contexto especiais que podem aprimorar a compreensão do contexto ou de outras informações relevantes no texto por um modelo. Esses tokens especiais podem incluir marcadores para palavras desconhecidas e limites de documentos, por exemplo. Em particular, modificaremos o vocabulário e o tokenizador, `SimpleTokenizerV2`, para suportar dois novos tokens, `<|unk|>` e `<|endoftext|>`, conforme ilustrado na Figura 9.

![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781633437166/files/Images/2-9.png)

Figura 9 Adicionamos tokens especiais a um vocabulário para lidar com determinados contextos. Por exemplo, adicionamos um token `<|unk|>` para representar palavras novas e desconhecidas que não faziam parte dos dados de treinamento e, portanto, não faziam parte do vocabulário existente. Além disso, adicionamos um token `<|endoftext|>` que podemos usar para separar duas fontes de texto não relacionadas.

Podemos modificar o tokenizador para usar um token `<|unk|>`  caso encontre uma palavra que não faça parte do vocabulário. Além disso, adicionamos um token entre textos não relacionados. Por exemplo, ao treinar LLMs do tipo GPT em vários documentos ou livros independentes, é comum inserir um token antes de cada documento ou livro que segue uma fonte textual anterior, como ilustrado na Figura 10. Isso ajuda o LLM a entender que, embora essas fontes textuais estejam concatenadas para treinamento, elas, na verdade, não são relacionadas.

![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781633437166/files/Images/2-10.png)

Figura 10 Ao trabalhar com múltiplas fontes textuais independentes, adicionamos `<|endoftext|>` tokens entre esses textos. Esses `<|endoftext|>` tokens funcionam como marcadores, sinalizando o início ou o fim de um segmento específico, permitindo um processamento e uma compreensão mais eficazes pelo LLM.

Vamos agora modificar o vocabulário para incluir esses dois tokens especiais, `<unk>` e `<|endoftext|>`, adicionando-os à nossa lista de todas as palavras únicas:

```Python
all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])
vocab = {token:integer for integer,token in enumerate(all_tokens)}
print(len(vocab.items()))
```

Com base na saída desta instrução de impressão, o novo tamanho do vocabulário é 1.132 (o tamanho do vocabulário anterior era 1.130).

Como uma verificação rápida adicional, vamos imprimir as últimas cinco entradas do vocabulário atualizado:

```Python
for i, item in enumerate(list(vocab.items())[-5:]):
    print(item)
```

O código imprime

```Python
('younger', 1127)
('your', 1128)
('yourself', 1129)
('<|endoftext|>', 1130)
('<|unk|>', 1131)
```

Com base na saída do código, podemos confirmar que os dois novos tokens especiais foram de fato incorporados com sucesso ao vocabulário. Em seguida, ajustamos o tokenizador da Listagem de Código 3 conforme mostrado na listagem a seguir.

Listagem 4 Um tokenizador de texto simples que manipula palavras desconhecidas

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

\#1 Substitui palavras desconhecidas por tokens <|unk|>\
\#2 Substitui espaços antes das pontuações especificadas

Comparado ao `SimpleTokenizerV1` que implementamos na listagem 3, o novo `SimpleTokenizerV2` substitui palavras desconhecidas por `<|unk|>` tokens.

Vamos agora testar este novo tokenizador na prática. Para isso, usaremos um exemplo de texto simples que concatenamos a partir de duas frases independentes e não relacionadas:

```Python
text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."
text = " <|endoftext|> ".join((text1, text2))
print(text)
```

A saída é

```Python
Hello, do you like tea? <|endoftext|> In the sunlit terraces of 
the palace.
```

Em seguida, vamos tokenizar o texto de exemplo usando o vocabulário `SimpleTokenizerV2` que criamos anteriormente na listagem 2:

```Python
tokenizer = SimpleTokenizerV2(vocab)
print(tokenizer.encode(text))
```

Isso imprime os seguintes IDs de token:

```Python
[1131, 5, 355, 1126, 628, 975, 10, 1130, 55, 988, 956, 984, 722, 
 988, 1131, 7]
```

Podemos ver que a lista de IDs de token contém `1130` o `<|endoftext|>` token separador, bem como dois `1131` tokens, que são usados para palavras desconhecidas.

Vamos destokenizar o texto para uma rápida verificação de integridade:

```Python
print(tokenizer.decode(tokenizer.encode(text)))
```

A saída é

```Python
<|unk|>, do you like tea? <|endoftext|> In the sunlit terraces of 
the <|unk|>.
```

Com base na comparação deste texto destokenizado com o texto de entrada original, sabemos que o conjunto de dados de treinamento, o conto de Edith Wharton “O Veredito”, não contém as palavras “Hello” e “palace”.

Dependendo do LLM, alguns pesquisadores também consideram tokens especiais adicionais, como os seguintes:

* `[BOS]` _(início da sequência)&#x2009;_&#xA0;— Este token marca o início de um texto. Indica para o LLM onde um trecho de conteúdo começa.

* `[EOS]` _(fim da sequência)&#x2009;_&#xA0;— Este token é posicionado no final de um texto e é especialmente útil ao concatenar vários textos não relacionados, semelhante a `<|endoftext|>`. Por exemplo, ao combinar dois artigos ou livros diferentes da Wikipédia, o `[EOS]` token indica onde um termina e o próximo começa.

* `[PAD]` _(preenchimento)&#x2009;_&#xA0;— Ao treinar LLMs com tamanhos de lote maiores que um, o lote pode conter textos de tamanhos variados. Para garantir que todos os textos tenham o mesmo comprimento, os textos mais curtos são estendidos ou "preenchidos" usando o `[PAD]` token, até o comprimento do texto mais longo do lote.

O tokenizador usado para modelos GPT não precisa de nenhum desses tokens; ele usa apenas um `<|endoftext|>` token para simplificar. `<|endoftext|>` é análogo ao `[EOS]` token. `<|endoftext|>` também é usado para preenchimento. No entanto, como exploraremos nos capítulos subsequentes, ao treinar entradas em lote, normalmente usamos uma máscara, o que significa que não consideramos tokens preenchidos. Portanto, o token específico escolhido para preenchimento torna-se irrelevante.

Além disso, o tokenizador usado nos modelos GPT também não utiliza um `<|unk|>` token para palavras fora do vocabulário. Em vez disso, os modelos GPT usam um tokenizador _de codificação de pares de bytes_ , que divide as palavras em unidades de subpalavra, que discutiremos a seguir.

## Codificação de pares de 2,5 bytes

Vamos analisar um esquema de tokenização mais sofisticado baseado em um conceito chamado codificação de pares de bytes (BPE). O tokenizador BPE foi usado para treinar LLMs como GPT-2, GPT-3 e o modelo original usado no ChatGPT.

Como implementar o BPE pode ser relativamente complexo, usaremos uma biblioteca Python de código aberto existente chamada _tiktoken_ ( <https://github.com/openai/tiktoken> ), que implementa o algoritmo BPE de forma muito eficiente com base no código-fonte em Rust. Semelhante a outras bibliotecas Python, podemos instalar a biblioteca tiktoken por meio do instalador Python `pip` a partir do terminal:

```Python
pip install tiktoken
```

O código que usaremos é baseado no TikTok 0.7.0. Você pode usar o seguinte código para verificar a versão instalada atualmente:

```Python
from importlib.metadata import version
import tiktoken
print("tiktoken version:", version("tiktoken"))
```

Uma vez instalado, podemos instanciar o tokenizador BPE do tiktoken da seguinte maneira:

```Python
tokenizer = tiktoken.get_encoding("gpt2")
```

O uso deste tokenizador é semelhante ao que `SimpleTokenizerV2` que implementamos anteriormente por meio de um método `encode`:

```Python
text = (
    "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
     "of someunknownPlace."
)
integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print(integers)
```

O código imprime os seguintes IDs de token:

```Python
[15496, 11, 466, 345, 588, 8887, 30, 220, 50256, 
 554, 262, 4252, 18250,
 8812, 2114, 286, 617, 34680, 27271, 13]
```

Podemos então converter os IDs de token novamente em texto usando o método de decodificação, semelhante ao nosso `SimpleTokenizerV2`:

```Python
strings = tokenizer.decode(integers)
print(strings)
```

O código imprime

```Python
Hello, do you like tea? <|endoftext|> In the sunlit terraces of
 someunknownPlace.
```

Podemos fazer duas observações dignas de nota com base nos IDs dos tokens e no texto decodificado. Primeiro, ao `<|endoftext|>` token é atribuído um ID de token relativamente grande, a saber, `50256`. De fato, o tokenizador BPE, usado para treinar modelos como GPT-2, GPT-3 e o modelo original usado no ChatGPT, tem um vocabulário total de 50.257, sendo que para o  `<|endoftext|>`  é atribuído o maior ID de token.

Em segundo lugar, o tokenizador BPE codifica e decodifica corretamente palavras desconhecidas, como `someunknownPlace`. O tokenizador BPE pode lidar com qualquer palavra desconhecida. Como ele consegue isso sem usar `<|unk|>` tokens?

O algoritmo subjacente ao BPE divide palavras que não estão em seu vocabulário predefinido em unidades menores de subpalavra ou mesmo em caracteres individuais, permitindo o tratamento de palavras fora do vocabulário. Assim, graças ao algoritmo BPE, se o tokenizador encontrar uma palavra desconhecida durante a tokenização, ele pode representá-la como uma sequência de tokens ou caracteres de subpalavra, conforme ilustrado na figura 11.

![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781633437166/files/Images/2-11.png)

Figura 11 Os tokenizadores BPE dividem palavras desconhecidas em subpalavras e caracteres individuais. Dessa forma, um tokenizador BPE pode analisar qualquer palavra e não precisa substituir palavras desconhecidas por tokens especiais, como `<|unk|>`.

A capacidade de dividir palavras desconhecidas em caracteres individuais garante que o tokenizador e, consequentemente, o LLM treinado com ele possam processar qualquer texto, mesmo que contenha palavras que não estavam presentes em seus dados de treinamento.

Exercício 1 Codificação de pares de bytes de palavras desconhecidas

Experimente o tokenizador BPE da biblioteca tiktoken nas palavras desconhecidas "Akwirw ier" e imprima os IDs individuais dos tokens. Em seguida, chame a função `decode` em cada um dos inteiros resultantes nesta lista para reproduzir o mapeamento mostrado na figura 11. Por fim, chame o método decode nos IDs dos tokens para verificar se ele consegue reconstruir a entrada original, "Akwirw ier".

Uma discussão detalhada e implementação do BPE estão fora do escopo deste livro, mas, em resumo, ele constrói seu vocabulário mesclando iterativamente caracteres frequentes em subpalavras e subpalavras frequentes em palavras. Por exemplo, o BPE começa adicionando todos os caracteres individuais ao seu vocabulário ("a", "b" etc.). Na etapa seguinte, ele mescla combinações de caracteres que ocorrem frequentemente juntas em subpalavras. Por exemplo, "d" e "e" podem ser mesclados na subpalavra "de", que é comum em muitas palavras em inglês como "define", "depend", "made" e "hidden". As mesclagens são determinadas por um limite de frequência.

## Amostragem de dados com janela deslizante

O próximo passo na criação dos embeddings para o LLM é gerar os pares entrada-alvo necessários para o treinamento de um LLM. Como são esses pares entrada-alvo? Como já aprendemos, os LLMs são pré-treinados pela previsão da próxima palavra em um texto, conforme ilustrado na Figura 12.

![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781633437166/files/Images/2-12.png)

Figura 12 Dado um exemplo de texto, extraia blocos de entrada como subamostras que servem como entrada para o LLM, e a tarefa de previsão do LLM durante o treinamento é prever a próxima palavra que segue o bloco de entrada. Durante o treinamento, mascaramos todas as palavras que estão além do alvo. Observe que o texto mostrado nesta figura deve passar por tokenização antes que o LLM possa processá-lo; no entanto, esta figura omite a etapa de tokenização para maior clareza.

Vamos implementar um carregador de dados que busca os pares de entrada-alvo da Figura 12 no conjunto de dados de treinamento usando uma abordagem de janela deslizante. Para começar, tokenizaremos toda a história resumida "O Veredito" usando o tokenizador BPE:

```Python
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

enc_text = tokenizer.encode(raw_text)
print(len(enc_text))
```

A execução deste código retornará o número `5145` total de tokens no conjunto de treinamento, após a aplicação do tokenizador BPE.

Em seguida, removemos os primeiros 50 tokens do conjunto de dados para fins de demonstração, pois isso resulta em uma passagem de texto um pouco mais interessante nas próximas etapas:
```Python
enc_sample = enc_text[50:]
```

Uma das maneiras mais fáceis e intuitivas de criar os pares de entrada-alvo para a tarefa de previsão da próxima palavra é criar duas variáveis, `x` e `y`, onde `x` contém os tokens de entrada e `y` contém os alvos, que são as entradas deslocadas em 1:

```Python
context_size = 4 #1
x = enc_sample[:context_size]
y = enc_sample[1:context_size+1]
print(f"x: {x}")
print(f"y: {y}")
```

\#1 O tamanho do contexto determina quantos tokens são incluídos na entrada.

A execução do código anterior imprime a seguinte saída:

```Python
x: [290, 4920, 2241, 287]
y: [4920, 2241, 287, 257]
```

Ao processar as entradas junto com os alvos, que são as entradas deslocadas em uma posição, podemos criar as tarefas de previsão da próxima palavra (veja a figura 12), da seguinte forma:

```Python
for i in range(1, context_size+1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    print(context, "---->", desired)
```

O código imprime

```Python
[290] ----> 4920
[290, 4920] ----> 2241
[290, 4920, 2241] ----> 287
[290, 4920, 2241, 287] ----> 257
```

Tudo à esquerda da seta (`---->`) refere-se à entrada que um LLM receberia, e o ID do token à direita da seta representa o ID do token de destino que o LLM deve prever. Vamos repetir o código anterior, mas converter os IDs dos tokens em texto:

```Python
for i in range(1, context_size+1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    print(tokenizer.decode(context), "---->", 
          tokenizer.decode([desired]))
```

As saídas a seguir mostram a aparência da entrada e das saídas no formato de texto:

```Python
 and ---->  established
 and established ---->  himself
 and established himself ---->  in
 and established himself in ---->  a
```

Agora criamos os pares de entrada-alvo que podemos usar para o treinamento LLM.

Resta apenas mais uma tarefa antes de transformarmos os tokens em embeddings: implementar um carregador de dados eficiente que itere sobre o conjunto de dados de entrada e retorne as entradas e os alvos como tensores PyTorch, que podem ser considerados matrizes multidimensionais. Em particular, estamos interessados ​​em retornar dois tensores: um tensor de entrada contendo o texto que o LLM vê e um tensor de alvo que inclui os alvos para o LLM prever, conforme ilustrado na Figura 13. Embora a figura mostre os tokens em formato de string para fins ilustrativos, a implementação do código operará diretamente nos IDs dos tokens, já que o método `encode` do tokenizador BPE realiza tanto a tokenização quanto a conversão em IDs de token em uma única etapa.

![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781633437166/files/Images/2-13.png)

Figura 13 Para implementar carregadores de dados eficientes, coletamos as entradas em um tensor, `x`, onde cada linha representa um contexto de entrada. Um segundo tensor, `y`, contém os alvos de predição correspondentes (próximas palavras), que são criados deslocando a entrada em uma posição.

Observação: Para uma implementação eficiente do carregador de dados, usaremos as classes e integradas do PyTorch `DataLoader`. Para obter mais informações e orientações sobre a instalação do PyTorch, consulte a seção A.2.1.3 no apêndice A.

O código para a classe dataset é mostrado na listagem a seguir.

Listagem 5 Um conjunto de dados para entradas e alvos em lote

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

\#1 Tokeniza todo o texto\
\#2 Usa uma janela deslizante para dividir o livro em sequências sobrepostas de max\_length\
\#3 Retorna o número total de linhas no conjunto de dados\
\#4 Retorna uma única linha do conjunto de dados

A classe `GPTDatasetV1`  é baseada na classe `Dataset` do PyTorch e define como linhas individuais são recuperadas do conjunto de dados, onde cada linha consiste em um número de IDs de token (com base em um `max_length`) atribuídos a um tensor `input_chunk`. O tensor `target_ chunk` contém os alvos correspondentes. Recomendo continuar lendo para ver como ficam os dados retornados deste conjunto de dados quando combinamos o conjunto de dados com um PyTorch `DataLoader` — isso trará mais intuição e clareza.

Nota: Se você é novo na estrutura das classes `Dataset` do PyTorch, como mostrado na Listagem 5, consulte a seção A.6 no apêndice A, que explica a estrutura geral e o uso do PyTorch `Dataset` e `DataLoader` das classes.

O código a seguir usa o `GPTDatasetV1` para carregar as entradas em lotes por meio de um PyTorch `DataLoader`.

Listagem 6 Um carregador de dados para gerar lotes com pares input-with

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

\#1 Inicializa o tokenizador\
\#2 Cria o conjunto de dados\
\#3 drop\_last=True descarta o último lote se ele for menor que o batch\_size especificado para evitar picos de perda durante o treinamento.\
\#4 O número de processos de CPU a serem usados para pré-processamento

Vamos testar O `dataloader` com um tamanho de lote de 1 para um LLM com um tamanho de contexto de 4 para desenvolver uma intuição de como a classe `GPTDatasetV1` da listagem 5 e a função `create_ dataloader_v1` da listagem 6 funcionam juntas:

```Python
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

dataloader = create_dataloader_v1(
    raw_text, batch_size=1, max_length=4, stride=1, shuffle=False)
data_iter = iter(dataloader) #1
first_batch = next(data_iter)
print(first_batch)
```

\#1 Converte o dataloader em um iterador Python para buscar a próxima entrada por meio da função next() interna do Python

A execução do código anterior imprime o seguinte:

```Python
[tensor([[  40,  367, 2885, 1464]]), 
 tensor([[ 367, 2885, 1464, 1807]])]
```

A variável `first_batch` contém dois tensores: o primeiro tensor armazena os IDs dos tokens de entrada e o segundo tensor armazena os IDs dos tokens de destino. Como o valor de `max_length` é 4, cada um dos dois tensores contém quatro IDs de token. Observe que um tamanho de entrada de 4 é bem pequeno e foi escolhido apenas por simplicidade. É comum treinar LLMs com tamanhos de entrada de pelo menos 256.

Para entender o significado de `stride=1`, vamos buscar outro lote deste conjunto de dados:

```Python
second_batch = next(data_iter)
print(second_batch)
```

O segundo lote tem o seguinte conteúdo:

```Python
[tensor([[ 367, 2885, 1464, 1807]]), 
 tensor([[2885, 1464, 1807, 3619]])]
```

Se compararmos o primeiro e o segundo lotes, podemos ver que os IDs de token do segundo lote são deslocados em uma posição (por exemplo, o segundo ID na entrada do primeiro lote é 367, que é o primeiro ID da entrada do segundo lote). A configuração `stride` determina o número de posições que as entradas deslocam entre os lotes, emulando uma abordagem de janela deslizante, como demonstrado na figura 14.

![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781633437166/files/Images/2-14.png)

Figura 14 Ao criar vários lotes a partir do conjunto de dados de entrada, deslizamos uma janela de entrada sobre o texto. Se o passo for definido como 1, deslocamos a janela de entrada em uma posição ao criar o próximo lote. Se definirmos o passo igual ao tamanho da janela de entrada, podemos evitar sobreposições entre os lotes.

Exercício 2 Carregadores de dados com diferentes passos e tamanhos de contexto

Para desenvolver mais intuição sobre como o carregador de dados funciona, tente executá-lo com configurações diferentes, como `max_length=2` e `stride=2,` e `max_length=8` e `stride=2`.

Tamanhos de lote de 1, como os que amostramos do carregador de dados até agora, são úteis para fins ilustrativos. Se você já tem experiência com aprendizado profundo, talvez saiba que tamanhos de lote pequenos exigem menos memória durante o treinamento, mas levam a atualizações de modelo mais ruidosas. Assim como no aprendizado profundo tradicional, o tamanho do lote é uma compensação e um hiperparâmetro a ser experimentado ao treinar LLMs.

Vamos dar uma olhada rápida em como podemos usar o carregador de dados para amostrar com um tamanho de lote maior que 1:

```Python
dataloader = create_dataloader_v1(
    raw_text, batch_size=8, max_length=4, stride=4,
    shuffle=False
)

data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("Inputs:\n", inputs)
print("\nTargets:\n", targets)
```

Isto imprime

```Python
Inputs:
 tensor([[  40,   367,  2885,  1464],
        [ 1807,  3619,   402,   271],
        [10899,  2138,   257,  7026],
        [15632,   438,  2016,   257],
        [  922,  5891,  1576,   438],
        [  568,   340,   373,   645],
        [ 1049,  5975,   284,   502],
        [  284,  3285,   326,    11]])

Targets:
 tensor([[ 367,  2885,  1464,  1807],
        [ 3619,   402,   271, 10899],
        [ 2138,   257,  7026, 15632],
        [  438,  2016,   257,   922],
        [ 5891,  1576,   438,   568],
        [  340,   373,   645,  1049],
        [ 5975,   284,   502,   284],
        [ 3285,   326,    11,   287]])
```

Observe que aumentamos o passo para 4 para utilizar o conjunto de dados completamente (não pulamos uma única palavra). Isso evita qualquer sobreposição entre os lotes, pois mais sobreposição poderia levar a um aumento no overfitting.

## Criação de embeddings de tokens

A última etapa na preparação do texto de entrada para o treinamento LLM é converter os IDs de token em vetores de _embedding_, conforme mostrado na figura 15. Como etapa preliminar, precisamos inicializar esses pesos de _embedding_ com valores aleatórios. Essa inicialização serve como ponto de partida para o processo de aprendizado do LLM. No capítulo 5, otimizaremos os pesos de _embedding_ como parte do treinamento do LLM.

![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781633437166/files/Images/2-15.png)

Figura 15 A preparação envolve a tokenização de texto, a conversão de tokens de texto em IDs de token e a conversão de IDs de token em vetores de _embedding_. Aqui, consideramos os IDs de token criados anteriormente para criar os vetores de _embedding_ de token.

Uma representação vetorial contínua, ou _embedding_, é necessária, pois LLMs semelhantes a GPT são redes neurais profundas treinadas com o algoritmo de retropropagação.

Nota: Se você não estiver familiarizado com a forma como as redes neurais são treinadas com retropropagação, leia a seção A.4 no apêndice A.

Vamos ver como funciona a conversão de ID de token para vetor de _embedding_ com um exemplo prático. Suponha que temos os quatro tokens de entrada a seguir, com IDs 2, 3, 5 e 1:

```Python
input_ids = torch.tensor([2, 3, 5, 1])
```

Para simplificar, suponha que temos um pequeno vocabulário de apenas 6 palavras (em vez das 50.257 palavras no vocabulário do tokenizador BPE) e queremos criar embeddings de tamanho 3 (no GPT-3, o tamanho do embedding é 12.288 dimensões):

```Python
vocab_size = 6
output_dim = 3
```

Usando `vocab_size` e `output_dim`, podemos instanciar uma camada de _embedding_ no PyTorch, definindo a semente aleatória para `123` para fins de reprodutibilidade:

```Python
torch.manual_seed(123)
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
print(embedding_layer.weight)
```

A instrução print imprime a matriz de peso subjacente da camada de _embedding_:

```Python
Parameter containing:
tensor([[ 0.3374, -0.1778, -0.1690],
        [ 0.9178,  1.5810,  1.3010],
        [ 1.2753, -0.2010, -0.1606],
        [-0.4015,  0.9666, -1.1481],
        [-1.1589,  0.3255, -0.6315],
        [-2.8400, -0.7849, -1.4096]], requires_grad=True)
```

A matriz de pesos da camada de _embedding_ contém pequenos valores aleatórios. Esses valores são otimizados durante o treinamento do LLM como parte da própria otimização do LLM. Além disso, podemos ver que a matriz de pesos possui seis linhas e três colunas. Há uma linha para cada um dos seis tokens possíveis no vocabulário e uma coluna para cada uma das três dimensões de _embedding_.

Agora, vamos aplicá-lo a um ID de token para obter o vetor de _embedding_:

```Python
tensor([[-0.4015,  0.9666, -1.1481]], grad_fn=<EmbeddingBackward0>)
```

O vetor de _embedding_ retornado é

```Python
tensor([[-0,4015, 0,9666, -1,1481]], grad_fn=<EmbeddingBackward0>)
```

Se compararmos o vetor de _embedding_ para o ID do token 3 com a matriz de _embedding_ anterior, vemos que ele é idêntico à quarta linha (o Python começa com um índice zero, portanto, é a linha correspondente ao índice 3). Em outras palavras, a camada de _embedding_ é essencialmente uma operação de consulta que recupera linhas da matriz de pesos da camada de _embedding_ por meio de um ID de token.

Nota: Para aqueles familiarizados com a codificação one-hot, a abordagem da camada de _embedding_ descrita aqui é essencialmente apenas uma maneira mais eficiente de implementar a codificação one-hot seguida pela multiplicação de matrizes em uma camada totalmente conectada, o que é ilustrado no código suplementar no GitHub em <https://mng.bz/ZEB5> . Como a camada de _embedding_ é apenas uma implementação mais eficiente equivalente à codificação one-hot e à abordagem de multiplicação de matrizes, ela pode ser vista como uma camada de rede neural que pode ser otimizada por meio de retropropagação.

Vimos como converter um único ID de token em um vetor de _embedding_ tridimensional. Vamos agora aplicar isso a todos os quatro IDs de entrada ( `torch.tensor([2,` `3,` `5,` `1])`):

```Python
print(embedding_layer(input_ids))
```

A saída de impressão revela que isso resulta em uma matriz 4 × 3:

```Python
tensor([[ 1.2753, -0.2010, -0.1606],
        [-0.4015,  0.9666, -1.1481],
        [-2.8400, -0.7849, -1.4096],
        [ 0.9178,  1.5810,  1.3010]], grad_fn=<EmbeddingBackward0>)
```

Cada linha nesta matriz de saída é obtida por meio de uma operação de pesquisa da matriz de peso de _embedding_, conforme ilustrado na figura 16.

![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781633437166/files/Images/2-16.png)

Figura 16 As camadas de _embedding_ realizam uma operação de consulta, recuperando o vetor de _embedding_ correspondente ao ID do token da matriz de pesos da camada de _embedding_. Por exemplo, o vetor de _embedding_ do ID do token 5 é a sexta linha da matriz de pesos da camada de _embedding_ (é a sexta linha em vez da quinta, porque o Python começa a contagem em 0). Assumimos que os IDs dos tokens foram produzidos pelo pequeno vocabulário da seção 3.

Agora que criamos vetores de _embedding_ a partir de IDs de tokens, adicionaremos uma pequena modificação a esses vetores de _embedding_ para codificar informações posicionais sobre um token dentro de um texto.

## Codificação de posições de palavras

Em princípio, a _embedding_ de tokens é uma entrada adequada para um LLM. No entanto, uma pequena deficiência dos LLMs é que seu mecanismo de autoatenção (ver capítulo 3) não possui uma noção de posição ou ordem para os tokens dentro de uma sequência. A camada de _embedding_ introduzida anteriormente funciona de forma que o mesmo ID de token sempre é mapeado para a mesma representação vetorial, independentemente de onde o ID de token esteja posicionado na sequência de entrada, como mostrado na figura 17.

![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781633437166/files/Images/2-17.png)

Figura 17 A camada de _embedding_ converte um ID de token na mesma representação vetorial, independentemente de onde esteja localizado na sequência de entrada. Por exemplo, o ID de token 5, esteja ele na primeira ou quarta posição no vetor de entrada de ID de token, resultará no mesmo vetor de _embedding_.

Em princípio, a _embedding_ determinística e independente da posição do ID do token é boa para fins de reprodutibilidade. No entanto, como o mecanismo de autoatenção dos LLMs também é independente da posição, é útil injetar informações adicionais sobre a posição no LLM.

Para isso, podemos usar duas categorias amplas de embeddings com reconhecimento de posição: embeddings posicionais relativos e embeddings posicionais absolutos. Os embeddings posicionais absolutos estão diretamente associados a posições específicas em uma sequência. Para cada posição na sequência de entrada, um embedding exclusivo é adicionado ao embedding do token para indicar sua localização exata. Por exemplo, o primeiro token terá um embedding posicional específico, o segundo token terá outro embedding distinto e assim por diante, conforme ilustrado na figura 18.

![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781633437166/files/Images/2-18.png)

Figura 18 Embeddings posicionais são adicionados ao vetor de embedding de tokens para criar os embeddings de entrada para um LLM. Os vetores posicionais têm a mesma dimensão dos embeddings de tokens originais. Os embeddings de tokens são mostrados com o valor 1 para simplificar.

Em vez de focar na posição absoluta de um token, a ênfase dos embeddings posicionais relativos está na posição relativa ou distância entre os tokens. Isso significa que o modelo aprende as relações em termos de "quão distantes" em vez de "em qual posição exata". A vantagem aqui é que o modelo pode generalizar melhor para sequências de comprimentos variados, mesmo que não tenha observado tais comprimentos durante o treinamento.

Ambos os tipos de embeddings posicionais visam aumentar a capacidade dos LLMs de compreender a ordem e as relações entre os tokens, garantindo previsões mais precisas e sensíveis ao contexto. A escolha entre eles geralmente depende da aplicação específica e da natureza dos dados processados.

Os modelos GPT da OpenAI utilizam embeddings posicionais absolutos que são otimizados durante o processo de treinamento, em vez de serem fixos ou predefinidos como as codificações posicionais no modelo de _transformer_ original. Esse processo de otimização faz parte do treinamento do modelo em si. Por enquanto, vamos criar os embeddings posicionais iniciais para criar as entradas do LLM.

Anteriormente, focamos em tamanhos de _embedding_ muito pequenos para simplificar. Agora, vamos considerar tamanhos de _embedding_ mais realistas e úteis e codificar os tokens de entrada em uma representação vetorial de 256 dimensões, que é menor do que o modelo GPT-3 original usado (no GPT-3, o tamanho de _embedding_ é de 12.288 dimensões), mas ainda razoável para experimentação. Além disso, assumimos que os IDs de token foram criados pelo tokenizador BPE que implementamos anteriormente, que tem um tamanho de vocabulário de 50.257:

```Python
vocab_size = 50257
output_dim = 256
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
```

Usando o método anterior `token_embedding_layer`, se amostrarmos dados do carregador de dados, criaremos _embeddings_ com cada token de cada lote em um vetor de 256 dimensões. Se tivermos um lote de 8 com quatro tokens cada, o resultado será um tensor de 8 × 4 × 256.

Vamos instanciar o carregador de dados (veja seção 6) primeiro:

```Python
max_length = 4
dataloader = create_dataloader_v1(
    raw_text, batch_size=8, max_length=max_length,
   stride=max_length, shuffle=False
)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("Token IDs:\n", inputs)
print("\nInputs shape:\n", inputs.shape)
```

Este código imprime

```Python
Token IDs:
 tensor([[  40,   367,  2885,  1464],
        [ 1807,  3619,   402,   271],
        [10899,  2138,   257,  7026],
        [15632,   438,  2016,   257],
        [  922,  5891,  1576,   438],
        [  568,   340,   373,   645],
        [ 1049,  5975,   284,   502],
        [  284,  3285,   326,    11]])

Inputs shape:
 torch.Size([8, 4])
```

Como podemos ver, o tensor de ID do token tem dimensão 8 × 4, o que significa que o lote de dados consiste em oito amostras de texto com quatro tokens cada.

Vamos agora usar a camada de _embedding_ para incorporar esses IDs de token em vetores de 256 dimensões:

```Python
token_embeddings = token_embedding_layer(inputs)
print(token_embeddings.shape)
```

A chamada da função print retorna

```Python
torch.Size([8, 4, 256])
```

A saída do tensor dimensional 8 × 4 × 256 mostra que cada ID de token agora está incorporado como um vetor dimensional 256.

Para uma abordagem de _embedding_ absoluta do modelo GPT, precisamos apenas criar outra camada de _embedding_ que tenha a mesma dimensão de _embedding_ que `token_embedding_ layer`:

```Python
context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(context_length))
print(pos_embeddings.shape)
```

A entrada para o `pos_embeddings` é geralmente um vetor de espaço reservado `torch.arange(context_length)`, que contém uma sequência de números 0, 1, ..., até o comprimento máximo de entrada –1. A variável `context_length` representa o tamanho de entrada suportado pelo LLM. Aqui, escolhemos um tamanho semelhante ao comprimento máximo do texto de entrada. Na prática, o texto de entrada pode ser maior que o comprimento de contexto suportado, caso em que precisamos truncar o texto.

A saída da instrução print é

```Python
torch.Size([4, 256])
```

Como podemos ver, o tensor de _embedding_ posicional consiste em quatro vetores de 256 dimensões. Agora podemos adicioná-los diretamente aos embeddings de token, onde o PyTorch adicionará o tensor `pos_embeddings ` de 4 × 256 dimensões a cada tensor de _embedding_ de token de 4 × 256 dimensões em cada um dos oito lotes:

```Python
input_embeddings = token_embeddings + pos_embeddings
print(input_embeddings.shape)
```

A saída de impressão é

```Python
torch.Size([8, 4, 256])
```

O `input_embeddings` que criamos, conforme resumido na figura 19, são os exemplos de entrada incorporados que agora podem ser processados pelos principais módulos LLM, que começaremos a implementar no próximo capítulo.

![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781633437166/files/Images/2-19.png)

Figura 19 Como parte do pipeline de processamento de entrada, o texto de entrada é primeiro dividido em tokens individuais. Esses tokens são então convertidos em IDs de token usando um vocabulário. Os IDs de token são convertidos em vetores de _embedding_ aos quais são adicionados embeddings posicionais de tamanho semelhante, resultando em embeddings de entrada que são usados ​​como entrada para as principais camadas do LLM.

## Resumo

* LLMs exigem que dados textuais sejam convertidos em vetores numéricos, conhecidos como embeddings, uma vez que não podem processar texto bruto. Os embeddings transformam dados discretos (como palavras ou imagens) em espaços vetoriais contínuos, tornando-os compatíveis com operações de redes neurais.

* Na primeira etapa, o texto bruto é dividido em tokens, que podem ser palavras ou caracteres. Em seguida, os tokens são convertidos em representações inteiras, denominadas IDs de token.

* Tokens especiais, como `<|unk|>` e `<|endoftext|>`, podem ser adicionados para melhorar a compreensão do modelo e lidar com vários contextos, como palavras desconhecidas ou marcar o limite entre textos não relacionados.

* O tokenizador de codificação de pares de bytes (BPE) usado para LLMs como GPT-2 e GPT-3 pode lidar eficientemente com palavras desconhecidas, dividindo-as em unidades de subpalavras ou caracteres individuais.

* Usamos uma abordagem de janela deslizante em dados tokenizados para gerar pares de entrada-alvo para treinamento LLM.

* A _embedding_ de camadas no PyTorch funciona como uma operação de consulta, recuperando vetores correspondentes aos IDs dos tokens. Os vetores de _embedding_ resultantes fornecem representações contínuas de tokens, o que é crucial para o treinamento de modelos de aprendizado profundo, como LLMs.

* Embora os embeddings de tokens forneçam representações vetoriais consistentes para cada token, eles não permitem uma noção da posição do token em uma sequência. Para corrigir isso, existem dois tipos principais de embeddings posicionais: absolutos e relativos. Os modelos GPT da OpenAI utilizam embeddings posicionais absolutos, que são adicionados aos vetores de embedding de tokens e otimizados durante o treinamento do modelo.

## Exercícios

1. **Por que os Modelos de Linguagem Grande (LLMs) não conseguem processar texto bruto diretamente?**

	A) Porque o texto bruto é muito grande para a memória dos LLMs.
	B) Porque o texto é categórico e incompatível com as operações matemáticas usadas nas redes neurais.
	C) Porque os LLMs só podem processar imagens e áudio.
	D) Porque o texto bruto contém muitos erros gramaticais e de ortografia.

2. **Qual é o primeiro passo na preparação de texto para o treinamento de um LLM, após a leitura do texto bruto?**

	A) Converter o texto em vetores de _embedding_ imediatamente.
    B) Dividir o texto em tokens individuais de palavras e subpalavras ou caracteres especiais.
    C) Remover todas as palavras desconhecidas do texto.
    D) Traduzir o texto para um idioma diferente.

3. **O que é uma _embedding_ (incorporação) no contexto de LLMs?**

	A) Um tipo de algoritmo de compressão de texto para reduzir o tamanho dos dados.
	B) Um mapeamento de objetos discretos (como palavras) para pontos em um espaço vetorial contínuo.
	C) Um método para adicionar contexto visual aos dados de texto.
	D) Uma técnica para criptografar texto e garantir a privacidade dos dados.

4. **Qual é a ideia principal por trás da abordagem Word2Vec para gerar embeddings de palavras?**

	A) A arquitetura do Word2Vec prevê a próxima palavra em uma sequência, semelhante aos LLMs.
    B) Ele analisa a frequência de cada palavra em um documento para determinar sua importância.
    C) Palavras que aparecem em contextos semelhantes tendem a ter significados semelhantes.
    D) Ele usa regras gramaticais para inferir o significado das palavras.

5. **Qual é a principal vantagem de otimizar os embeddings como parte do treinamento de um LLM, em vez de usar modelos pré-treinados como o Word2Vec?** 

	A) Os embeddings gerados pelo LLM são sempre menores em dimensionalidade, economizando memória. 
	B) Os LLMs podem processar texto bruto diretamente sem a necessidade de embeddings. 
	C) Os embeddings são otimizados para a tarefa e os dados específicos em questão. 
	D) O Word2Vec é um algoritmo desatualizado e não é mais usado na prática.

6. **Qual é o propósito principal do token especial&#x20;**<∣unk∣>**&#x20;(desconhecido) em um tokenizador?** 
	
	A) Marcar o fim de uma sequência de texto. 
	B) Representar palavras novas e desconhecidas que não faziam parte dos dados de treinamento. 
	C) Indicar o início de um novo documento. 
	D) Ignorar palavras comuns que não contribuem para o significado.

7. **Ao treinar LLMs com múltiplas fontes textuais independentes, qual é a função do token&#x20;**<∣endoftext∣>**?** 
	
	A) Indicar que o modelo deve parar de gerar texto. 
	B) Melhorar a legibilidade do texto concatenado para o usuário. 
	C) Funcionar como um marcador, sinalizando o início ou o fim de um segmento específico de texto não relacionado. 
	D) Substituir todas as palavras que não são reconhecidas pelo vocabulário.

16. **Como a Codificação de Pares de Bytes (BPE) lida com palavras desconhecidas (fora do vocabulário)?** 
	A) Substitui a palavra desconhecida por um token genérico <∣unk∣>. 
	B) Ignora a palavra desconhecida e remove-a do texto de entrada. 
	C) Divide a palavra desconhecida em unidades menores de subpalavra ou mesmo em caracteres individuais. 
	D) Solicita ao usuário uma definição para a palavra desconhecida.

17. **Na tarefa de previsão da próxima palavra, como os LLMs são pré-treinados, de acordo com o texto?** 
	
	A) Pela tradução de frases de um idioma para outro. 
	B) Pela classificação de documentos em categorias predefinidas. 
	C) Pela previsão da próxima palavra em um texto, usando uma abordagem de janela deslizante para gerar pares entrada-alvo. 
	D) Pela sumarização de longos artigos em resumos concisos.

18. **Por que os embeddings posicionais são adicionados aos vetores de embedding de tokens na entrada de um LLM?** 
	
	A) Para reduzir a dimensionalidade dos embeddings de tokens e economizar memória. 
	B) Para criptografar os dados de entrada e garantir a segurança. 
	C) Para injetar informações sobre a posição ou ordem dos tokens na sequência, pois o mecanismo de autoatenção do LLM não tem essa noção inerente. 
	D) Para traduzir os tokens para um idioma diferente antes do processamento.

## Respostas: 

1. **B)** Porque o texto é categórico e incompatível com as operações matemáticas usadas nas redes neurais. 2. **B)** Dividir o texto em tokens individuais de palavras e subpalavras ou caracteres especiais. 3. **B)** Um mapeamento de objetos discretos (como palavras) para pontos em um espaço vetorial contínuo. 4. **C)** Palavras que aparecem em contextos semelhantes tendem a ter significados semelhantes. 5. **C)** Os embeddings são otimizados para a tarefa e os dados específicos em questão. 6. **B)** Representar palavras novas e desconhecidas que não faziam parte dos dados de treinamento. 7. **C)** Funcionar como um marcador, sinalizando o início ou o fim de um segmento específico de texto não relacionado. 8. **C)** Divide a palavra desconhecida em unidades menores de subpalavra ou mesmo em caracteres individuais. 9. **C)** Pela previsão da próxima palavra em um texto, usando uma abordagem de janela deslizante para gerar pares entrada-alvo. 10. **C)** Para injetar informações sobre a posição ou ordem dos tokens na sequência, pois o mecanismo de autoatenção do LLM não tem essa noção inerente.
