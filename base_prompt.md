# Contexto

O arquivo `PA3.pdf` define o enunciado de um assignment de programação para a disciplina de Deep Learning.

O repositório atualmente está com a seguinte estrutura de pastas/arquivos:

```bash
.
├── PA3.pdf
├── README.md
├── base_prompt.md
├── prompt.md
├── pyproject.toml
├── readme.md
├── requirements.txt
├── runs
│   ├── resnet18_b5_s0
│   │   ├── best.pt
│   │   ├── config.json
│   │   ├── curves.png
│   │   ├── history.csv
│   │   ├── test_confusion.csv
│   │   ├── test_metrics.json
│   │   └── test_per_class.csv
│   ├── resnet18_b5_s1
│   ├── resnet18_b5_s2
│   ├── sanity_unet
├── scripts
│   ├── smoke_test_model.py
│   └── train_supervised.py
├── src
│   ├── config.py
│   ├── models
│   │   ├── __init__.py
│   │   ├── factory.py
│   │   └── unet_classifier.py
│   └── train
│       ├── __init__.py
│       └── metric.py
└── uv.lock
```

Já tenho a parte 0 concluída, todos os resultados em `/runs`. **ATENÇÃO**: se precisar do conteúdo de quaisquer arquivos para me ajudar com a resposta, me avise.

# Tarefa

Minha escolha para a parte 1 é a trilha C, pelo menos por enquanto, e esse é o planejamento de implementação dos próximos passos:

## Trilha C — Professor e aluno (destilação com pseudo-rótulos)

### Como funciona, na prática

O fluxo tem cinco etapas, e a lógica se apoia numa assimetria entre **conhecimento** e **tamanho**:

1. **Escolher o professor.** Um *foundation model* aberto com alinhamento imagem-texto: **BioCLIP 2** (Imageomics, treinado em TreeOfLife-10M — taxonomia biológica, ideal para fauna), **SigLIP** ou **CLIP**. Todos aceitam "a photo of a {classe}" como prompt textual. Pesos abertos, roda local.

2. **Zero-shot no `test_public`.** Para cada imagem, calcula-se o embedding visual $\mathbf{v}$, o embedding textual $\mathbf{t}_c$ de cada uma das 21 classes, e $p(c \mid x) = \text{softmax}(\mathbf{v} \cdot \mathbf{t}_c / \tau)$. Isso gera o **F1 zero-shot do professor**, que é a linha de referência obrigatória. Nenhum rótulo é usado nesse passo.

3. **Gerar pseudo-rótulos para o `train`.** O professor roda sobre as 6.505 imagens do `train` e salva, para cada uma, **o vetor de probabilidades completo** (não só o argmax). Isso é importante: o vetor completo vai servir tanto para filtro por confiança quanto para *soft labels*.

4. **Tratar o ruído.** O professor erra, e erra de forma sistemática (a matriz de confusão dele sobre o `test_public` **é** a matriz de ruído dos pseudo-rótulos). Estratégias comuns, da mais simples à mais elaborada:
   - **filtro por confiança** (aceita só amostras com $\max_c p > \tau$);
   - **filtro por margem** ($p_{(1)} - p_{(2)} > m$);
   - **limiar por classe** (τ adaptado para equalizar o número de pseudo-rótulos entre as 21 classes — combate o viés de frequência);
   - **rótulos suaves com temperatura** (usa o vetor completo, não o argmax, e aquece com $T>1$ para atenuar erros de alta confiança);
   - **correção de rótulo tipo SELFIE** (estima a matriz de transição ruído→verdade a partir das 105 amostras rotuladas e "desfaz" o ruído dos pseudo-rótulos).

5. **Treinar o aluno.** O aluno é uma ResNet-18 pequena (o mesmo backbone da Parte 0b, como o PA exige). Ele treina com:
   - as **105 amostras com rótulo verdadeiro** (loss CE supervisionada normal);
   - as **6.505 amostras com pseudo-rótulo filtrado/corrigido** (loss CE no rótulo duro, ou KL no rótulo suave).

O objetivo final é um modelo **pequeno**, que roda em CPU, e que — idealmente — **bate ou se aproxima do teto de 0,686**.

### O que esperar (e por que é arriscado)

**Cenário bom** (mais provável com BioCLIP 2): o professor zero-shot já entrega F1 macro entre **0,55 e 0,75** no `test_public`. Nesse caso:

- o **piso da Trilha C já é competitivo com o piso supervisionado** (que costuma ser ~0,25–0,35 com 105 rótulos);
- o aluno destilado pode **chegar perto do teto** e, com boa correção de ruído, **ultrapassá-lo** — porque a destilação transfere o conhecimento do professor (treinado em 10M imagens) e ao mesmo tempo deixa o aluno se especializar nas 21 classes com augmentation supervisionada apropriada.

**Cenário ruim**: o professor zero-shot fica **abaixo de 0,3** (por exemplo, se BioCLIP não reconhecer bem as classes como nomes comuns). Nesse caso a Trilha C vira um problema de ruído em que o aluno aprende os erros do professor. Sinais de alerta:
- o professor confunde "bird" com "american_black_bear" (colapso de embedding);
- erros concentrados em poucas classes dominantes (a distribuição de predições vira uma delta em "deer" ou "raccoon");
- o aluno destilado fica **pior** que o piso supervisionado.

Por isso a checagem de zero-shot no `test_public` é o **primeiro experimento a rodar**. Se o resultado for ruim, é melhor pivotar para a Trilha B (FixMatch) ou A.

### Por que tem o maior potencial de ganho

Três razões independentes:

1. **O professor já viu fauna.** BioCLIP 2 foi treinado em ~10M imagens de espécies com taxonomia — ele carrega uma representação interna dos animais exatamente na granularidade do ENA24 (*Ursus americanus*, *Procyon lotor*, *Odocoileus virginianus*, etc.). O teto de 0,686 é de uma ResNet-18 treinada em ~6.500 imagens rotuladas — é um modelo **especializado em 21 classes**, mas que **não tem conhecimento prévio de fauna**. O professor tem esse conhecimento.

2. **O aluno aprende mais que o professor pode "falar".** O professor tem que resolver um problema de *open vocabulary* em 21 classes. O aluno resolve um problema de classificação fechada em 21 classes, com **augmentation supervisionada apropriada** (o professor é zero-shot, não usa augmentation) e com **os 105 rótulos verdadeiros como âncora** — que corrigem a distribuição do professor. Distilação *teacher→student* é uma receita conhecida de ganhar performance no aluno mesmo quando o professor é forte.

3. **O erro do professor é ruído, não viés irrecuperável.** Ele erra de forma sistemática (bird↔american_crow, animais pequenos, noite), mas essas confusões são **observáveis** na matriz de ruído. Isso permite tratá-las (filtro, temperatura, SELFIE), coisa que um piso supervisionado com 105 rótulos simplesmente não tem como fazer — ele não tem informação suficiente.

O trade-off honesto: a Trilha C tem o **maior teto** e o **maior risco**. Se o professor é bom, ganha disparado; se é ruim, afunda. As Trilhas A e B têm variância menor, mas também teto menor.

### O que a trilha **obriga** a fazer (do enunciado)

Vale ter isso em mente desde o início, porque cada item vira um artefato da apresentação:

| Obrigação | Artefato esperado |
|---|---|
| F1 zero-shot do professor no `test_public` | `test_public` do professor, com os 21 nomes como prompt |
| Matriz de confusão do professor = matriz de ruído; onde erra sistematicamente | `teacher_confusion.csv` + análise qualitativa (noite via brilho, animal pequeno via área estimada, classes parecidas via pares de confusão) |
| Tratar o ruído e responder onde o aluno supera o professor | curvas `professor vs aluno` por classe, e discussão de por quê |

# Tarefa