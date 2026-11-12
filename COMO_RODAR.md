# Como rodar o FitRec-Attn

Guia para treinar o modelo **FitRec-Attn** (previsão de frequência cardíaca de curto prazo) a partir deste fork. As correções necessárias para rodar com Python, PyTorch e NumPy atuais, em CPU ou GPU e no Windows, estão na branch `fitrec-attn-run`.

> O modelo **FitRec** (pasta `FitRec/`) usa Keras 2.0 e não roda em Python moderno sem ser reescrito. Este guia cobre só o **FitRec-Attn**.

## Requisitos

| Item | Mínimo |
|---|---|
| RAM | **8 GB** para treinar com `fitrec_compact.npz`. Gerar esse arquivo (passo 3, uma vez só) precisa de **32 GB**. |
| Disco | 400 MB com o arquivo compacto, ou uns 7 GB para gerá-lo a partir do `.npy`. |
| Python | 3.11, via Anaconda/Miniconda. |
| GPU | Opcional. Uma GPU NVIDIA acelera muito o treino. |

## 1. Baixar o código

```bash
git clone -b fitrec-attn-run https://github.com/laracmv/fit-rec.git
cd fit-rec/FitRec-Attn
mkdir data
```

## 2. Colocar os dados em `FitRec-Attn/data/`

Os dados não estão no git, por causa do tamanho e da licença.

**Caminho curto:** se você já tem o `fitrec_compact.npz` (374 MB, gerado no PC original em `C:\Users\princ\fitrec-data\`), copie só ele para `data/`, instale o ambiente (passo 3, sem rodar os scripts de dados) e vá para o passo 4. Os arquivos abaixo só são necessários para gerar o arquivo compacto.

| Arquivo | Tamanho | De onde vem |
|---|---|---|
| `processed_endomondoHR_proper_interpolate.npy` | 6 GB | Copie do PC original (`C:\Users\princ\fitrec-data\`) ou baixe do [espelho da UCSD](https://mcauleylab.ucsd.edu/public_datasets/gdrive/fitrec/processed_endomondoHR_proper_interpolate.npy) |
| `endomondoHR_proper_temporal_dataset.pkl` | 67 MB | Copie do PC original. Se não tiver, gere com o passo 3. |
| `endomondoHR_proper_metaData.pkl` | 2 MB | Opcional. Se faltar, o script recria sozinho. |

> Os links do Google Drive no `README.md` estão quebrados (erro 404). Use o espelho da UCSD.
> A licença dos dados permite só uso acadêmico. Não redistribua.

Se for baixar pelo terminal, use um comando que retome de onde parou, porque a conexão costuma cair:

```bash
curl -L -C - --retry 5 -o data/processed_endomondoHR_proper_interpolate.npy \
  https://mcauleylab.ucsd.edu/public_datasets/gdrive/fitrec/processed_endomondoHR_proper_interpolate.npy
```

O arquivo completo tem exatamente **6.457.513.404 bytes**.

## 3. Criar o ambiente

```bash
conda create -y -n fitrec python=3.11
conda activate fitrec
pip install numpy tqdm haversine
```

Instale o PyTorch:

- **Sem GPU:** `pip install torch --index-url https://download.pytorch.org/whl/cpu`
- **Com GPU NVIDIA:** use o comando que [pytorch.org](https://pytorch.org/get-started/locally/) indica para a sua versão de CUDA.

Para confirmar se a GPU foi detectada:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

**Só se você não tiver o `.pkl` de divisão**, gere a partir do `.npy`. Leva uns 5 minutos e usa uns 24 GB de RAM:

```bash
python data_split.py
```

O script deve imprimir `train/valid/test = 80810/10113/10533`.

**Gere o arquivo compacto** (uma vez só; precisa de uns 24 GB de RAM e leva uns 10 minutos):

```bash
python compact_data.py --verify
```

Ele cria `data/fitrec_compact.npz` (374 MB) só com o que o modelo usa e confere que as batches são idênticas às do código original. Deve terminar com `VERIFY OK`. Depois disso, o treino não usa mais o `.npy` de 6 GB.

## 4. Rodar

Sempre de dentro de `FitRec-Attn/`:

```bash
# Teste rápido: 1 época, para validar que tudo funciona
python fitrect-attn.py 1

# Treino completo: até 50 épocas, com parada antecipada após 3 épocas sem melhora
python fitrect-attn.py

# Continuar um treino interrompido, a partir da última época concluída
python fitrect-attn.py --resume
```

### O que esperar

1. **Alguns segundos carregando os dados** (`loading data/fitrec_compact.npz`).
2. Depois aparece a barra de treino:
   ```
   epoch 0 train:   3%|  57/1830 [02:23<1:13:16, 2.48s/it, loss=12025.553]
   ```
   O número entre `<` e `,` é o tempo estimado para terminar a época.
3. No fim de cada época vêm a validação e a linha `valid loss`. O melhor modelo é salvo em `data/fitrec-attn/model_states/<execução>/`, junto com `last.pt`, o checkpoint que o `--resume` usa.
4. No final aparece a linha `| test loss X |`, que é o resultado do modelo.

### Tempo e loss

| Hardware | Tempo por época | Treino completo (10 a 20 épocas) |
|---|---|---|
| CPU (Intel Core Ultra 7, medido) | cerca de 1h15 | 12 a 25 horas |
| GPU NVIDIA ou Colab (T4, medido) | cerca de 15 min | 2 a 3 horas |

A loss é o erro quadrático (MSE) da frequência cardíaca em bpm². Ela começa perto de 20.000 e deve cair bastante. Para a previsão de curto prazo, o artigo (Tabela 6) reporta RMSE de 2,783 bpm para o FitRec-Attn (U/C), ou seja, uma loss perto de 7,7. A nossa execução no Colab chegou a uma test loss de 8,401, ou RMSE ≈ 2,90 bpm.

> **Dá para parar e continuar depois com `--resume`.** O checkpoint é salvo no fim de cada época, então interromper no meio de uma época perde só o progresso dela. Sem argumento, `--resume` continua a execução mais recente. Também aceita uma pasta: `--resume data/fitrec-attn/model_states/<execução>`.

Para deixar rodando e salvar o log num arquivo:

```bash
python -u fitrect-attn.py 2>&1 | tee treino.log
```

Cada execução cria uma pasta em `data/fitrec-attn/model_states/` com os checkpoints, o `config.json` (as opções usadas) e o `history.json` (a loss de treino e de validação de cada época).

## Experimentos

Sem opções, o script treina o modelo original. Estas opções mudam uma coisa de cada vez, para comparar com ele:

| Opção | O que muda |
|---|---|
| `--lr-decay 0.7` | A taxa de aprendizado é multiplicada por 0,7 a cada época, em vez de ficar fixa em 0,005. |
| `--no-context` | Ablação: tira o treino anterior do usuário (o codificador de contexto). |
| `--no-user` | Ablação: tira o embedding de usuário. |
| `--sport` | Inclui o esporte como entrada (embedding de 5 dimensões). |

```bash
python -u fitrect-attn.py --no-context 2>&1 | tee treino_sem_contexto.log
```

O nome da pasta da execução mostra as opções (por exemplo, `userId_2026-...` sem contexto). O `--resume` lê as opções do `config.json` da execução, então não é preciso repeti-las.

Para gerar as figuras de uma execução nova (ficam em `figures/<execução>/`):

```bash
python make_figures.py --run data/fitrec-attn/model_states/<execução>
```

## Rodar no Google Colab

Com o `fitrec_compact.npz`, o Colab gratuito (12 GB de RAM, GPU T4) deve bastar.

1. No seu Google Drive, crie a pasta `fitrec-data` e suba o `fitrec_compact.npz` para ela.
2. Abra `FitRec-Attn/colab_fitrec_attn.ipynb` no Colab (*Arquivo → Fazer upload de notebook*, ou pela aba GitHub).
3. Vá em *Ambiente de execução → Alterar o tipo de ambiente de execução → GPU (T4)*.
4. Na última célula, escolha o `EXPERIMENTO` (vazio para o modelo original) e rode todas as células. Os checkpoints ficam no Drive, em `fitrec-data/fitrec-attn/model_states/`.
5. Se o Colab desconectar (o gratuito desconecta depois de algumas horas ou se a aba ficar parada), mude `NOVA_EXECUCAO` para `False` e rode as células de novo. O notebook usa `--resume`, então continua da última época concluída.

## Problemas comuns

| Erro | Causa e solução |
|---|---|
| `DLL load failed ... Uma política de Controle de Aplicativo bloqueou este arquivo` | O Smart App Control do Windows está bloqueando uma biblioteca. O código já não importa pandas nem matplotlib por isso. Se acontecer com outro pacote, avise. |
| `FileNotFoundError: data/...` | Algum arquivo do passo 2 não está em `FitRec-Attn/data/`, ou o script foi rodado de outra pasta. |
| PC travando ou `MemoryError` | Só deve acontecer ao gerar o arquivo compacto (passo 3), que precisa de uns 24 GB livres. Feche outros programas. O treino com `fitrec_compact.npz` usa pouca RAM. |
| `gdown` falha ou dá 404 | Os links do Google Drive estão quebrados. Use o espelho da UCSD do passo 2. |
