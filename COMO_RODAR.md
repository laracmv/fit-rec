# Como rodar o FitRec-Attn

Guia para treinar o modelo **FitRec-Attn** (previsão de frequência cardíaca de curto prazo) a partir deste fork. As correções necessárias para rodar com Python, PyTorch e NumPy atuais, em CPU ou GPU e no Windows, estão na branch `fitrec-attn-run`.

> O modelo **FitRec** (pasta `FitRec/`) usa Keras 2.0 e não roda em Python moderno sem ser reescrito. Este guia cobre só o **FitRec-Attn**.

## Requisitos

| Item | Mínimo |
|---|---|
| RAM | **32 GB**. Carregar os dados chega a uns 24 GB. |
| Disco | Uns 7 GB livres para os dados. |
| Python | 3.11, via Anaconda/Miniconda. |
| GPU | Opcional. Uma GPU NVIDIA acelera muito o treino. |

## 1. Baixar o código

```bash
git clone -b fitrec-attn-run https://github.com/laracmv/fit-rec.git
cd fit-rec/FitRec-Attn
mkdir data
```

## 2. Colocar os dados em `FitRec-Attn/data/`

Os dados não estão no git, por causa do tamanho e da licença. Coloque estes arquivos na pasta `data/`:

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

## 4. Rodar

Sempre de dentro de `FitRec-Attn/`:

```bash
# Teste rápido: 1 época, para validar que tudo funciona
python fitrect-attn.py 1

# Treino completo: até 50 épocas, com parada antecipada após 3 épocas sem melhora
python fitrect-attn.py
```

### O que esperar

1. **Uns 5 minutos carregando os dados**, sem barra de progresso. É normal a RAM subir até uns 24 GB.
2. Depois aparece a barra de treino:
   ```
   epoch 0 train:   1%|  16/1830 [01:20<2:05:29, 4.15s/it, loss=15774.699]
   ```
   O número entre `<` e `,` é o tempo estimado para terminar a época.
3. No fim de cada época vêm a validação e a linha `valid loss`. O melhor modelo é salvo em `data/fitrec-attn/model_states/`.
4. No final aparece a linha `| test loss X |`, que é o resultado do modelo.

### Tempo e loss

| Hardware | Tempo por época | Treino completo (10 a 20 épocas) |
|---|---|---|
| CPU (medido num notebook) | cerca de 2h30 | 1 a 2 dias |
| GPU NVIDIA | estimado bem menor | algumas horas |

A loss é o erro quadrático (MSE) da frequência cardíaca em bpm². Ela começa perto de 20.000 e deve cair bastante. O artigo reporta RMSE de uns 6 a 7 bpm, o que corresponde a uma loss perto de 40 a 50.

> **Não dá para pausar e continuar depois.** O modelo só é salvo no fim de cada época, então interromper no meio de uma época perde o progresso dela.

Para deixar rodando e salvar o log num arquivo:

```bash
python -u fitrect-attn.py 2>&1 | tee treino.log
```

## Problemas comuns

| Erro | Causa e solução |
|---|---|
| `DLL load failed ... Uma política de Controle de Aplicativo bloqueou este arquivo` | O Smart App Control do Windows está bloqueando uma biblioteca. O código já não importa pandas nem matplotlib por isso. Se acontecer com outro pacote, avise. |
| `FileNotFoundError: data/...` | Algum arquivo do passo 2 não está em `FitRec-Attn/data/`, ou o script foi rodado de outra pasta. |
| PC travando ou `MemoryError` | Falta RAM. Feche outros programas, porque são necessários uns 24 GB livres. |
| `gdown` falha ou dá 404 | Os links do Google Drive estão quebrados. Use o espelho da UCSD do passo 2. |
