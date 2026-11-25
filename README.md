# 🎬 LipSync Studio · Dublagem de vídeo com InfiniteTalk

![Modelo](https://img.shields.io/badge/Modelo-InfiniteTalk%2014B%20(fp8)-brightgreen.svg)
[![Testes](https://github.com/dgurgel-info/ai-colab-lipsync/actions/workflows/tests.yml/badge.svg)](https://github.com/dgurgel-info/ai-colab-lipsync/actions/workflows/tests.yml)
![Licença](https://img.shields.io/badge/Licen%C3%A7a-MIT-yellow.svg)

O **LipSync Studio** faz uma pessoa de um vídeo (ou de uma foto) falar um novo áudio. Ele usa o **[InfiniteTalk](https://github.com/MeiGen-AI/InfiniteTalk)**, da MeiGen-AI, que não troca só a boca: recria o quadro inteiro para que **lábios, rosto, cabeça e corpo** acompanhem a fala, mantendo a identidade da pessoa.

É o par natural do [Matraca Studio](https://github.com/dgurgel-info/ai-colab-matracastudio): o Matraca dubla a voz e o LipSync Studio sincroniza o vídeo com essa dublagem.

Roda no **Google Colab**, direto no navegador. O modelo e todas as dependências são baixados no próprio Colab; nada é instalado no seu computador.

## O que ele faz

- Dubla vídeos: a pessoa passa a falar o novo áudio, com boca e expressões no ritmo da fala.
- Anima fotos: a partir de uma foto do rosto, gera um vídeo da pessoa falando.
- Divide vídeos longos em partes, que você pode ouvir, refazer com outra semente e remontar, sem refazer tudo.
- Ajusta durações diferentes: se o áudio for maior, o vídeo vai e volta (ou recomeça); se for menor, o vídeo é cortado (ou o áudio é completado com silêncio).
- Entrega MP4 em 480p ou 720p, 25 fps, com o **áudio original em alta qualidade** (48 kHz, AAC 192 kbps).

## Como usar no Google Colab

1. Abra o notebook:

   [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dgurgel-info/ai-colab-lipsync/blob/main/LipSync_Studio.ipynb)

2. Em **Ambiente de execução → Alterar tipo de ambiente de execução**, escolha a GPU **L4**, **A100** ou **H100** (com RAM alta, se houver a opção).
3. Execute o **Passo 1** (base) e o **Passo 2** (instala o InfiniteTalk, baixa ~32 GB de pesos e carrega o modelo). Na primeira vez o Passo 2 leva de 15 a 30 minutos.
4. Execute o **Passo 3** e abra o endereço `gradio.live` que aparece. É a interface do LipSync Studio.
5. Quando terminar, execute o **Passo 4** para baixar todos os vídeos num ZIP.

Na interface, a aba **🎬 Dublar vídeo** guia você em cinco etapas: pessoa, áudio, ajustes, geração e resultado. A aba **🧩 Revisão por partes** serve para ouvir cada parte, gerar de novo a que não ficou boa e montar o vídeo final.

> **Dica:** use um vídeo com **uma pessoa**, rosto visível e bem iluminado. Descreva a cena no prompt, de preferência em inglês (ex.: *"A woman is talking to the camera in a studio"*).

> **Segurança:** o Passo 3 cria um link público **sem senha**. Quem tiver o link pode usar a GPU da sua sessão. Não compartilhe o link e encerre a sessão do Colab quando terminar.

## O modelo

| Componente | Origem | Para que serve |
| :--- | :--- | :--- |
| **InfiniteTalk (fp8)** | [MeiGen-AI/InfiniteTalk](https://huggingface.co/MeiGen-AI/InfiniteTalk) | DiT de 14B (Wan2.1 + adaptador de áudio do InfiniteTalk), quantizada em fp8 |
| **T5 (fp8)** | [MeiGen-AI/InfiniteTalk](https://huggingface.co/MeiGen-AI/InfiniteTalk) | Codificador do prompt |
| **VAE, CLIP e tokenizadores** | [Wan-AI/Wan2.1-I2V-14B-480P](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P) | Codificação do vídeo e da imagem de referência |
| **chinese-wav2vec2-base** | [TencentGameMate](https://huggingface.co/TencentGameMate/chinese-wav2vec2-base) | Codificador de áudio usado no treino do InfiniteTalk (funciona com qualquer idioma) |

### Como o modelo é carregado

- **Tudo no Colab.** O Passo 2 baixa o código oficial do InfiniteTalk (commit fixo), cria um ambiente isolado com Python 3.10 e as versões que ele exige (PyTorch 2.4.1, xformers 0.0.28, flash-attn 2.7.4) e baixa os pesos do Hugging Face.
- **Versão fp8 (~32 GB).** Em vez da DiT completa (~66 GB em fp32), usa a versão quantizada oficial, que cabe na RAM e na VRAM do Colab.
- **Memória da GPU automática.** Com 40 GB ou mais (A100, H100), a DiT fica inteira na GPU. Na L4 (24 GB), as camadas são trazidas da RAM conforme o uso (modo oficial de pouca VRAM: mais lento, mas cabe). Dá para forçar um dos modos no parâmetro `MEMORIA_GPU` do Passo 2.
- **Uma vez só.** O modelo é carregado uma vez e fica pronto entre as gerações. Se o processo cair, ele é reiniciado sozinho na próxima geração.
- **Reexecutar o Passo 2** só refaz o que faltar (o ambiente e os pesos ficam em `/content/lipsync_engine` enquanto a sessão do Colab existir).

### Velocidade e qualidade

O InfiniteTalk é um modelo de 14B: cada janela de ~3 s de vídeo passa por dezenas de passos de difusão. Conte com **vários minutos por parte de 30 s** (bem menos na A100/H100). O painel mostra o passo atual e o tempo estimado da parte.

| Ajuste | Efeito |
| :--- | :--- |
| **Velocidade × qualidade** | 20, 30 ou 40 passos (40 é o valor oficial). |
| **TeaCache** | Reaproveita passos parecidos: ~2× mais rápido, com perda pequena. Ligado por padrão. |
| **Resolução** | 480p é o recomendado. 720p é bem mais lento e usa mais memória. |
| **Força do áudio** | Mais alto = boca mais fiel ao áudio (3 a 5 é o ideal). |
| **Tamanho de cada parte** | Partes menores usam menos memória e podem ser refeitas isoladamente. |

O InfiniteTalk gera o quadro no tamanho do "balde" de proporção mais próximo (ex.: 16:9 vira 896×448 em 480p, com recorte central). A interface mostra o tamanho de saída antes de gerar.

## Se algo der errado

| Problema | O que fazer |
| :--- | :--- |
| O Passo 2 recusa a GPU | Troque para L4, A100 ou H100. A T4 não tem bfloat16 nem memória suficiente. |
| A instalação ou o download falhou | Execute o Passo 2 de novo: ele continua de onde parou. O registro completo fica em `/content/lipsync_engine/install.log`. |
| O modelo não carrega | Veja `/content/lipsync_engine/worker.log`. Na L4, ative **RAM alta**; se não houver, escolha `Pouca VRAM` no `MEMORIA_GPU`. |
| "A GPU ficou sem memória" | Use 480p, partes menores ou `Pouca VRAM` no Passo 2. |
| Uma parte ficou estranha | Na aba **🧩 Revisão por partes**, ouça a parte, sorteie outra semente, gere de novo e monte o vídeo final. |
| Perdi os vídeos depois de reiniciar o Colab | As pastas `outputs/` e `sessoes/` são apagadas ao reiniciar. Baixe o ZIP no Passo 4 (ou copie para o Google Drive) antes de encerrar. |

## Para quem desenvolve

```
lipsync_studio/
├── media.py            FFmpeg: 25 fps, ajuste de duração, partes exatas por quadro, montagem
├── session.py          sessão e partes (sessoes/<id>/session.json)
├── studio.py           fluxo: preparar → gerar partes → montar
├── jobs.py             tarefa em segundo plano (sobrevive ao F5)
├── engine/
│   ├── infinitetalk.py instala o venv, baixa os pesos e controla o worker
│   ├── worker.py       roda DENTRO do venv: carrega o InfiniteTalk e atende por HTTP local
│   ├── download.py     roda DENTRO do venv: baixa os pesos fp8 do Hugging Face
│   └── fake.py         motor simulado (testes e prévia local)
└── ui/                 interface Gradio (mesmo design system do Matraca Studio)
```

Prévia da interface no seu computador, sem GPU e sem baixar o modelo (usa um motor simulado): `pip install -r requirements-dev.txt` e `python scripts/preview_ui.py`. Testes automáticos: `python -m pytest -q tests`.

## Licença

O projeto é distribuído sob a licença [MIT](LICENSE). O InfiniteTalk e seus pesos são distribuídos pela MeiGen-AI sob a licença Apache 2.0; o Wan2.1 e o chinese-wav2vec2 têm termos próprios. Confira-os antes de uso comercial ou redistribuição, e use o modelo de forma responsável: não gere vídeos de pessoas sem o consentimento delas.
