<div align="center">
  <img src="./misc/fastc.svg" alt="Logo" height="100" />
  <p><strong>Unattended Lightweight Text Classifiers with LLM Embeddings</strong></p>
</div>
<br/>
<p align="center">
    <a href="https://pypi.python.org/pypi/fastc/"><img alt="PyPi" src="https://img.shields.io/pypi/v/fastc.svg?style=flat-square"></a>
    <a href="https://github.com/EveripediaNetwork/fastc/releases"><img alt="GitHub releases" src="https://img.shields.io/github/release/EveripediaNetwork/fastc.svg?style=flat-square"></a>
    <a href="https://github.com/EveripediaNetwork/fastc/blob/master/LICENSE"><img alt="License" src="https://img.shields.io/github/license/EveripediaNetwork/fastc.svg?style=flat-square"></a>
</p>


# Key features
- **Suitable for limited-memory CPU execution:** Use efficient distilled models such as [`deepset/tinyroberta-6l-768d`](https://huggingface.co/deepset/tinyroberta-6l-768d) for embedding generation.
- **Logistic Regression and Nearest Centroid classification:** Bypass the need for fine-tuning by utilizing LLM embeddings to efficiently categorize texts using either logistic regression or the nearest centroid through cosine similarity.
- **Efficient Parallel Execution:** Run hundreds of classifiers concurrently with minimal overhead by sharing the same model for embedding generation.

# Installation
```bash
pip install -U fastc
```

# Train a model
You can train a text classifier with just a few lines of code:
```python
from fastc import Fastc

tuples = [
    ("I just got a promotion! Feeling fantastic.", 'positive'),
    ("Today was terrible. I lost my wallet and missed the bus.", 'negative'),
    ("I had a great time with my friends at the party.", 'positive'),
    ("I'm so frustrated with the traffic jam this morning.", 'negative'),
    ("My vacation was wonderful and relaxing.", 'positive'),
    ("I didn't get any sleep last night because of the noise.", 'negative'),
    ("I'm so excited for the concert tonight!", 'positive'),
    ("I'm disappointed with the service at the restaurant.", 'negative'),
    ("The weather is beautiful and I enjoyed my walk.", 'positive'),
    ("I had a bad day. Nothing went right.", 'negative'),
    ("I'm thrilled to announce that we are expecting a baby!", 'positive'),
    ("I feel so lonely and sad today.", 'negative'),
    ("My team won the championship! We are the champions.", 'positive'),
    ("I can't stand my job anymore, it's so stressful.", 'negative'),
    ("I love spending time with my family during the holidays.", 'positive'),
    ("My computer crashed and I lost all my work.", 'negative'),
    ("I'm proud of my achievements this year.", 'positive'),
    ("I'm exhausted and overwhelmed with everything.", 'positive'),
]
```

## Classification Kernels
### Nearest Centroid
```python
model = Fastc(
    embeddings_model='microsoft/deberta-base',
    kernel=Kernels.NEAREST_CENTROID,
)

model.load_dataset(tuples)
model.train()
```

### Logistic Regression
```python
from fastc import Kernels

model = Fastc(
    embeddings_model='microsoft/deberta-base',
    kernel=Kernels.LOGISTIC_REGRESSION,
    # cross_validation_splits=5,
    # cross_validation_repeats=3,
    # iterations=100,
    # parameters={...},
    # seed=1984,
)

model.load_dataset(tuples)
model.train()
```

## Pooling Strategies
The implemented pooling strategies are:
- `MEAN` (default)
- `MEAN_MASKED`
- `MAX`
- `MAX_MASKED`
- `CLS`
- `SUM`
- `ATTENTION_WEIGHTED`

```python
from fastc import Pooling

model = Fastc(
    embeddings_model='microsoft/deberta-base',
    pooling=Pooling.MEAN_MASKED,
)

model.load_dataset(tuples)
model.train()
```

## Templates and Instruct Models
You can use instruct templates with instruct models such as `intfloat/multilingual-e5-large-instruct`. Other models may also improve in performance by using templates, even if they were not explicitly trained with them.

```python
from fastc import ModelTemplates, Fastc, Template

# template_text = 'Instruct: {instruction}\nQuery: {text}'
template_text = ModelTemplates.E5_INSTRUCT

model = Fastc(
    embeddings_model='intfloat/multilingual-e5-large-instruct',
    template=Template(
        template_text,
        instruction='Classify as positive or negative'
    ),
)
```

# Save, load and export models
After training, you can save the model for future use:
```python
model.save_model('./sentiment-classifier/')
```

## Publish a model to HuggingFace
> [!IMPORTANT]  
> Log in to HuggingFace first with `huggingface-cli login`

```python
model.push_to_hub(
    'braindao/sentiment-classifier',
    tags=['sentiment-analysis'],
    languages=['multilingual'],
    private=False,
)
```

## Load an existing model
You can load a pre-trained model either from a directory or from HuggingFace:
```python
# From a directory
model = Fastc('./sentiment-classifier/')

# From HuggingFace
model = Fastc('braindao/sentiment-classifier')
```

# Class prediction
```python
sentences = [
    'I am feeling well.',
    'I am in pain.',
]

# Single prediction
scores = model.predict_one(sentences[0])
print(scores['label'])

# Batch predictions
scores_list = model.predict(sentences)
for scores in scores_list:
    print(scores['label'])
```

# Inference Server

To launch the dockerized inference server, use the following script:
```bash
./server/scripts/start-docker.sh
```

Alternatively, on the host machine:
```bash
./server/scripts/start-server.sh
```

In both cases, an HTTP API will be available, listening on the `fastc-server` *[hashport](https://github.com/labteral/hashport)* `53256`.

## Inference

To classify text, use `POST /` with a JSON payload such as:
```json
{
    "model": "braindao/tinyroberta-6l-768d-language-identifier-en-es-ko-zh-fastc-lr",
    "text": "오늘 저녁에 친구들과 함께 pizza를 먹을 거예요."
}
```

Response:
```json
{
    "label": "ko",
    "scores": {
        "en": 1.0146501463135055e-08,
        "es": 6.806091549848057e-09,
        "ko": 0.9999852640487916,
        "zh": 1.471899861513275e-05
    }
}
```

## Version

To check the `fastc` version, use `GET /version`:

Response:
```json
{
    "version": "2.2407.0"
}
```
