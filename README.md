[![license](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/skplanet/Dialog-KoELECTRA/blob/master/LICENSE)


# Dialog-KoELECTRA 언어모델

<br>

## 모델 소개


**Dialog-KoELECTRA**는 대화체에 특화된 언어 모델입니다. 대화체는 채팅이나 통화에서 사용하는 어체를 말합니다. 기존 언어 모델들이 문어체를 기반으로 학습 되었기 때문에 저희는 대화체에 적합한 언어 모델을 만들게 되었습니다. 또한, 실제 서비스에 적합한 가벼운 모델을 만들고자 하여 small 모델부터 공개하게 되었습니다. Dialog-KoELECTRA 모델은 가볍지만 대화체 태스크에서는 기존 base 모델과 비슷한 성능을 보여줍니다.

Dialog-KoELECTRA 모델은 22GB의 대화체 및 문어체 한글 텍스트 데이터로 훈련되었습니다. Dialog-KoELECTRA 모델은 [ELECTRA](https://openreview.net/pdf?id=r1xMH1BtvB) 모델을 기반으로 만들어졌습니다. ELECTRA는 자가지도 언어 표현 학습 방법으로 비교적 적은 계산을 사용하여 사전 언어 모델을 훈련할 수 있습니다. ELECTRA 모델은 [GAN](https://arxiv.org/pdf/1406.2661.pdf)과 유사하게 생성자가 생성한 "가짜" 입력 토큰과 "실제" 입력 토큰을 판별자가 구별하도록 훈련되었습니다. ELECTRA는 단일 GPU에서 학습했을 때도 강력한 결과를 얻을 수 있는 장점을 가지고 있습니다.

### 모델 차별점
1. 최적의 어휘 사전 생성
	- 여러 파라미터 값 조합 실험을 통해 어휘 사전 생성시 최적의 값으로 설정
2. 최적의 대화체/문어체 데이터 비율 구성
	- 대화체와 문어체 데이터 비율 조합을 통해 대화체 성능 향상
3. 형태소 분석 기반 토크나이저 사용
	- 한글 처리에 적합한 형태소 분석 결과를 subword 기반의 tokenizer 입력으로 사용하여 어휘 사전 생성
4. 사전 학습시 mixed precision 적용
	- 사전 훈련 중에 mixed precision 옵션을 사용하여 학습 속도 향상 및 메모리 절약
5. fine-tuning시 NNI 옵션 사용 가능
	- 모델 fine-tuning시 [NNI](https://github.com/microsoft/nni) 옵션을 사용하여 파라미터 최적화가 가능


<br>

## Released Models

서비스에 적합한 small 버전을 먼저 출시하였습니다. 향후 base 모델 등 다른 모델도 출시 할 예정입니다.

| Model | Layers | Hidden Size | Params | Max<br/>Seq Len | Learning<br/>Rate | Batch Size | Train Steps  | Train Time |
| :---: | :---: | :---: | :---: | :---:  | :---: | :---:  | :---:  | :---: | 
| Dialog-KoELECTRA-Small | 12 | 256 | 14M | 128 | 1e-4 | 512 | 1M | 28일 |

<br>

### transformers 라이브러리를 통한 사용법

Dialog-KoELECTRA 모델은 Hugging Face에 업로드되어 있어 쉽게 사용 가능합니다.

```python
from transformers import ElectraTokenizer
from transformers.modeling_electra import ElectraForSequenceClassification
  
tokenizer = ElectraTokenizer.from_pretrained("skplanet/dialog-koelectra-small-discriminator")

model = ElectraForSequenceClassification.from_pretrained("skplanet/dialog-koelectra-small-discriminator")
```

<br>

Transformers 라이브러리를 사용하지 않고 직접 모델을 다운로드하려면 아래 링크를 통해 다운로드 할 수 있습니다.


<br>

| Model | Pytorch-Generator | Pytorch-Discriminator | Tensorflow-v1 | ONNX |
| :---: | :---: | :---: | :---: | :---: |
| Dialog-KoELECTRA-Small | [link](https://drive.google.com/file/d/1uaAcnfLKqftmvMG7oB7jt66J1sQ0hyuW/view?usp=sharing) | [link](https://drive.google.com/file/d/1TdINJDUB5Imfbo9JWHlO15HAFsLu7DAo/view?usp=sharing) | [link](https://drive.google.com/file/d/13smn2Bt5Zjxkr7DNIPyGK0nFLErDRhAR/view?usp=sharing) |[link](https://drive.google.com/file/d/1EM0Krkylnhm5j83jPP9uiRa88skc7yTi/view?usp=sharing)|

<br>

## 모델 성능

<table>
<thead>
<tr>
<th style="text-align:center"></th>
<th style="text-align:center" colspan="3">대화체</th>
<th style="text-align:center" colspan="3">문어체</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align:center"></td>
<td style="text-align:center"><strong>NSMC (acc)</strong></td>
<td style="text-align:center"><strong>Question Pair (acc)</strong></td>
<td style="text-align:center"><strong>Korean-Hate-Speech (F1)</strong></td>
<td style="text-align:center"><strong>Naver NER (F1)</strong></td>
<td style="text-align:center"><strong>KorNLI (acc)</strong></td>
<td style="text-align:center"><strong>KorSTS (spearman)</strong></td>
</tr>
<tr>
<td style="text-align:center">DistilKoBERT</td>
<td style="text-align:center">88.60</td>
<td style="text-align:center">92.48</td>
<td style="text-align:center">60.72</td>
<td style="text-align:center">84.65</td>
<td style="text-align:center">72.00</td>
<td style="text-align:center">72.59</td>
</tr>
<tr>
<td style="text-align:center">KoELECTRA-Small</td>
<td style="text-align:center">89.36</td>
<td style="text-align:center">94.85</td>
<td style="text-align:center">63.07</td>
<td style="text-align:center">85.40</td>
<td style="text-align:center"><strong>78.60</strong></td>
<td style="text-align:center"><strong>80.79</strong></td>
</tr>
<tr>
<td style="text-align:center"><strong>Dialog-KoELECTRA-Small</strong></td>
<td style="text-align:center"><strong>90.01</strong></td>
<td style="text-align:center"><strong>94.99</strong></td>
<td style="text-align:center"><strong>68.26</strong></td>
<td style="text-align:center"><strong>85.51</strong></td>
<td style="text-align:center">78.54</td>
<td style="text-align:center">78.96</td>
</tr>
</tbody>
</table>


<br>

## 학습데이터


<table class="tg">
<thead>
  <tr>
    <th class="tg-c3ow"></th>
    <th class="tg-c3ow">corpus name</th>
    <th class="tg-c3ow">size</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-c3ow" rowspan="4">대화체</td>
    <td class="tg-0pky"><a href="https://aihub.or.kr/aidata/85" target="_blank" rel="noopener noreferrer">Aihub Korean dialog corpus</a></td>
    <td class="tg-c3ow" rowspan="4">7GB</td>
  </tr>
  <tr>
    <td class="tg-0pky"><a href="https://corpus.korean.go.kr/" target="_blank" rel="noopener noreferrer">NIKL Spoken corpus</a></td>
  </tr>
  <tr>
    <td class="tg-0pky"><a href="https://github.com/songys/Chatbot_data" target="_blank" rel="noopener noreferrer">Korean chatbot data</a></td>
  </tr>
  <tr>
    <td class="tg-0pky"><a href="https://github.com/Beomi/KcBERT" target="_blank" rel="noopener noreferrer">KcBERT</a></td>
  </tr>
  <tr>
    <td class="tg-c3ow" rowspan="2">문어체</td>
    <td class="tg-0pky"><a href="https://corpus.korean.go.kr/" target="_blank" rel="noopener noreferrer">NIKL Newspaper corpus</a></td>
    <td class="tg-c3ow" rowspan="2">15GB</td>
  </tr>
  <tr>
    <td class="tg-0pky"><a href="https://github.com/lovit/namuwikitext" target="_blank" rel="noopener noreferrer">namuwikitext</a></td>
  </tr>
</tbody>
</table>

<br>


## Vocabulary

어휘 사전 생성시 [huggingface_konlpy](https://github.com/lovit/huggingface_konlpy)를 이용한 형태소 분석을 적용했습니다.
실험 결과, 형태소 분석을 적용하지 않고 만든 어휘 사전보다 더 나은 성능을 보였습니다.
<table>
<thead>
  <tr>
    <th>vocabulary size</th>
    <th>unused token size</th>
    <th>limit alphabet</th>
    <th>min frequency</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>40,000</td>
    <td>500</td>
    <td>6,000</td>
    <td>3</td>
  </tr>
</tbody>
</table>

<br>

## 데모
- [Dialog-KoELECTRA Named Entity Recognition](https://share.streamlit.io/krevas/dialog-electra-ner)

<br>

## 코드 사용 방법
### Pre-training

데이터 전처리를 하려면`preprocess.py`를 사용하세요. 데이터 전처리는 반복되는 문자를 축약하고 한자를 제거하는 작업을 수행합니다.

* `--corpus_dir`: 텍스트 파일을 포함하는 디렉토리
* `--output_file`: 전처리 후 생성 되는 파일명

실행 예시
```python
python3 preprocess.py \
    --corpus_dir raw_data_dir \
    --output_file preprocessed_data.txt \
```

---

`build_vocab.py`를 사용하여 텍스트 데이터에서 어휘 파일을 만듭니다.

* `--corpus`: 어휘 파일로 변환 할 텍스트 파일
* `--tokenizer`: wordpiece / mecab_wordpiece와 같은 토크나이저의 이름 (기본값 : wordpiece)
* `--vocab_size`: 어휘사전 단어 수 (기본값 : 40000)
* `--min_frequency`: 토큰 페어가 병합 작업을 수행하는데 필요한 최소 빈도 (기본값 : 3)
* `--limit_alphabet`: 병합을 계산하기 전에 보관할 수있는 초기 토큰 수 (기본값 : 6000)
* `--unused_size`: 사전의 범용성을 위한 unused 개수 (기본값 : 500)

실행 예시
```python
python3 build_vocab.py \
    --corpus preprocessed_data.txt \
    --tokenizer mecab_wordpiece \
    --vocab_size 40000 \
    --min_frequency 3 \
    --limit_alphabet 6000 \
    --unused_size 500
```

---

사전학습을 위한 tfrecord를 만들려면`build_pretraining_dataset.py`를 사용하세요.

* `--corpus_dir`: tfrecord로 전환 할 텍스트 파일이 포함 된 디렉토리
* `--vocab_file`: build_vocab.py을 통해 만든 어휘 파일
* `--output_dir`: tfrecord 생성 디렉토리
* `--max_seq_length`: 최대 토큰 수 (기본값 : 128개)
* `--num_processes`: 프로세스 병렬화 개수 (기본값 : 1)
* `--blanks-separate-docs`: 빈 줄이 문서 경계를 나타내는 지 여부 (기본값 : False)
* `--do-lower-case/--no-lower-case`: 입력 텍스트의 소문자 여부 (기본값 : False)
* `--tokenizer_type`: wordpiece / mecab_wordpiece와 같은 토크나이저의 이름 (기본값 : wordpiece)

실행 예시
```python
python3 build_pretraining_dataset.py \
    --corpus_dir data/train_data/raw/split_normalize \
    --vocab_file data/vocab/vocab.txt \
    --tokenizer_type wordpiece \
    --output_dir data/train_data/tfrecord/pretrain_tfrecords_len_128_wordpiece_train \
    --max_seq_length 128 \
    --num_processes 8
```

---

`run_pretraining.py`를 사용하여 사전 학습을 수행합니다.

* `--data_dir`: 사전 훈련 데이터, 모델 가중치 등이 저장되는 디렉토리
* `--model_name`: 훈련중인 모델의 이름. 모델 가중치는 기본적으로`<data-dir> / models / <model-name>`에 저장
* `--hparams` (optional): 모델 매개 변수, 데이터 경로 등을 포함하는 JSON 파일 경로 (지원되는 매개 변수는`configure_pretraining.py`를 참조하세요.)
* `--use_tpu` (optional): 모델을 훈련 할 때 tpu를 사용하는 옵션
* `--mixed_precision` (optional): 모델 훈련시 혼합 정밀도를 사용할지 여부에 대한 옵션

실행 예시
```python
python3 run_pretraining.py \
    --data_dir data/train_data/tfrecord/pretrain_tfrecords_len_128_wordpiece_train \
    --model_name data/ckpt/pretrain_ckpt_len_128_small_wordpiece_train \
    --hparams data/config/small_config_kor_wordpiece_train.json \
    --mixed_precision
```

---

`pytorch_convert.py`를 사용하여 tf 모델을 pytorch 모델로 변환합니다.

* `--tf_ckpt_path`: 체크포인트 파일이 저장되는 디렉토리
* `--pt_discriminator_path`: pytorch 판별 모델을 작성할 위치
* `--pt_generator_path` (optional): pytorch 생성 모델을 작성할 위치
실행 예시
```python
python3 pytorch_convert.py \
    --tf_ckpt_path model/ckpt/pretrain_ckpt_len_128_small \
    --pt_discriminator_path model/pytorch/dialog-koelectra-small-discriminator \
    --pt_generator_path model/pytorch/dialog-koelectra-small-generator \
```
<br>

## Fine-tuning

`run_finetuning.py`를 사용하여 다운스트림 NLP 태스크에서 Dialog-KoELECTRA 모델을 미세 조정하고 평가합니다. 세 가지 인수가 필요합니다.

* `--config_file`: 모델 하이퍼 파라미터, 데이터 경로 등을 포함하는 YAML 파일
* `--nni`: 모델을 미세 조정할 때 nni를 사용할지 여부에 대한 옵션

실행 예시
```python
python3 run_finetune.py --config_file conf/hate-speech/electra-small.yaml
```

<br>

## References
- [ELECTRA](https://github.com/google-research/electra): Pre-training Text Encoders as Discriminators Rather Than Generators.
- [KoELECTRA](https://github.com/monologg/KoELECTRA): Pretrained ELECTRA Model for Korean

<br>

## Contact Info

Dialog-KoELECTRA 사용에 대한 도움이나 문제가 있는 경우 [GitHub issue](https://github.com/skplanet/Dialog-KoELECTRA/issues)에 올려주세요.

Dialog-KoELECTRA와 관련된 개인적인 커뮤니케이션은 [wonchul.kim@sk.com](https://github.com/krevas)으로 연락 주시기 바랍니다.

<br>

## Citation

이 라이브러리를 프로젝트 및 연구에 적용하는 경우, 아래와 같이 인용 부탁드립니다.

```
@misc{DialogKoELECTRA,
  author       = {Wonchul Kim and Junseok Kim and Okkyun Jeong},
  title        = {Dialog-KoELECTRA: Korean conversational language model based on ELECTRA model},
  howpublished = {\url{https://github.com/skplanet/Dialog-KoELECTRA}},
  year         = {2021},
}
```

<br>

## License

Dialog-KoELECTRA 프로젝트는 [Apache License 2.0](https://github.com/skplanet/Dialog-KoELECTRA/blob/master/LICENSE) 라이센스를 기반으로 합니다.

```
 Copyright 2020 ~ present SK Planet Co. RB Dialog solution

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.      
```
