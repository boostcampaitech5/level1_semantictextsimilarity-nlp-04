# STS_NLP_팀 리포트(04)

# 프로젝트 개요

## 프로젝트 주제 - Semantic Textual Similarity

중복된 문장을 검출하고 수정하기 위해 우수한 성능의 문장 유사도를 구해주는 Semantic Text Similarity 대회

학습 데이터셋 9,324개, 검증 데이터셋 550개, 평가 데이터는 1,100개 제공

## 프로젝트 환경

- 팀 구성 및 컴퓨팅 환경 : 5인 1팀 / V100 server 인당 1대
- 협업 환경 : Github, Notion
- 실험관리 : Wandb, CSVLogger
- 의사소통 : Slack, 카카오톡, Zoom

# 팀 구성 및 역할

- **김성우**

W&B sweep 기본 코드 작성 / 모델 성능 실험 / 팀원들의 조사 내용 바탕으로 Loss function, weight decay 등의 영향 파악

- **김윤희**

Data augmentation, 모델 성능 실험

- **이도현**

Data augmentation(EDA), 모델 성능 실험, Classifier head 설계 

- **진정민**

팀 협업 리딩, 모델 디버깅, 모델 성능 실험

- **최규빈**

Back translation, 모델 리서치, loss function 구축, Ensemble

# 프로젝트 수행 절차 및 방법

## A. 팀 목표 설정

(1) pearson 0.9 이상 달성, 상위 25% 이내 달성

(2) EDA와 모델 성능 실험

(3) 다양한 metric을 사용해보기

## B. 협업 문화

(1) 데일리 스크럼과 Pear Session때 각자 진행상황 공유하기 / 마주한 에러를 디버깅해보는 시간 가지기

(2) Github을 이용하여 코드 공유

(3) 좋은 인사이트 Notion과 카카오톡을 통해 자유롭게 공유, 주기적으로 실험 과정 정리

## C. 수행 과정

(1) STS 문제 정의

(2) 대회 metric(피어슨 상관계수)에 적합한 모델 및 loss function 탐색

(3) Baseline 코드를 역할에 맞게 객체화, 세분화

(4) 실험해 볼 Hyper parameter들 정의 및 모델 구체화

(5) 성능이 우수한 제출본들을 토대로 ensemble

# 수행 내용

## 1. EDA (Exploratory Data Analysis)

![Untitled](STS_NLP_%E1%84%90%E1%85%B5%E1%86%B7%20%E1%84%85%E1%85%B5%E1%84%91%E1%85%A9%E1%84%90%E1%85%B3(04)%204046bfdb5ce24724afbb40fd838c68eb/Untitled.png)

![Untitled](STS_NLP_%E1%84%90%E1%85%B5%E1%86%B7%20%E1%84%85%E1%85%B5%E1%84%91%E1%85%A9%E1%84%90%E1%85%B3(04)%204046bfdb5ce24724afbb40fd838c68eb/Untitled%201.png)

Train dataset의 label 분포를 확인했을 때, 다른 값에 비해 0.0에 많은 데이터가 몰려 있어, label 불균형이 존재함을 알 수 있다.

- train.csv 파일의 경우 label 0.0인 값이 50% 이상으로 데이터 불균형이 존재하는 것을 확인했다.
- 반면 dev.csv 의 경우 label의 분포가 고르게 나타남을 알 수 있었다.

## 2. Data Augmentation

### 2.1. EDA (Easy Data Augmentation)

- Train data의 수는 9324개로, 충분하지 않다고 판단되었으며, Generalization 성능을 올리기 위해 데이터 증강은 필수적이라고 생각하였다.
- 텍스트 데이터 증강을 위한 방법론이 많지는 않지만, 우리는 그 중 신뢰성있다고 판단되는 
****EDA: Easy Data Augmentation Techniques for Boosting Performance on Text Classification Tasks**** 을 적용해보았다.
- EDA는 SR, RI, RS, RD라는 간단한 4개의 Operation으로 동작하며, 한국어로 구현된 오픈소스는 https://github.com/catSirup/KorEDA 를 사용하였다.
- 3배 정도 증강한 결과, 성능 향상은 있었으나 유의미한 결과라고 판단되지는 않았다.
    - 먼저, EDA 특성상 문장이 굉장히 Dramatic하게 바뀌지 않았다.
    - 또한 두 문장의 특정 단어가 없어지거나 바뀌면서 의미가 크게 달라져, 두 문장의 유사도 측정에 오히려 방해가 될 것이라 생각하였다.
    - 성능 향상이 된 이유는, 증강한 데이터가 원본 데이터와 크게 다르지 않으므로, 단순히 epoch을 늘린 효과 같다는 생각도 들었다. (즉, 원본 데이터로 3epoch 학습 = 증강한 데이터로 1epoch 학습)
- 따라서 최종적으로 EDA를 통한 데이터증강은 실시하지 않았다.

### 2.2. Back-translation

- Back-translation이란 번역 모델을 활용하여, Korean → English → Korean 처럼 번역하여 데이터를 증강하는 방법이다.
- 직관적으로 생각하였을때, 효율적인 데이터 증강 방법이라고 생각하였으나, 결과가 번역모델의 성능에 크게 좌우되었다.
- Google API를 통해 번역을 진행하였으나, 생각보다 문장의 퀄리티가 좋지 않았다.
    - 어순이 정리되지 않은 경우
    - ‘그것’, ‘이것’ 과 같은 부자연스러운 어휘가 자주 등장

### 2.3. Sentence Switch

- sentence 1과 sentence 2의 순서를 바꿔 데이터 증강
- sentence 1과 sentence 2의 embedding값이 달라 유의미한 증강 방법이라고 생각했다.

### 2.4. Undersampling + Oversampling

- train데이터의 50% 이상이 label 0이었기 때문에 해당 Label을 under sampling 하였습니다.
- label 5인 데이터는 label 0인 데이터보다 4배 적었기 때문에 sentence 1을 sentence 2로 복사하여 label 5 데이터 증강

![스크린샷 2023-04-20 오후 5.42.00.png](STS_NLP_%E1%84%90%E1%85%B5%E1%86%B7%20%E1%84%85%E1%85%B5%E1%84%91%E1%85%A9%E1%84%90%E1%85%B3(04)%204046bfdb5ce24724afbb40fd838c68eb/%25E1%2584%2589%25E1%2585%25B3%25E1%2584%258F%25E1%2585%25B3%25E1%2584%2585%25E1%2585%25B5%25E1%2586%25AB%25E1%2584%2589%25E1%2585%25A3%25E1%2586%25BA_2023-04-20_%25E1%2584%258B%25E1%2585%25A9%25E1%2584%2592%25E1%2585%25AE_5.42.00.png)

- 다음과 같이 label 0은 50% 언더 샘플링, 없앤 label 0의 데이터 중 50%를 이용해 label 5 오버 샘플링하였다.
- 데이터 label 불균형을 어느정도 해소하고 나니 모델의 성능이 향상되어 유의미한 기법이라고 판단했다.

## 3. Model Selection

RoBERTa와 ELECTRA 모델을 사용하여 실험했다. RoBERTa는 더 큰 데이터와 큰 배치사이즈를 사용해 성능을 크게 개선한 모델로 Dynamic Masking을 도입했다. ELECTRA는 기존 BERT의 학습 효율성의 단점을 보완하기 위해 Replaced Token Detection을 도입한 모델이다. BERT와는 달리 Discriminator 구조를 사용함으로 전체 token 중 15%만 학습하던  것과 달리 모든 token에 대해 학습하는 효과를 얻을 수 있어 높은 성능을 보이는 모델이다. 모든 token에 대해 사전학습하는 기법으로 더 높은 성능과 RoBERTa 기반 모델의 실험과 앙상블 효과를 기대하기 위해 ELECTRA를 선택했다.

각 모델별로 

- 전처리방식(특수문자제거 + 영어소문자) 적용의 유무
- Augmentation(back translation / Sentece 1 Sentence 2 switch)
- epoch, batch size와 loss function

에 변화를 주어 다양한 실험을 하였다. 

## 4. Learning Rate Scheduling

실험을 진행하면서 learning rate는 1e-5 값으로 통일하여 실행했지만, learning rate scheduler로 `StepLR`을 사용하였다. `StepLR` 외에도 `CosineAnnealingLR` , `CyclicLR` 등을 적용해보고자 했다. 하지만 이런 learning rate scheduler들은 유효하게 이용하려면 epoch를 길게 하여 실험해야 했다. 다양한 실험을 위해 epoch를 짧게 하여 실험을 진행해 왔기 때문에 이번 실험에서 `CosineAnnealingLR` , `CyclicLR` 에 대해 충분한 검증을 할 수는 없으므로 `StepLR`을 주로 사용했다. 

![StepLR(step_size=2, gamma=0.7) 에 의한 learning rate decay](STS_NLP_%E1%84%90%E1%85%B5%E1%86%B7%20%E1%84%85%E1%85%B5%E1%84%91%E1%85%A9%E1%84%90%E1%85%B3(04)%204046bfdb5ce24724afbb40fd838c68eb/WB_Chart_4_21_2023_11_36_06_AM.png)

StepLR(step_size=2, gamma=0.7) 에 의한 learning rate decay

```python
scheduler = StepLR(optimizer, step_size=1, gamma=0.5) # 1 epoch마다 lr이 0.5의 비율로 줄어듦
scheduler = StepLR(optimizer, step_size=2, gamma=0.7) # 2 epoch마다 lr이 0.7의 비율로 줄어듦
```

## 5. Loss Function

Regression에서 사용할 수 있는 대표적인 Loss function 3가지를 적용해보았다.

### 5.1. L1-Loss (MAE)

![Untitled](STS_NLP_%E1%84%90%E1%85%B5%E1%86%B7%20%E1%84%85%E1%85%B5%E1%84%91%E1%85%A9%E1%84%90%E1%85%B3(04)%204046bfdb5ce24724afbb40fd838c68eb/Untitled%202.png)

- 기본 Baseline에 설정되어 있던 Loss이다.
- $\space (Y_i-\hat{Y}_i) = 0$ 일 때 미분이 불가능하다는 단점이 있다.
- outlier 데이터가 있을 때 MSE보다 효과적이라고 한다.
- 우리 데이터는 0~5로 범위가 정해져 있기에, outlier는 없다고 생각하였다.
- 하지만, MSE와 큰 성능 차이가 나지 않아 해당 Loss를 주로 사용하였다.

### 5.2. L2-Loss (MSE)

![Untitled](STS_NLP_%E1%84%90%E1%85%B5%E1%86%B7%20%E1%84%85%E1%85%B5%E1%84%91%E1%85%A9%E1%84%90%E1%85%B3(04)%204046bfdb5ce24724afbb40fd838c68eb/Untitled%203.png)

- $error \space (Y_i-\hat{Y}_i)$ 가 클수록 quadratic하게 Loss 가 커지는 단점이 있다.
- STS 학습 예시를 찾아보았을때, 보통 MSE loss를 많이 사용하였다.
- MAE보다 성능이 좋을 것이라 예상하였으나, 큰 성능 향상을 보이진 않았다.

### 5.3. Huber Loss

![Untitled](STS_NLP_%E1%84%90%E1%85%B5%E1%86%B7%20%E1%84%85%E1%85%B5%E1%84%91%E1%85%A9%E1%84%90%E1%85%B3(04)%204046bfdb5ce24724afbb40fd838c68eb/Untitled%204.png)

- 모든 지점에서 미분 가능하며 이상치에 강건하다.
- L1 Loss와 L2 Loss의 단점을 서로 보완한 Loss function
    - $error < \delta$ : L2 Loss와 같은 형태로, 미분 가능하다.
    - $error > \delta$ : L1 Loss와 비슷하게 Linear하여, error가 클수록 quadratic하게 Loss 가 커지는 것을 방지한다.

### 5.4. PLCC Loss

이번 프로젝트의 평가 지표인 pearson correlation을 loss function으로 직접 활용하기 위해 PLCC loss를 도입하였다.

$$
PLCC\space Loss(y,\hat{y}) = 1-corr(y,\hat{y})
$$

```python
class PLCCLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output, target):
        vx = output - torch.mean(output)
        vy = target - torch.mean(target)

        corr = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
        return 1 - corr**2
```

PLCC의 경우, 학습 초기부터 사용하는 경우 pearson correlation이 -1로 수렴할 가능성이 있기 때문에, 어느정도 학습이 안정화 된 이후부터 사용했다.

![W&B Chart 4_21_2023, 9_41_27 AM.png](STS_NLP_%E1%84%90%E1%85%B5%E1%86%B7%20%E1%84%85%E1%85%B5%E1%84%91%E1%85%A9%E1%84%90%E1%85%B3(04)%204046bfdb5ce24724afbb40fd838c68eb/WB_Chart_4_21_2023_9_41_27_AM.png)

![W&B Chart 4_21_2023, 9_49_46 AM.png](STS_NLP_%E1%84%90%E1%85%B5%E1%86%B7%20%E1%84%85%E1%85%B5%E1%84%91%E1%85%A9%E1%84%90%E1%85%B3(04)%204046bfdb5ce24724afbb40fd838c68eb/WB_Chart_4_21_2023_9_49_46_AM.png)

- 파란색 실선 : Huber loss를 이용하여 먼저 fine tuning
    - →초록색 실선 : PLCC Loss를 이용하여 이어서 fine tuning
    - →빨간색 실선 : 다시 Huber Loss를 이용하여 fine tuning

결론적으로, 위의 두 실험에서는 PLCC를 이용하는 것이 다른 Loss를 사용하는 것보다 validation 성능은 더 높게 나왔지만, 제출 횟수 제한으로 인하여 일반적으로 더 나은 성능을 보여주는지 확인할 수 없었다.

## 6. 기타 Regularization

- **EarlyStopping 사용**

wandb에서 다른 하이퍼파라미터 조합에 대한 실험을 더 효율적으로 진행하기 위해 일정 epoch 이상 validation pearson correlation 개선이 없으면 성능개선이 없는 것으로 판단하고 학습을 종료하도록 했다.

- **Weight Decay (L2 regularization)**
    
    $$
    Loss_{new}(w) = Loss_{old}(w) + \lambda w^T w 
    $$
    

![pearson correlation에 크게 영향을 주지 않으면서 사용할 수 있는 weight decay 값은 1e-3임을 실험적으로 확인 후 적용하였다.](STS_NLP_%E1%84%90%E1%85%B5%E1%86%B7%20%E1%84%85%E1%85%B5%E1%84%91%E1%85%A9%E1%84%90%E1%85%B3(04)%204046bfdb5ce24724afbb40fd838c68eb/WB_Chart_4_21_2023_9_36_35_AM.png)

pearson correlation에 크게 영향을 주지 않으면서 사용할 수 있는 weight decay 값은 1e-3임을 실험적으로 확인 후 적용하였다.

Weight decay는 일반적으로 overfitting을 방지하기 위해 사용된다. 먼저 weight decay를 사용했을 때 성능 저하가 발생하는지 sweep을 통해 확인한 뒤, validation set에 대해 성능저하가 발생하지 않는 선에서 (1e-3) 사용했다. 하지만, weight decay를 적용하더라도 test set에서 pearson correlation 성능저하가 0.02 내외였다. 이는 weight decay가 적용되지 않은 모델보다 일반화 성능이 더 좋다고 할 수 없는 수치였다.

## 7. 출력층 MLP

- 일반적으로 [CLS]토큰의 Classifier head는 MLP 구조라고 알고있었다.
- 하지만, hugging face에서 pre-trained 모델을 불러올 때 num_labels=1로 설정하면 1개의 FC layer만이 자동으로 설정되었다.
- model_size가 768인데, 768차원을 1차원으로 바로 projection 시키는 것에 보다, MLP를 통해 차원을 점차 줄여나가는 것이 효과적일 것이라 생각했다.
- 따라서 3층 MLP 구조를 구현해보았다.
    - 시도1 : 768 → 256 → 64 → 1
        - Classifier head에 parameter 수가 많고, pre_trained model의 parameter도 학습 그런지 학습이 정상적으로 이루어지지 않았다.
        - 따라서 train loss가 크게 줄지 않았다.
    - 시도2 : 768 → 256 → 64 → 1 + Pre-trained model은 freeze
        - 이 역시 정상적으로 학습이 어려웠다..
        - classifier head만 fine-tuning 하는 것은 표현력이 조금 부족한 것인가 하는 생각이 들었다.
    - 시도3 : 768 → 256 → 64 → 1 + Pre-trained model은 10층까지만 freeze
        - 이 정도면 학습이 잘 될것이라 생각했는데 역시 학습이 안됐다.
        - 학습이 잘 되기 위해서 Batch normalization을 사용해보았다.
    - 시도4 : 768 → 256 → 64 → 1 + Pre-trained model은 10층까지만 freeze + Batch norm
        - 역시 학습이 잘 안되었다. Batch norm으로 인해 학습되어야 할 parameter가 더 늘어서 그런 것인가 생각을 했다.
- 이 외에도 여러 Xavier initialization 등을 적용해보았지만 학습이 잘 이루어지지 않았다. 왜 학습이 잘 되지 않는 것인지, 왜 768차원을 1차원으로 바로 Projection하는 것이 성능이 더 좋은지 추후에 공부해봐야할 것 같다.

## 8. Ensemble

- 학습 결과가 우수한 모델들의 inference 결과들에 weighted average를 취하여 더 높은 성능을 얻을 수 있었다.
- **결과적으로 단일 모델만 사용하였을 때 우수한 성능을 보여준 ‘KR-ELECTRA-discriminator’와 ‘monologg/koelectra-base-v3-discriminator’ 모델을 ‘Weighted averaging’ 기법을 통해 블렌딩하였다.**

---

# 프로젝트 수행 결과

![Public 0.9216 → Private 0.9337 (1등 : 0.9420)](STS_NLP_%E1%84%90%E1%85%B5%E1%86%B7%20%E1%84%85%E1%85%B5%E1%84%91%E1%85%A9%E1%84%90%E1%85%B3(04)%204046bfdb5ce24724afbb40fd838c68eb/Untitled%205.png)

Public 0.9216 → Private 0.9337 (1등 : 0.9420)

Data Augmentation, Data imbalance, Model selection / 크게는 데이터와 모델을 기준으로 다양한 모델을 실험했다. wandb 및 CSVLogger를 기반으로 최적의 파라미터를 서치하여 최종 모델에 반영했다.

Sentence Switching을 통한 Data Augmentation으로 성능 향상 효과를 보았으나, 성능이 좋을 것이라고 예상했던 Back translation에서 오히려 성능이 떨어졌다. 그 원인을 부정확한 구글 API 번역 성능으로 파악하였지만 시간상 다른 모델을 다시 불러오기 어렵다고 판단하여 back translation은 적용하지 않기로 결정하였다.

Data imbalance 문제를 label 0을 undersampling 하는 방식과 undersampling한 비율만큼 label 5를 생성하는 방식을 적용하여 해결하려 했으나 Data imbalance 문제를 완벽히 해결하진 못했다. 

이밖에 Loss function을 다양하게 적용해보는 방식을 적용해보았으나 validation pearson 값은 높게 나오나 정작 test pearson 값은 낮게 나오는 문제가 발생했다.

데이터 부족 문제와 불균형 문제를 잘 해결하는 것이 대회의 key point라고 생각하여 다양한 실험 설계를 진행했으나 문제를 완벽히 해결하진 못했다. 
결과적으로 ‘KR-ELECTRA-discriminator’와 ‘monologg/koelectra-base-v3-discriminator’ 모델을 ‘Weighted averaging’ 기법을 통해 블렌딩하여, 최종 순위인 7위로 결과를 마무리 지었다.

---

# 자체 평가 의견

**잘했던 점**

- 다양한 아이디어들을 문제해결에 적용해 볼 수 있었다.
- 깃허브를 활용하여 협업 경험을 하였다.
- 팀원 간 의견 공유가 활발하였고, 역할 분담이 잘 이루어졌다.

**시도했으나 잘 되지 않았던 것들**

- Classifier head를 MLP 구조로 설계했을 때 성능이 좋지 않았다.
- 구글 API를 통한 back translation 결과가 매끄럽지 못했다.
- 일반적으로 많이 사용하는 L2 Loss를 사용해봤으나 성능이 좋지 않았다.

**아쉬웠던 점들**

- EDA, Back translation과 같은 데이터 증강 방법의 성능이 좋지 않았다.
- 데이터가 많은 label은 loss에 대한 가중치를 줄여서 학습시켜보고 싶었는데, 시도해보지 못했다.
- 더 다양한 learning rate scheduler를 사용해보고자 `CyclicLR`, `CosineAnnealingLR` 등을 도입하고자 했지만, learning rate shceduling의 효과를 내기 위해서는 훨씬 많은 epoch를 두고 실험해야 해서 충분히 적용해보지 못했다.
- 실험 환경 셋팅과 실험 수행이 병렬적으로 이루어지다 보니, 처음부터 하나의 환경에서 실험 관리가 이루어지지 못해 결과를 종합하는 데 많은 에너지가 소요되었다. 이번 경험을 바탕으로 다음에 프로젝트를 진행할 때에는 초기에 실험 환경 셋팅을 어떻게 해야 할지 생각해 볼 수 있었다.

**프로젝트를 통해 배운 점 또는 시사점**

- 이론적으로 성능을 높여줄 수 있는 방법들을 적용해 보았으나, 항상 성능을 높여주지는 않는다는 것을 알게 되었다.
- Wandb와 같은 실험 관리 툴들을 자유자재로 사용하는 능력이 중요함을 깨달았다.
- 336M 모델을 학습시키는 것도 힘든데 100B 이상의 모델들은 도대체 어떻게 학습을 시켰을까? 딥러닝 모델 학습에 컴퓨팅 파워가 얼마나 중요한지 다시 한번 깨달을 수 있었다.
- 결국 가장 중요한 것은 우리가 가진 데이터를 분석하고 가다듬는 EDA, 전처리 과정이라는 것을 깨달았다.