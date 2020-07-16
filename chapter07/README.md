# chapter07. RNN을 사용한 문장 생성

5장과 6장에서 RNN과 LSTM의 구조와 구현을 자세하게 살펴봤습니다. 바야흐로 우리는 이 개념들을 구현 수준에서 이해하게 된 것입니다. 이번 장에서는 지금까지의 성과, RNN과 LSTM이 꽃을 피웁니다. LSTM을 이용해 재미있는 애플리케이션을 구현해볼 것이기 때문입니다.

이번 장에서는 언어 모델을 사용해 '문장 생성'을 수행합니다. 구체적으로는 우선 말뭉치를 사용해 학습한 언어 모델을 이용하여 새로운 문장을 만들어냅니다. 그런 다음 개선된 언어 모델을 이용하여 더 자연스러운 문장을 생성하는 모습을 선보이겠습니다. 여기까지 해보면 'AI로 글을 쓰게 한다'라는 개념을 간단하게라도 실감할 수 있을 것입니다.

여기서 멈추지 않고 seq2seq라는 새로운 구조의 신경망도 다룹니다. seq2seq란 'from sequence to sequence', 즉 '시계열에서 시계열로'를 뜻하는 말로, 한 시계열 데이터를 다른 시계열 데이터로 변환하는 걸 말압니다. 이번 장에서는 RNN 두 개를 연결하는 아주 간단한 방법으로 seq2seq를 구현해볼 것입니다. seq2seq는 기계 번역, 챗봇, 메일의 자동 답신 등 다양하게 응용될 수 있습니다. 간단하면서 영리하고 강력한 seq2seq를 이해하고 나면 딥러닝의 가능성이 더욱 크게 느껴질 것입니다!

## 7.1 언어 모델을 사용한 문장 생성

지금까지 여러 장에 걸쳐서 언어 모델을 다뤄왔습니다. 다시 말하지만, 언어 모델은 다양한 애플리케이션에서 활용할 수 있습니다. 대표적인 예로는 기계 번역, 음성 인식, 문장 생성 등이 있습니다. 이번 절에서는 언어 모델로 문장을 생성해보려 합니다.

### 7.1.1 RNN을 사용한 문장 생성의 순서

앞 장에서는 LSTM 계층을 이용하여 언어 모델을 구현했는데, 그 모델의 신경망 구성은 [그림 7-1]처럼 생겼었습니다. 그리고 시계열 데이터를 T개분 만큼 모아 처리하는 Time LSTM과 Time Affine 계층 등을 만들었습니다.

<img src="README.assets/fig 7-1.png" alt="fig 7-1" style="zoom:50%;" />

이제 언어 모델에게 문장을 생성시키는 순서를 설명해보겠습니다. 이번에도 친숙한 'you say goodbye and I say hello.'라는 말뭉치로 학습한 언어 모델을 예로 생각하겠습니다. 이 학습된 언어 모델에 'I'라는 단어를 입력으로 주면 언어 모델은 [그림 7-2]와 같은확률 분포를 출력합니다.

<img src="README.assets/fig 7-2.png" alt="fig 7-2" style="zoom:50%;" />

언어 모델은 지금까지 주어진 단어들에서 다음에 출현하는 단어의 확률분포를 출력합니다. [그림 7-2]의 예는 'I'라는 단어를 주었을 때 출력한 확률분포를 보여줍니다. 이 결과를 기초로 다음 단어를 새로 생성하려면 어떻게 해야 할까요?

첫 번째로 확률이 가장 높은 단어를 선택하는 방법을 떠올릴 수 있을 것입니다. 확률이 가장 높은 단어를 선택할 뿐이므로 결과가 일정하게 정해지는 '결정적'인 방법입니다. 또한, '확률적'으로 선택하는 방법도 생각할 수 있습니다. 각 후보 단어의 확률에 맞게 선택하는 것으로, 확률이 높은 단어는 선택되기 쉽고, 확률이 낮은 단어는 선택되기 어려워집니다. 이 방식에서는 선택되는 단어, 즉 샘플링 단어가 매번 다를 수 있습니다.

만약 매번 다른 문장을 생성하도록 한다면 생성되는 문장이 다양해져서 재밌을 것입니다. 그래서 후자의 방법, 즉 확률적으로 선택하는 방법으로 단어를 선택하겠습니다. 예시로 돌아와서 [그림 7-3]과 같이 'say'라는 단어가 확률적으로 선택되었다고 합시다.

<img src="README.assets/fig 7-3.png" alt="fig 7-3" style="zoom:50%;" />

[그림 7-3]은 확률분포로부터 샘플링을 수행한 결과로 'say'가 선택된 경우를 보여줍니다. 실제로 [그림 7-3]의 확률분포에서는 'say'의 확률이 가장 높기 때문에 'say'가 샘플링될 확률이 가장 높습니다. 다만, 필연적, 즉 '결정적'이지 않고 '확률적'으로 결정된다는 점에 주의합시다. 다른 단어들도 해당 단어의 출현 확률에 따라 정해진 비율만큼 샘플링될 가능성이 있다는 뜻입니다.

```markdown
**NOTE** 결정적이란 알고리즘의 결과가 하나로 정해지는 것, 결과가 예측 가능한 것을 말합니다. 예컨대 앞의 예에서 확률이 가장 높은 단어를 선택하도록 하면, 그것은 '결정적'인 알고리즘입니다. 한편, 확률적인 알고리즘에서는 결과가 확률에 따라 정해집니다. 따라서 선택되는 단어는 실행할 때마다 달라질 수 있습니다.
```

그러면 계속해서 두 번째 단어를 샘플링해봅시다. 이 작업은 앞에서 한 작업을 되풀이하기만하면 됩니다. 즉, 방금 생성한 단어인 'say'를 언어 모델에 입력하여 다음 단어의 확률 분포를 얻습니다. 그런 다음 그 확률분포를 기초로 다음에 출현할 단어를 샘플링하는 것입니다. 이는 [그림 7-4]에 자세히 나와있습니다.

<img src="README.assets/fig 7-4.png" alt="fig 7-4" style="zoom:50%;" />

다음은 이 작업을 원하는 만큼 반복거나 <eos> 같은 종결 기호가 나타날 때까지 반복합니다. 그러면 새로운 문장을 생성할 수 있습니다.

여기에서 주목할 것은 이렇게 생성한 문장은 훈련 데이터에는 존재하지 않는, 말 그래도 새로 생성된 문장이라는 것입니다. 왜냐하면 언어 모델은 훈련 데이터를 암기한 것이 아니라, 훈련 데이터에서 사용된 단어의 정렬 패턴을 학습한 것이기 때문이죠. 만약 언어 모델이 말뭉치로부터 단어의 출현 패턴을 올바르게 학습할 수 있다면, 그 모델이 새로 생성하는 문장은 우리 인간에게도 자연스럽고 의미가 통하는 문장이 될 것으로 기대할 수 있습니다.

### 7.1.2 문장 생성 구현

그럼 문장을 생성하는 코드를 구현해보겠습니다. 앞 장에서 구현한 RNNLM 클래스(chapter06/rnnlm.py)를 상속해 RNNLMGen 클래스를 만들고, 이 클래스에 문장 생성 메서드를 추가하겠습니다.

```markdown
**NOTE** 클래스 상속이란 기존 클래스를 게승하여 새로운 클래스를 만드는 매커니즘입니다. Python에서 클래스를 상속하는 예시는 Base 클래스를 상속하는 경우, 새로 정의할 클래스 이름이 NewClass라면 `class NewClass(Base)`라고 작성하면 됩니다.
```

RNNLMGen 클래스의 구현은 rnnlm_gen.py를 확인하시면 됩니다.

```python
import numpy as np
from rnnlm import RNNLM
from better_rnnlm import BetterRNNLM
from commons.functions import softmax


class RNNLMGen(RNNLM):
    def generate(self, start_id, skip_ids=None, sample_size=100):
        word_ids = [start_id]

        x = start_id
        while len(word_ids) < sample_size:
            x = np.array(x).reshape(1, 1)
            score = self.predict(x)
            p = softmax(score.flatten())

            sampled = np.random.choice(len(p), size=1, p=p)
            if (skip_ids is None) or (sampled not in skip_ids):
                x = sampled
                word_ids.append(int(x))

        return word_ids

    def get_state(self):
        return self.lstm_layer.h, self.lstm_layer.c

    def set_state(self, state):
        self.lstm_layer.set_state(*state)
```

RNNLMGen 클래스에서 문장 생성을 수행하는 메서드는 바로 `generate(start_id, skip_ids, sample_size)`입니다. 인수 중 start_id는 최초로 주는 단어의 ID, sample_size는 샘플링하는 단어의 수를 말합니다. 그리고 skip_ids는 단어 ID의 리스트로 여기에 속하는 단어 ID는 샘플링되지 않도록 하는데 PTB 데이터셋에 있는 `<unk>`나 N 등 전처리된 단어를 샘플링하지 않게 사용하는 용도로 사용합니다.

```markdown
**WARNING** PTB 데이터셋은 원래 문장들에 이미 전처리를 해둔 것으로, 희소 단어는 `<unk>`로 숫자는 N으로 대체해놨습니다. 참고로, 이 책의 예제에서는 각 문장을 구분하는 데 `<eos>`라는 문자열을 사용합니다.
```

`generate()` 메서드는 가장 먼저 `model.predict(x)`를 호출해 각 단어의 점수를 출력하는데 이 값은 정규화되기 전의 값입니다. 그리고 `p = softmax(score)` 코드에서는 이 점수들을 소프트맥스 함수를 이용해 정규화합니다. 이것으로 목표로 하는 확률분포 p를 얻을 수 있습니다. 그런 다음 확률분포 p로부터 다음 단어를 샘플링합니다. 참고로, 확률분포로부터 샘플링할 때는 `np.random.choice()`를 사용합니다. 이 함수의 사용법은 '4.2.6 네거티브 샘플링의 샘플링 기법'에서 설명했습니다.

```markdown
**WARNING** model의 `predict()` 메서드는 미니배치 처리를 하므로 입력 x는 2차원 배열이어야 합니다. 그래서 단어 ID를 하나만 입력하더라도 미니배치 크기를 1로 간주해 1 by 1 numpy 배열로 성형, reshape합니다.
```

RNNLMGen 클래스를 사용해 문장을 생성해보겠습니다. 이번에는 아무런 학습도 수행하지 않은 상태, 즉 가중치 매개변수는 무작위 초깃값인 상태에서 문장을 생성합니다. 문장 생성을 위한 코드는 generate_text.py와 같습니다.

```python
from datasets import ptb
from rnnlm_gen import RNNLMGen

corpus, word_to_id, id_to_word = ptb.load_data('train')
vocab_size = len(word_to_id)
corpus_size = len(corpus)

model = RNNLMGen()
# model.load_params('./chapter06/RNNLM.pkl')

start_word = 'you'
start_id = word_to_id[start_word]
skip_words = ['N', '<unk>', '$']
skip_ids = [word_to_id[w] for w in skip_words]

word_ids = model.generate(start_id, skip_ids)
txt = ' '.join([id_to_word[i] for i in word_ids])
txt = txt.replace(' <eos>', '.\n')
print(txt)
```

첫 단어를 'you'로 하고, 그 단어 ID를 start_id로 설정한 다음 문장을 생성합니다. 샘플링하지 않을 단어로는 `['N', '<unk>', '$']`를 지정했습니다. 참고로, 문장을 생성하는 `generate()` 메서드는 단어 ID들을 배열 형태로 반환합니다. 그래서 단어 ID 배열을 문장으로 변환해야 하는데, `txt = ''.join([id_to_word[i] for i in word_ids])` 코드가 그 일을 담당합니다. `join()` 메서드는 `seperator.join(iterator)` 형태로 작성하며, iterator의 단어들 사이에 구분자를 삽입해 모두 연결합니다.

위 코드를 실행하면 단어들을 엉터리로 나열한 글이 출력됩니다. 당연하게도, 모델의 가중치 초깃값이 무작위한 값을 사용했기 때문에 의미가 통하지 않는 문장이 출력된 것입니다. 그렇다면 학습을 수행한 언어 모델은 어떻게 다를까요? 바로 이어서, 앞 장에서 학습을 끝낸 가중치를 이용해 문장을 생성해보겠습니다. 위 코드에서 주석 처리된 `model.load_params('./chapter06/RNNLM.pkl')`을 주석 해제하면 앞 장에서 학습한 가중치 매개변수를 사용합니다. 그 상태에서 문장을 생성하면 됩니다.

문법적으로 이상하거나 의미가 통하지 않는 문장이 섞여 있지만, 그럴듯한 문장도 있는 결과가 출력됩니다. 주어와 동사를 짝지어 올바른 순서로 배치한 문장들도 있으며 형용사와 명사의 사용법도 어느 정도 이해하는 문장도 있습니다. 또한, 일부 문장은 의미적으로도 올바른 문장입니다.

이처럼 두 번째 시도로 생성한 문장은 어느 정도는 올바른 문장이라고 할 수 있을 것입니다. 하지만 부자연스러운 문장도 발견되니, 아직 개선할 여지가 있습니다. '완벽한 문장'은 존재하지 않지만, 더 자연스러운 문장이 필요합니다. 그러기 위해서는 더 나은 언어 모델을 사용하면 됩니다.
