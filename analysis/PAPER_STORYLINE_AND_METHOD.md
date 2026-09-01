# CTC 구간 좌표와 계층적 Dark Knowledge 증류

> 작업 제목: **Occurrence-aware Hierarchical Dark Knowledge Distillation for CTC ASR**  
> 현재 구현명: **Mass3 + NTDK**  
> 확정 설정: `delta=6`, `M=32`, `lambda_mass=25`, `lambda_NTDK=8`  
> 상태: 메소드와 메인 비교 프로토콜은 동결. 정식 이름과 다중 seed 통계는 추후 확정.

## 0. 한 문장 요약

CTC teacher의 blank-suppressed forward-backward occupancy는 **토큰 occurrence가 존재할 시간 구간(WHERE)**만 정하고, 실제 지식은 원래 teacher posterior에서 읽는다. 그 구간 posterior를 **blank/정답/non-target 총질량(Mass3)**과 **정답이 아닌 클래스들의 조건부 상대분포(NTDK)**로 분리해 student에게 각각 증류한다.

핵심은 “posterior를 억지로 부드럽게 만들어 dark knowledge를 생성”하는 것이 아니다. 전체 분포에서는 정답 질량에 가려졌지만, 정답과 blank를 조건부로 제거하면 이미 존재하는 안정적인 non-target 관계를 독립적인 gradient로 노출한다.

---

## 1. 문제 정의와 논문 스토리라인

### 1.1 CTC가 KD에 어려운 이유

입력 음성 프레임을 $X=(x_1,\ldots,x_T)$, 정답 BPE 열을 $Y=(y_1,\ldots,y_U)$라 하자. CTC는 blank $\varnothing$를 포함한 모든 유효 경로 $\pi$를 합산한다.

$$
P(Y\mid X)=\sum_{\pi:\mathcal B(\pi)=Y}\prod_{t=1}^{T}p(\pi_t\mid X).
$$

이 목적함수는 transcript에는 민감하지만, 특정 토큰을 어느 프레임에서 방출할지는 유일하게 정하지 않는다. 따라서 같은 transcript를 맞히는 teacher와 student도 서로 다른 프레임에 매우 뾰족한 non-blank spike를 둘 수 있다.

일반적인 frame-level KD는 같은 프레임의 teacher와 student posterior를 직접 비교한다.

$$
\mathcal L_{\mathrm{frameKD}}=\sum_t D_{\mathrm{KL}}\!\left(p_t^T\|p_t^S\right).
$$

CTC에서는 여기서 두 문제가 동시에 생긴다.

1. **시간 정렬 불일치:** 같은 토큰을 맞히면서도 teacher와 student spike가 한두 프레임 어긋날 수 있다. 그러면 의미적으로 같은 예측에 잘못된 KD gradient가 간다.
2. **blank 및 정답 spike의 지배:** teacher posterior의 대부분은 blank이고, emission 구간에서도 정답 하나가 거의 모든 질량을 가진다. 전체 KL이나 구간 평균만으로는 작은 non-target 질량 안의 클래스 관계가 거의 학습되지 않는다.

우리의 진단에서 no-KD student의 토큰 중 56.6%가 teacher와 다른 프레임에 spike를 두었고, 그 차이가 난 사례의 82%는 정확히 $\pm1$ frame이었다. 한편 occupancy로 구간 평균한 raw teacher posterior도 median GT mass가 0.99953, median effective class count가 1.0049로 사실상 one-hot이었다. 즉 **정렬을 해결하는 것만으로 dark knowledge가 자동으로 전달되지는 않는다.**

### 1.2 핵심 가설

우리는 CTC teacher가 주는 정보를 두 역할로 나눈다.

- **WHERE:** transcript-constrained FB occupancy가 각 token occurrence의 시간 좌표를 준다.
- **WHAT:** 원래 teacher posterior가 그 좌표 안의 정답, blank, 혼동 후보 관계를 준다.

그리고 WHAT을 다시 두 층으로 분해한다.

- **얼마나:** blank, occurrence GT, 모든 non-target에 각각 얼마의 총확률 질량을 둘 것인가.
- **무엇끼리:** non-target 질량 안에서 어떤 BPE들이 서로 더 그럴듯한가.

이 분리는 정렬 문제와 dark-knowledge 가림 문제를 하나의 CTC-specific 계층 구조로 연결한다.

### 1.3 주장할 contribution

1. **Occurrence-aware temporal coordinate:** blank-suppressed transcript-constrained FB를 의미 target이 아니라 occurrence별 pooling coordinate로 사용한다.
2. **CTC-specific hierarchical decoupling:** pooled posterior를 $[\text{blank},\text{GT},\text{non-target}]$ 질량과 conditional non-target 관계로 나누어 각각 증류한다.
3. **Dark knowledge without artificial softening:** temperature나 logit 표준화로 새로운 불확실성을 만들지 않고, raw $T=1$ posterior에 이미 존재하는 conditional non-target 구조를 독립적으로 학습한다.
4. **Alignment-tolerant student matching:** teacher와 student를 단일 동일 frame에서 비교하지 않고 같은 occurrence support에서 각각 확률 공간 mixture로 pooling한다.
5. **실증:** LibriSpeech-100과 TED-LIUM3-full의 동일 환경에서 기존 CTC KD 및 consistency 방법과 비교하고, alignment coverage와 conditional-tail 진단으로 작동 원리를 검증한다.

“최초”라는 표현은 더 넓은 문헌 검색 후에만 사용한다. 안전하고 정확한 novelty 주장은 **occurrence-level FB pooling과 CTC 3-way mass/conditional non-target decoupling의 결합**이다.

---

## 2. 기존 연구는 무엇을 해결했는가

### 2.1 일반 Knowledge Distillation

[Hinton et al.](https://arxiv.org/abs/1503.02531)은 temperature로 완화한 teacher class distribution을 student가 맞추게 해 정답 외 클래스 관계를 전달했다. 하지만 CTC에서는 blank logit이 매우 크고 spike 위치가 모델마다 다르므로, 같은 temperature와 같은 frame을 전제로 한 KD가 그대로 작동하기 어렵다.

### 2.2 Sequence-level CTC KD: S-CTC

[Huang et al., Interspeech 2018](https://www.isca-archive.org/interspeech_2018/huang18d_interspeech.html)은 frame KD와 sequence training의 불일치를 지적하고, 정답 transcript에 제한된 teacher CTC forward-backward occupancy를 soft target으로 사용하는 S-CTC를 제안했다.

- 해결: teacher의 latent alignment를 단일 best path가 아닌 sequence-level occupancy로 전달한다.
- 한계: soft target의 중심은 transcript state와 blank occupancy다. **정답이 아닌 vocabulary class들 사이의 의미적 상대관계**를 별도 채널로 강화하지 않는다.

### 2.3 Spike alignment: Guided CTC

[Kurata and Audhkhasi, Interspeech 2019](https://www.isca-archive.org/interspeech_2019/kurata19_interspeech.html)은 CTC 모델마다 arbitrary한 spike timing을 갖는 문제를 줄이기 위해 guide model의 non-blank spike 위치를 따라가도록 학습한다.

- 해결: teacher/student spike를 직접 정렬해 framewise 결합이나 KD를 쉽게 한다.
- 한계: student의 emission timing을 guide 위치로 끌어간다. 정렬 차이를 허용하는 occurrence-level comparison이 아니며, pooled non-target class 관계를 명시적으로 분리하지 않는다.

### 2.4 Blank-frame elimination: CARL/KD-BE

[Tian et al., Interspeech 2022](https://www.isca-archive.org/interspeech_2022/tian22_interspeech.html)은 teacher argmax가 blank인 frame을 posterior KD에서 제외하고, representation learning과 posterior KD를 결합했다.

- 해결: 수가 많고 정보가 적을 수 있는 blank frame이 KD를 지배하는 문제를 완화한다.
- 한계: blank frame을 모두 버리면 alignment/emission 정보와 non-blank spike 주변의 유용한 blank shoulder도 함께 잃을 수 있다. 또한 선택 후의 비교는 여전히 frame-indexed다.

### 2.5 Blank는 전부 버려야 하는가: symmetric selection

[Hilmes et al., Interspeech 2025](https://arxiv.org/abs/2506.01503)은 blank frame을 전부 제거하면 유용한 alignment 정보까지 사라진다고 분석하고, teacher non-blank spike 주변의 blank frame을 대칭적으로 포함하는 selection을 제안했다.

- 해결: blank elimination보다 spike 주변 문맥을 보존한다.
- 한계: 고정된 $n$-frame dilation은 transcript occurrence의 단조 경로 확률을 사용하지 않는 시간 heuristic이다. 무엇을 전달할지에 대해서는 전체 posterior KD에 의존하며, non-target semantics를 독립적으로 증류하지 않는다.

### 2.6 Alignment mismatch와 blank 유지

[Kim et al., Interspeech 2024](https://arxiv.org/abs/2406.07909)은 CTC KD 실패의 핵심을 단순한 spike sparsity보다 teacher/student alignment disagreement에서 찾고, alignment가 맞으면 blank를 유지하는 편이 유리할 수 있음을 보였다.

이 결과는 우리와 충돌하지 않는다. 우리는 blank를 target에서 제거하지 않는다. blank suppression은 오직 **FB를 위한 WHERE posterior**에만 적용하고, Mass3는 raw posterior의 blank 질량을 그대로 증류한다. 즉 정렬 좌표를 넓히기 위한 blank 억제와 의미 target에서 blank를 삭제하는 것은 다른 연산이다.

### 2.7 Consistency regularization: CR-CTC

[CR-CTC, ICLR 2025](https://arxiv.org/abs/2410.05101)은 같은 발화의 두 augmented view에 CTC를 적용하고, 두 posterior 간 stop-gradient symmetric consistency loss를 사용해 CTC peakiness를 낮춘다.

- 해결: 외부 teacher 없이 augmentation 간 일관성으로 더 부드럽고 강건한 representation을 학습한다.
- 한계: 두 view의 forward가 필요하며, 특정 external teacher가 가진 occurrence-level confusion relation을 직접 전달하는 방법은 아니다. 우리와 경쟁 방법인 동시에 결합 가능한 orthogonal 축이지만, 현재 논문에서는 메소드를 깔끔하게 유지하기 위해 결합하지 않는다.

### 2.8 Decoupled KD

[DKD, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Zhao_Decoupled_Knowledge_Distillation_CVPR_2022_paper.html)는 classification KD를 target-class knowledge와 non-target class knowledge로 분리해, teacher가 정답에 확신할 때 non-target gradient가 사라지는 문제를 해결했다.

- 우리가 계승한 원리: 작은 non-target 총질량과 그 내부 상대분포를 분리한다.
- CTC에서 그대로 쓸 수 없는 이유: CTC의 blank는 일반 오답 class가 아니라 emission timing과 path allocation을 담당하고, 같은 BPE id도 transcript 안에서 여러 occurrence를 가질 수 있다.
- 우리의 차이: 먼저 FB로 occurrence 좌표를 만든 뒤, $[\text{blank},\text{occurrence GT},\text{NT}]$와 conditional NT로 분해한다.

### 2.9 Factorized/progressive CTC KD

[Tian et al., Speech Communication 2024](https://doi.org/10.1016/j.specom.2024.103071)는 blank와 non-blank frame의 KD를 분리하고 단계적으로 학습해 CTC posterior의 불균형을 다룬다.

- 해결: blank/non-blank가 학습에 미치는 서로 다른 역할을 loss와 stage 수준에서 분리한다.
- 한계: 기본 비교 단위는 frame이며, transcript occurrence별 support에서 pooled GT mass와 conditional non-target vocabulary relation을 분리하는 구조는 아니다.

### 2.10 가장 가까운 연구들과의 차이

| 방법 | 시간 문제 처리 | blank 처리 | non-target 관계 | 비교 단위 | 우리의 차이 |
|---|---|---|---|---|---|
| Vanilla KD | 없음 | 전체 포함 | 전체 KL에 묻힘 | frame | occurrence support + 독립 NTDK |
| Guided CTC | spike 위치 강제 | guide non-blank 중심 | 없음 | selected frame | 정렬 강제 대신 구간 허용 |
| KD-BE | blank frame 삭제 | 제거 | 선택 frame posterior | frame | raw blank 질량 보존 |
| Symmetric selection | spike 주변 고정 dilation | 일부 보존 | 전체 KL | frame neighborhood | transcript-constrained occurrence FB |
| S-CTC | teacher FB occupancy | state occupancy로 포함 | 명시적 분리 없음 | frame/state | FB는 WHERE, raw vocabulary posterior는 WHAT |
| Factorized CTC KD | blank/non-blank 분기 | 별도 loss | 주로 frame posterior | frame/stage | occurrence pooling + GT를 분리한 3-way mass + conditional NT |
| CR-CTC | augmentation consistency | 간접 de-peaking | external teacher relation 없음 | paired frame | teacher의 semantic confusion 전달 |
| DKD | 해당 없음 | 일반 non-target 취급 | target/NT 분리 | sample | blank를 별도 구조로 둔 CTC occurrence 분해 |

---

## 3. 제안 방법

### 3.1 표기

- $\mathcal V$: BPE vocabulary, 크기 1024
- $\bar{\mathcal V}=\mathcal V\cup\{\varnothing\}$: CTC 출력 집합
- $p_t^T(k)$, $p_t^S(k)$: teacher/student의 raw $T=1$ posterior
- $y_u$: transcript의 $u$번째 **occurrence**. 같은 BPE id가 반복되어도 서로 다른 $u$다.
- $\delta=6$: WHERE 계산에만 쓰는 teacher blank penalty

### 3.2 Step 1 — blank-suppressed posterior로 WHERE 계산

Teacher raw posterior에서 blank만 log-domain penalty를 주고 다시 정규화한다.

$$
\widetilde p_t^T(k)=
\frac{p_t^T(k)\exp[-\delta\,\mathbf 1(k=\varnothing)]}
{\sum_j p_t^T(j)\exp[-\delta\,\mathbf 1(j=\varnothing)]}.
$$

이 $\widetilde p^T$와 정답 transcript로 CTC forward-backward를 수행해 occurrence occupancy를 얻는다.

$$
\gamma_{t,u}=P(\text{CTC path is at occurrence }u\text{ at }t\mid X,Y;\widetilde p^T).
$$

중요한 해석은 다음과 같다.

- blank suppression은 새로운 semantic target을 만들기 위한 것이 아니다.
- FB는 단순히 non-blank posterior를 넓히는 것과 다르다. transcript 순서, 반복 토큰, 가능한 CTC path를 사용해 각 frame의 질량을 특정 occurrence에 할당한다.
- $\delta=6$은 spike를 무한히 넓히는 값이 아니라, 실제 mismatch의 대부분인 $\pm1$ frame shoulder를 복원하는 선택이다.

Occurrence별 teacher pooling weight는

$$
a^T_{u,t}=\frac{\gamma_{t,u}}{\sum_{t'}\gamma_{t',u}}
$$

로 정규화한다.

### 3.3 Step 2 — raw posterior에서 WHAT을 확률 mixture로 pooling

Teacher semantic target은 blank-suppressed posterior가 아니라 원래 raw posterior에서 읽는다.

$$
q_u^T(k)=\sum_t a^T_{u,t}p_t^T(k).
$$

Student frame 수가 teacher와 다를 수 있으므로 teacher occupancy를 student time grid로 resample하여 $a^S_{u,t}$를 만들고,

$$
q_u^S(k)=\sum_t a^S_{u,t}p_t^S(k)
$$

를 계산한다. 현재 구현은 각 student cell의 1/4, 3/4 지점에 대응하는 teacher support 두 값을 평균한 뒤 occurrence별로 재정규화한다. student 자신의 FB occupancy는 사용하지 않는다.

이 결합은 반드시 **확률 공간의 arithmetic mixture**다. log-probability를 가중평균하면 확률 공간에서 product of experts가 되어 여러 frame의 교집합만 남고 가장 peaky한 성분이 지배한다. 우리는 구간 안의 대안성을 보존하기 위해 mixture를 쓴다.

### 3.4 Step 3 — Mass3: blank/GT/non-target 총질량

각 occurrence $u$에 대해 pooled posterior를 세 범주로 축약한다.

$$
m_u^T=
\begin{bmatrix}
q_u^T(\varnothing)\\
q_u^T(y_u)\\
1-q_u^T(\varnothing)-q_u^T(y_u)
\end{bmatrix},\qquad
m_u^S=
\begin{bmatrix}
q_u^S(\varnothing)\\
q_u^S(y_u)\\
1-q_u^S(\varnothing)-q_u^S(y_u)
\end{bmatrix}.
$$

Mass3 loss는 occurrence 평균 KL이다.

$$
\mathcal L_{\mathrm{Mass3}}=\frac{1}{U}\sum_{u=1}^{U}
D_{\mathrm{KL}}(m_u^T\|m_u^S).
$$

이 loss가 가르치는 것은 “정답을 무조건 1로 만들어라”가 아니다. teacher가 그 occurrence 구간에서

- 얼마를 blank로 남겼는지,
- 얼마를 정답 occurrence에 주었는지,
- 얼마를 혼동 가능성으로 남겼는지

를 student가 맞추게 한다.

#### 왜 GT/NT 두 개가 아니라 세 개여야 하는가

일반 DKD라면 GT와 non-target만 나누면 된다. 하지만 CTC blank는 단순 오답이 아니다. blank 질량은 해당 구간에서 **언제 방출하고 언제 기다릴지**를 나타내는 path/timing 변수다. GT/NT만 남기고 blank를 조건부 제거하면 이 emission calibration은 CTC loss에만 떠넘겨진다.

Mass3는 정답 의미와 시간적 여유를 동시에 보존하되, blank를 vocabulary confusion과 섞지 않는다. 이 점이 CTC-specific한 이유다.

### 3.5 Step 4 — NTDK: conditional non-target 관계

정답과 blank를 제외한 집합을

$$
\mathcal N_u=\bar{\mathcal V}\setminus\{\varnothing,y_u\}
$$

라 하자. Conditional non-target distribution은

$$
r_u^T(k)=\frac{q_u^T(k)}{1-q_u^T(\varnothing)-q_u^T(y_u)},\qquad k\in\mathcal N_u
$$

이고 student도 동일하게 $r_u^S$를 만든다.

Teacher가 정답에 0.999를 주면 non-target 총질량은 매우 작다. 전체 KL에서는 그 구조의 gradient도 자동으로 작아진다. NTDK는 그 작은 질량을 1로 조건부 정규화하고 독립 weight를 주므로, “정답이 아닐 경우 어떤 BPE가 더 그럴듯한가”를 직접 학습한다.

#### Top-32 + tail 압축

전체 1022개 non-target class를 모두 저장하지 않고 teacher $r_u^T$의 상위 $M=32$ id와 확률을 저장한다. 나머지는 하나의 tail bucket으로 합친다.

$$
\widehat r_u^T=
[r_u^T(k_1),\ldots,r_u^T(k_M),\sum_{k\notin\{k_1,\ldots,k_M\}}r_u^T(k)].
$$

Student도 teacher가 선택한 같은 id의 확률과 잔여 총질량으로 $\widehat r_u^S$를 만든다.

$$
\mathcal L_{\mathrm{NTDK}}=\frac{1}{U}\sum_{u=1}^{U}
D_{\mathrm{KL}}(\widehat r_u^T\|\widehat r_u^S).
$$

이것은 일반적인 “top-k만 남기고 재정규화”와 다르다. 선택되지 않은 질량을 버리지 않고 tail 총량으로 유지한다. 다만 tail 내부 클래스 사이의 관계는 압축되므로 **완전 무손실이라고 주장하지 않는다.** 진단에서 raw conditional NT의 top-32 coverage는 mean 97.65%, median 98.81%였기 때문에 저장량과 보존율의 타협으로 $M=32$를 택했다.

### 3.6 최종 목적함수

$$
\boxed{
\mathcal L
=\mathcal L_{\mathrm{CTC}}
+25\,\mathcal L_{\mathrm{Mass3}}
+8\,\mathcal L_{\mathrm{NTDK}}
}
$$

두 KD loss는 occurrence 단위 평균이며, 최종 설정에는 별도의 reliability gate나 adaptive weight가 없다. target 생성 온도는 $T=1$이다.

### 3.7 실제 학습 파이프라인

1. Teacher raw frame posterior를 한 번 생성하고 cache한다.
2. raw posterior의 복사본에만 blank penalty $\delta=6$을 적용한다.
3. transcript-constrained FB로 occurrence occupancy $\gamma_{t,u}$를 계산한다.
4. 원래 raw posterior를 occupancy-weighted arithmetic mean하여 $q_u^T$를 만든다.
5. $q_u^T$에서 Mass3와 conditional NTDK top-32+tail을 저장한다.
6. 학습 중 student posterior를 같은 teacher occurrence support로 pooling하여 $q_u^S$를 만든다.
7. CTC, Mass3 KL, NTDK KL을 동시에 최적화한다.
8. 추론 시에는 평범한 student CTC 모델만 사용한다. teacher, FB, target cache가 필요 없으므로 inference overhead는 없다.

### 3.8 최종 방법이 하지 않는 것

- blank-suppressed posterior 자체를 semantic target으로 사용하지 않는다.
- raw posterior를 log 공간에서 pooling하지 않는다.
- 최종 primary branch에서 blank 제거 → 재정규화 → legacy top-8 target을 사용하지 않는다.
- temperature로 posterior를 펴지 않는다.
- student 자체 occupancy나 외부 forced alignment를 사용하지 않는다.
- top-32만 재정규화해 tail을 버리지 않는다.

Target 파일 안에는 이전 실험 호환용 `avg_ids/avg_probs` top-8 필드가 남아 있지만, `span_primary_mode=mass3`인 최종 경로에서는 읽지 않는다. 최종 메소드에 쓰이는 top-k는 NTDK 압축용 top-32+tail뿐이다.

---

## 4. 왜 단순 구간 평균과 다른가

Mass3와 NTDK를 각각 맞추면 결국 전체 $q^T$를 맞추는 것과 같아 보일 수 있다. 개념적으로 full conditional NT를 저장한다면 두 층은 pooled posterior의 계층적 factorization이다. 하지만 **학습 gradient의 배분이 다르다.**

예를 들어

$$
q^T=[p(\varnothing)=0.02,\ p(y)=0.979,\ p(\text{all NT})=0.001]
$$

이라고 하자. 전체 KL에서 non-target 내부 관계에 배정되는 영향은 총 0.001에 비례한다. Mass3는 0.001이라는 혼동 총량을 맞추고, NTDK는 그 0.001 안을 다시 1로 정규화하여 별도 weight 8로 비교한다. 따라서

- Mass3: **얼마나 헷갈려야 하는가**
- NTDK: **헷갈린다면 무엇과 헷갈려야 하는가**

를 분리한다. 이것이 단순 full-posterior KL과의 본질적 차이다.

---

## 5. 진단 결과가 지지하는 메커니즘

### 5.1 WHERE는 의미 있는가

- no-KD student의 teacher 대비 mean absolute spike offset은 0.69 frame이다.
- 토큰의 56.6%가 다른 frame을 사용하며, 어긋난 토큰의 82%가 $\pm1$ frame이다.
- $\delta=6$ support는 $\pm1$ offset hit을 10.5%에서 62.1%로, 전체 hit을 48.4%에서 73.0%로 높인다.
- blank suppression으로 새로 편입한 질량의 98.7%가 MFA 자기 단어 구간 $\pm1$ frame 안에 있다.
- 같은 폭의 대칭 kernel보다 자기-token purity가 0.68 대 0.32, 이웃 오염이 0.07 대 0.21이다.

따라서 이득은 “아무 방향으로 smoothing”해서가 아니라 FB가 transcript 순서에 맞는 occurrence shoulder를 선택해서 생긴다.

### 5.2 전체 posterior는 one-hot인데 dark knowledge가 있는가

LibriSpeech test-other 200 utterance, 5,025 token 진단:

- pooled raw distribution의 median GT mass: 0.9986
- pooled raw distribution의 median entropy: 0.0129
- GT/blank 제거 후 conditional NT median entropy: 1.7544
- conditional NT median effective class count: 5.78
- top-32 conditional mass: mean 0.9765, median 0.9881
- 30 dB noise view 안정성: top-32 Jaccard median 0.9394, JS divergence median 0.00362

즉 dark knowledge가 없는 것이 아니라 **전체 질량에서 보이지 않았던 것**이다. Conditional factorization은 그 구조를 인위적으로 만들지 않고 노출한다.

### 5.3 왜 temperature 실험을 메인으로 두지 않는가

Temperature $T=2$는 conditional NT effective class count를 약 90까지 늘리고 top-32 coverage를 약 69%로 낮췄다. 이는 의미 있는 후보를 드러낸다기보다 넓은 tail까지 일괄 증폭할 위험이 있다. 실제 temperature 변형은 기존 방법 대비 분명한 WER 개선을 만들지 못했다.

Non-blank logit standardization도 $T=1$에서 conditional effective class count가 약 627로 거의 uniform에 가까워졌다. 따라서 “더 높은 entropy = 더 좋은 dark knowledge”라는 가설은 폐기하고, raw posterior의 conditional relation을 직접 증류하는 방향을 채택했다.

---

## 6. Experiments

### 6.1 연구 질문

- **RQ1:** Mass3+NTDK가 동일 student/teacher 환경에서 no-KD와 기존 CTC KD보다 WER를 낮추는가?
- **RQ2:** 성능 향상이 단순 blank 제거나 spike dilation이 아니라 occurrence-aware FB support에서 오는가?
- **RQ3:** Mass3와 NTDK를 분리하는 것이 full posterior 또는 coarse mass 하나만 맞추는 것보다 유효한가?
- **RQ4:** raw conditional non-target structure가 충분히 풍부하고 perturbation에 안정적인가?
- **RQ5:** LibriSpeech에서 정한 $(25,8,\delta=6,M=32)$를 조정 없이 TED-LIUM3에 옮겨도 개선이 유지되는가?

### 6.2 데이터와 공통 설정

| 항목 | LibriSpeech | TED-LIUM3 |
|---|---|---|
| Train | train-clean-100 | full train, 268,263 supervisions, 453.813 h |
| Dev/Test | official clean/other | official legacy dev/test |
| Epoch | 100 | 50 |
| Tokenizer | 동일 BPE 1024 | 동일 BPE 1024 |
| Student | 동일 Conformer-CTC 계열 | 동일 구조 |
| Teacher | 동일 계열의 고정 teacher | 동일 teacher recipe |
| Checkpoint | minimum dev greedy WER | minimum dev greedy WER |
| Primary decode | no-LM BPE prefix beam, width 16 | 동일 |

TED text는 고정 LibriSpeech BPE에 맞춰 Kaldi/Lhotse식 noise 및 `<unk>` marker 제거, clitic 결합, number verbalization을 적용한다. `<unk>`가 있다는 이유로 전체 utterance를 버리지 않는다.

### 6.3 메인 비교 방법

1. No KD
2. Vanilla full-posterior frame KD, $T=1$
3. Blank elimination KD
4. Symmetric blank selection KD
5. exact Guided CTC loss
6. S-CTC + supervised CTC fine-tune
7. CR-CTC, two-view compute-matched
8. Mass3 + NTDK (ours)

동일 숫자의 weight를 모든 방법에 강제하지 않는다. loss normalization이 다르기 때문이다. 기존 방법은 논문 설정을 우선하고, dataset-specific 값은 사전에 고정한다. 우리 방법은 LibriSpeech에서 고정한 25/8을 TED에서 재튜닝하지 않는다.

- LibriSpeech vanilla/KD-BE: $\lambda=0.25$; symmetric: $\lambda=0.25,n=4$
- TED vanilla/KD-BE: $\lambda=0.9$; symmetric: $\lambda=1,n=2$
- Guided: paper loss weight 1
- S-CTC: 80+20 epochs(LibriSpeech), 40+10(TED), 총 노출량 유지
- CR-CTC: 두 view이므로 50/25 epochs와 physical batch 16으로 compute-match
- Ours: Mass3 25, NTDK 8, 두 데이터셋 동일

절대 WER를 원 논문과 직접 재현했다고 주장하지 않는다. 모델, 단위, 데이터, LM이 다르므로 **각 논문의 loss/selection rule을 동일한 현재 환경에 재구현한 controlled comparison**이다.

### 6.4 평가 및 보고

Primary table은 no-LM beam-16 WER를 보고한다.

| Method | LibriSpeech test-clean | LibriSpeech test-other | TED-LIUM3 test |
|---|---:|---:|---:|
| No KD | pending | pending | pending |
| Vanilla frame KD | pending | pending | pending |
| Blank elimination KD | pending | pending | pending |
| Symmetric selection KD | pending | pending | pending |
| Guided CTC | pending | pending | pending |
| S-CTC + CTC fine-tune | pending | pending | pending |
| CR-CTC, compute matched | pending | pending | pending |
| **Mass3 + NTDK** | pending | pending | pending |

실행 후 실제 표는 `analysis/paper_main_table_beam16.md`, 전체 greedy/beam 원자료는 `analysis/paper_main_metrics_beam16.csv`에 자동 생성된다. Dev 결과와 greedy 결과는 appendix에 두고, test는 모델 선택에 사용하지 않는다. LM 결과는 primary가 아니라 별도 표로 둔다.

현재 확보된 LibriSpeech pilot에서 no-KD 대비 ours의 no-LM beam-16 WER는 test-clean 14.9821→12.0454, test-other 34.1440→29.7270이었다. 이는 각각 19.60%, 12.93% relative WER reduction이다. 다만 새 exact-baseline suite와 TED 결과가 완성되기 전에는 pilot 수치를 최종 메인 테이블의 결론으로 사용하지 않는다.

### 6.5 필수 ablation과 이미 끝난 진단

| 질문 | 실험/진단 | 상태 |
|---|---|---|
| FB support가 단순 smoothing보다 나은가 | matched-width kernel, no-FB, MFA, student-FB | 완료; mechanism/appendix 사용 |
| $\delta$ 범위 | 0/3/6/9/12 support purity·coverage 및 WER | 완료; $\delta=6$ 근거 |
| temperature가 해결하는가 | raw/T2/logit-standardization 진단 및 학습 | 완료; 메인 방향에서 제외 |
| conditional NT가 존재하는가 | entropy, effective classes, stability, top-M coverage | 완료 |
| Mass3가 GT/NT보다 필요한가 | GT/NT coarse ablation | 완료; 성능 열세로 appendix |
| Mass3와 NTDK 각각의 기여 | mass-only, NTDK-only, combined | 최종 논문 ablation 표에 정리 필요 |
| NTDK weight 범위 | 0.5/2/8/15/20 등 | 완료된 run을 한 표로 재정리 필요 |
| top-32+tail 선택 | M=8/16/32/64 또는 full 소규모 대조 | 저장 근거 진단은 완료; 학습 ablation은 선택 사항 |
| 일반화 | TED-LIUM3-full, 동일 25/8 | 메인 suite 진행 |
| 분산 | 최종 rows multi-seed | 메소드·프로토콜 동결 후 마지막 수행 |

기존 span-KD 변형들은 메인 비교 행으로 되살리지 않는다. 필요한 경우 WHERE mechanism을 증명하는 appendix ablation으로만 사용한다.

### 6.6 결과 해석 기준

- Ours가 no-KD와 기존 KD를 두 데이터셋에서 일관되게 이기면 주된 성능 주장을 지지한다.
- CR-CTC와 비슷하더라도 single-view external-teacher KD로 compute-matched CR-CTC와 동률이면 의미가 있다. 단, “dark knowledge라서 beam gain이 특별히 크다”는 주장은 실제 beam-minus-greedy 차이가 더 클 때만 한다.
- Mass3+NTDK가 Mass3-only보다 좋아야 conditional dark knowledge의 독립 기여를 주장할 수 있다.
- TED에서 weight 재튜닝 없이 개선되면 단순 LibriSpeech weight fitting 반론을 줄인다.
- seed 평균의 차이가 표준편차보다 작으면 우위가 아니라 동률로 표현한다.

### 6.7 재현 명령

```bash
bash experiments/run_paper_main_all.sh
```

이 명령은 순서대로 TED-LIUM3 full 준비, cached posterior 기반 target 생성/재사용, LibriSpeech와 TED 학습/재사용, beam 평가, Markdown/CSV 표 생성을 수행한다. 기존 완료 실험과 target은 기본적으로 재사용한다.

강제로 다시 학습할 때만 `FORCE=1`, 현재 best checkpoint로 평가 로그를 강제로 갱신할 때만 `FORCE_EVAL=1`을 사용한다.

---

## 7. 논문에서 예상되는 질문과 답변

### Q1. “DKD를 CTC에 그대로 적용한 것 아닌가?”

원리는 계승했지만 단위와 factorization이 다르다. DKD는 한 sample의 target/non-target를 나눈다. 우리는 CTC FB로 transcript 내 **occurrence 좌표**를 먼저 만들고, CTC 고유의 blank를 일반 non-target에서 분리한 3-way mass와 occurrence-conditional NT를 사용한다. 시간 정렬과 class decoupling을 동시에 다룬다는 것이 핵심 차이다.

### Q2. “blank를 없애면 alignment가 해결된 후 오히려 손해 아닌가?”

최종 semantic target에서 blank를 없애지 않는다. blank suppression은 WHERE occupancy를 계산할 때만 쓰며, 실제 raw blank mass는 Mass3로 전달한다. 따라서 useful blank information을 삭제하는 KD-BE와 다르다.

### Q3. “FB까지 할 필요 없이 blank suppression만 하면 되지 않나?”

Blank suppression은 각 frame에서 non-blank 후보를 살릴 뿐, 동일 BPE 반복 occurrence와 transcript 순서를 구분하지 못한다. FB가 CTC path 제약으로 살아난 frame 질량을 특정 $u$에 할당한다. no-FB와 matched-width kernel이 더 낮은 purity와 높은 이웃 오염을 보인 진단이 이를 지지한다.

### Q4. “Mass3와 NTDK를 합치면 그냥 full KL 아닌가?”

분포 표현만 보면 계층 분해지만 optimization은 다르다. full KL에서는 NT 내부 gradient가 NT 총질량에 곱해져 사라진다. NTDK는 조건부 정규화와 독립 weight로 이 coupling을 끊는다. 또한 top-32+tail은 tail 내부를 압축하는 실용적 근사다.

### Q5. “높은 entropy가 noise일 수도 있지 않나?”

맞다. 그래서 temperature로 entropy를 키우는 방식을 채택하지 않았다. raw conditional NT의 candidate overlap과 JS divergence를 작은 음향 perturbation에서 측정해 안정성을 확인했고, 성능 ablation으로 최종 유효성을 판단한다.

### Q6. “왜 NTDK weight 8이고 Mass3 weight 25인가?”

두 loss는 정규화와 정보 역할이 달라 동일 weight가 수식적으로 공정한 것이 아니다. LibriSpeech dev sweep에서 각각의 scale을 정했고, test를 보지 않은 채 25/8을 동결해 TED로 전이한다. 논문에는 탐색 범위와 dev selection rule을 투명하게 공개한다.

---

## 8. 한계

1. Teacher transcript와 raw posterior가 필요하므로 target 사전 생성 및 저장 비용이 있다.
2. $\delta=6$, $M=32$, 25/8의 architecture·tokenizer 변화에 대한 민감도는 추가 검증이 필요하다.
3. Top-32+tail은 tail 총질량은 보존하지만 tail 내부 클래스 관계는 보존하지 않는다.
4. Teacher가 systematic하게 잘못된 non-target 관계를 갖는 경우 NTDK가 그 오류를 강화할 수 있다.
5. 현재 stability 진단은 제한된 subset과 한 종류의 perturbation에 기반한다.
6. LibriSpeech와 TED-LIUM3는 모두 영어 read/talk speech이므로 multilingual 및 극단적 domain shift 일반화는 미검증이다.
7. 현재 메인 러너는 seed 1의 protocol-freeze 표를 만든다. 최종 논문 수치는 다중 seed 평균과 표준편차가 필요하다.

---

## 9. Conclusion 초안

CTC distillation의 어려움은 spike alignment와 dark-knowledge scarcity를 하나의 frame posterior KL로 동시에 해결하려는 데서 생긴다. 본 연구는 forward-backward occupancy를 semantic target이 아닌 token-occurrence coordinate로 재해석한다. Blank-suppressed FB가 WHERE를 정하고, unmodified teacher posterior가 WHAT을 제공한다. Occurrence-pooled posterior를 blank/GT/non-target 총질량과 conditional non-target relation으로 분리함으로써 emission calibration을 유지하면서, 정답 spike에 가려진 dark knowledge에 독립적인 학습 신호를 부여한다.

이 방법은 temperature로 불확실성을 인위적으로 만들거나 blank를 semantic target에서 삭제하지 않으며, inference-time overhead도 없다. LibriSpeech-100과 TED-LIUM3-full의 controlled comparison을 통해 occurrence-aware temporal tolerance와 hierarchical dark-knowledge transfer가 기존 frame selection, alignment forcing, sequence occupancy KD 및 consistency regularization을 넘어서는지 검증한다.

---

## 10. 참고문헌 후보

1. [Graves et al., *Connectionist Temporal Classification*, ICML 2006](https://mlanthology.org/icml/2006/graves2006icml-connectionist/)
2. [Hinton et al., *Distilling the Knowledge in a Neural Network*, 2015](https://arxiv.org/abs/1503.02531)
3. [Huang et al., *Knowledge Distillation for Sequence Model*, Interspeech 2018](https://www.isca-archive.org/interspeech_2018/huang18d_interspeech.html)
4. [Kurata and Audhkhasi, *Guiding CTC Posterior Spike Timings for Improved Posterior Fusion and Knowledge Distillation*, Interspeech 2019](https://www.isca-archive.org/interspeech_2019/kurata19_interspeech.html)
5. [Tian et al., *Knowledge Distillation for CTC-based Speech Recognition via Consistent Acoustic Representation Learning*, Interspeech 2022](https://www.isca-archive.org/interspeech_2022/tian22_interspeech.html)
6. [Lee and Chang, *Inter-KD: Intermediate Knowledge Distillation for CTC-based ASR*, 2022](https://arxiv.org/abs/2211.15075)
7. [Zhao et al., *Decoupled Knowledge Distillation*, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Zhao_Decoupled_Knowledge_Distillation_CVPR_2022_paper.html)
8. [Kim et al., *Analyzing and Reducing the Performance Gap in CTC-based Knowledge Distillation*, Interspeech 2024](https://arxiv.org/abs/2406.07909)
9. [CR-CTC, *Consistency Regularization for CTC-based Speech Recognition*, ICLR 2025](https://arxiv.org/abs/2410.05101)
10. [Hilmes et al., *On the Influence of CTC Blank Frames in Knowledge Distillation*, Interspeech 2025](https://arxiv.org/abs/2506.01503)
11. [Tian et al., *Factorized and Progressive Knowledge Distillation for CTC-based ASR Models*, Speech Communication 2024](https://doi.org/10.1016/j.specom.2024.103071)
12. [Hernandez et al., *TED-LIUM 3*, 2018](https://arxiv.org/abs/1805.04699)
