# Token-Averaging KD for CTC ASR

## 배경 및 문제 정의

**설정**: Teacher (stt_en_conformer_ctc_small) → Student (Conformer d=144, 8L, 4× sub)

### 기존 방법의 한계

**① Vanilla Logit KD** — blank 프레임 무차별 supervision
- CTC의 구조적 특성상 전체 프레임의 **78.5%가 blank** (음소 정보 없음)
- 전 프레임에 동일한 KD loss → loss 대부분이 blank 분포 복제에 낭비, 음소 신호 희석
- blank/non-blank 프레임을 동등하게 취급 → CTC alignment 구조를 전혀 활용하지 않음

**② KD-BE** (Hilmes et al. 2025) — argmax 기반 마스킹의 한계
- Teacher argmax ≠ blank인 프레임만 supervision → blank 제거 효과는 있으나
- **"어떤 토큰" 구간인지 모름**: 프레임이 어떤 linguistic 단위에 속하는지 정보 없이 독립 supervision
- Hard threshold (argmax 기반): teacher 확률이 불확실한 전환 구간(e.g., P(blank)=0.45, P(tok)=0.40)도 non-blank로 처리
- 서로 다른 subsampling rate의 teacher–student 쌍에 직접 적용 불가 (프레임 1:1 대응 가정)

**③ Symmetric blank selection** (Hilmes et al. 2025) — 맥락 확장의 효과 제한
- non-blank 주변 ±n blank 프레임까지 supervision 확장 → 맥락 정보 활용 의도
- 실제 효과 분석: blank-normalized posterior의 top-1 정확도가 **dist=1에서 48.7%**, **dist=2에서 6.3%**로 급감
- dist=2 이상은 correct token 신호가 거의 없음(랜덤 수준) → n≥2 확장은 오히려 노이즈 추가
- alignment 없이 거리 기반 휴리스틱에 의존 → 세그먼트 경계와 무관

**→ 공통 문제**: 세 방법 모두 CTC forced alignment 없이 프레임 단위로 supervision, 토큰 정체성(token identity)을 supervision 신호에 반영하지 못함

---

## 방법 비교

| 방법 | 핵심 아이디어 | 비고 |
|------|-------------|------|
| **Vanilla Logit KD** | 전 프레임 KD loss | Hilmes et al. 2025 |
| **KD-BE** (Blank Elimination) | Teacher argmax ≠ blank인 프레임만 | Hilmes et al. 2025 |
| **Symmetric (n=1)** | 비blank 주변 ±1 프레임까지 포함 | Hilmes et al. 2025 |
| **Token-avg KD** *(제안)* | Viterbi forced alignment → 토큰 세그먼트 단위 평균 posterior를 student에 매칭 | 본 실험 |

**Token-avg KD 방식**:  
1. Teacher greedy decoding → 토큰 시퀀스 추출  
2. Viterbi forced alignment → 각 프레임을 토큰/blank에 할당  
3. 비blank 토큰 세그먼트별로 teacher posterior 평균  
4. 세그먼트에 대응하는 student 프레임 구간의 평균 logit과 KL loss

**기존 한계 극복 포인트**:
- blank 78.5% 문제 → Viterbi로 non-blank 세그먼트만 명시적 추출 (① ② 해결)
- 토큰 정체성 없음 → 각 KD 타겟이 특정 토큰에 귀속된 세그먼트 단위 supervision (② 해결)
- hard threshold 문제 → Viterbi로 각 프레임을 가장 likely한 토큰에 확정 할당 (② 해결)
- subsampling rate 불일치 → 세그먼트 단위 매칭으로 teacher/student frame rate 차이 흡수 (② 해결)
- 맥락 거리 한계 → 세그먼트 내 프레임 평균으로 단일 프레임 posterior의 노이즈 감소 (③ 해결)

---

## 실험 결과 (LibriSpeech 100h, 4× subsampling)

| 방법 | KD Weight | test-clean | test-other |
|------|-----------|-----------|------------|
| No KD | — | 15.08% | 34.29% |
| Vanilla Logit KD | w=10 | 13.92% | 33.22% |
| KD-BE | w=10 | **12.84%** | 31.16% |
| Symmetric (n=1) | w=10 | 13.12% | 31.60% |
| Token-avg KD | w=10 | 12.90% | 30.99% |
| Token-avg KD | w=20 | 12.66%* | 30.96%* |

> \* **주의**: token-avg 타겟이 blank_id 버그(BLANK=0 사용, 정상값=1024)로 오염된 상태에서 학습됨. 현재 올바른 타겟으로 재빌드 완료, 재실험 예정.

---

## 핵심 관찰

- **KD-BE가 Vanilla Logit KD 대비 +1.08%p 개선** (13.92% → 12.84%)  
  → blank 프레임 제거만으로도 유의미한 효과  

- **Token-avg ≈ KD-BE in practice**  
  → CTC peaky behavior로 인해 Viterbi alignment의 대부분 토큰이 1프레임짜리 spike  
  → 1프레임 세그먼트의 "평균"은 사실상 해당 프레임의 posterior와 동일  
  → Token-avg의 이론적 우위(다중 프레임 집계)가 실제로는 거의 발현되지 않음  

- **Token-avg 재실험 필요**: 올바른 타겟으로 학습 시 KD-BE와의 실질적 차이 확인 예정
