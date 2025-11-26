# 데이터 전처리 결과서

## 1. 수집한 데이터 설명
### 1.1. 수집목적
고객이탈을 예측 및 방지하여 회사의 손해를 줄이고, 충성할 고객을 유치함.
즉, **고객 생애가치(LTV)를 극대화**하기 위함.
>>
- 선제적 이탈 방지 (Proactive Retention):
> 고객이 곧 서비스를 떠날 위험이 있는지 사전에 파악
고위험 고객에게 프로모션/혜택 제공 → 이탈률 감소
마케팅 비용 최적화.

- 고수익 고객 관리 (VIP Retention):
> 'Total_Trans_Amt', 'Credit_Limit', 'Avg_Utilization_Ratio' 등을 통해
고가치(VIP) 고객, 저활동 고객, 리스크 고객
등을 세분화하여 마케팅 전략에 사용.

- 서비스 불만 감지 및 휴면 고객 활성화:

> 장기간 비활성 고객이나 콜센터 접촉이 많은 고객을 파악
고객 불만 조기 감지 → 서비스 개선

### 1.2. 수집한 방법
[데이터 출처] </br>
2020년 말 ([Original link](https://analyttica.com/leaps/)) -> 웹사이트의 운영정책 변경으로 현재 데이터 접근 불가 </br> -> 캐글에 있는 데이터셋 사용 ([kaggle:Credit Card Customer Churn dataset](https://www.kaggle.com/datasets/thedevastator/predicting-credit-card-customer-attrition-with-m))

### 1.3. Feature들에 대한 설명
**전체 컬럼 목록**

| 번호 | 컬럼명 | 설명 |
| --- | --- | --- |
| 1 | CLIENTNUM | 회원 ID |
| 2 | Attrition_Flag | 이탈 여부 |
| 3 | Customer_Age | 나이 |
| 4 | Gender | 성별 |
| 5 | Dependent_count | 부양 가족 수 |
| 6 | Education_Level | 학력 |
| 7 | Marital_Status | 결혼 여부 |
| 8 | Income_Category | 소득 구간 |
| 9 | Card_Category | 카드 등급 |
| 10 | Months_on_book | 고객 관계 기간 |
| 11 | Total_Relationship_Count | 총 상품 수 |
| 12 | Months_Inactive_12_mon | 비활성화 개월 수 |
| 13 | Contacts_Count_12_mon | Contact 횟수 |
| 14 | Credit_Limit | 신용한도 |
| 15 | Total_Revolving_Bal | 총 리볼빙 금액 |
| 16 | Avg_Open_To_Buy | 평균 사용가능 금액 |
| 17 | Total_Amt_Chng_Q4_Q1 | 거래금액 변화율 |
| 18 | Total_Trans_Amt | 총 거래량 |
| 19 | Total_Trans_Ct | 총 거래 횟수 |
| 20 | Total_Ct_Chng_Q4_Q1 | 거래 횟수 변화율 |
| 21 | Avg_Utilization_Ratio | 평균 신용 사용률 |
| 22 | Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1 | 예측 모델 1 |
| 23 | Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2 | 예측 모델 2 |


## 2. EDA 결과

> ✓ Feature Engineering 완료!  
> 생성된 파생 변수: Trans_Change_Ratio, Inactivity_Score, Engagement_Score, Utilization_Risk_Level  
> 최종 데이터 shape: (10127, 34)
### 2.1. 결측치

#### 결측치 확인 코드

```python
# 'Unknown' 문자열을 결측치로 간주
unknown_values = pd.Series(dtype=int)
for col in df.select_dtypes(include='object').columns:
    unknown_count = (df[col].str.lower() == 'unknown').sum()
    if unknown_count > 0:
        unknown_values[col] = unknown_count

# 결측치 정보
missing_df = pd.DataFrame({
    'Unknown_Count': unknown_values,
    'Missing_Percent': (unknown_values / len(df)) * 100
})
missing_df = missing_df[missing_df['Unknown_Count'] > 0].sort_values('Unknown_Count', ascending=False)

if len(missing_df) > 0:
    print("결측치가 있는 컬럼 ('Unknown' 포함):")
    print(missing_df)
```

#### 결측치 확인 결과

```
결측치가 있는 컬럼 ('Unknown' 포함):
                 Unknown_Count  Missing_Percent
Education_Level           1519        14.999506
Income_Category           1112        10.980547
Marital_Status             749         7.396070

총 결측치 비율: 1.59%
```

- **결측치가 있는 컬럼**: Education_Level - 1519개 (15.00%), Income_Category - 1112개 (10.98%), Marital_Status - 749개 (7.40%)

#### 어떻게 처리했는가?

- **범주형 컬럼**: `df[col].mode()[0]`를 사용하여 최빈값(가장 많이 등장하는 값)으로 결측치 채움.
- **수치형 컬럼**: 0의 값을 "결측값"으로 여기기. 0인 부분을 NaN으로 바꾼 후에 평균(mean_val)을 계산 후 평균값으로 채움.

#### 왜 그렇게 처리 했는가?

- **범주형 컬럼 (Object 타입)**: 최빈값으로 채우며, 기존 데이터 패턴을 덜 깨뜨리면서, 새로운 "가짜 카테고리"를 생성할 필요성이 줄어듦.
- **수치형 컬럼 (number 타입)**: 분포를 크게 왜곡하지 않으면서, 모델이 결측 때문에 튀는 패턴을 배우지 않도록 막아주는 안전한 선택.

### 2.2. 이상치

#### 이상치 탐지 코드

```python
def plot_boxplots_by_column(df, columns=None, figsize=(15, 10), cols_per_row=3, save_path=None):
    """
    숫자형 칼럼별로 박스플롯을 그려서 이상치를 탐지하는 함수
    IQR 방법을 사용하여 이상치를 계산
    """
    # 숫자형 칼럼 자동 선택
    if columns is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'Attrition_Binary' in numeric_cols:
            numeric_cols.remove('Attrition_Binary')
        columns = numeric_cols
    
    # 각 칼럼별로 박스플롯 그리기 및 이상치 계산
    for col in columns:
        data = df[col].dropna()
        # 이상치 계산 (IQR 방법)
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = data[(data < lower_bound) | (data > upper_bound)]
        # ... (시각화 코드)
```

#### 이상치 탐지 결과

![이상치 탐지 박스플롯](images/boxplot_outliers_by_column.png)

**주요 이상치 발견:**
- Credit_Limit: 984개 이상치 (9.72%)
- Avg_Open_To_Buy: 963개 이상치 (9.51%)
- Total_Trans_Amt: 896개 이상치 (8.85%)
- Contacts_Count_12_mon: 629개 이상치 (6.21%)

#### 이상치 판정 기준

- **IQR (Interquartile Range) 방법**: Q1 - 1.5×IQR 이하 또는 Q3 + 1.5×IQR 이상인 값을 이상치로 판정
- 코드 상에선 명시적인 이상치 탐지/제거 로직(IQR, z-score 등)은 구현되어 있지만, 제거는 하지 않음.

#### 어떻게 처리했는가?

- 명시적인 이상치 제거는 하지 않았지만, 간접적으로 완화되는 부분은 감지:
  - 0 → 평균값 대체 로직: 만약 어떤 컬럼에서 0이 비정상적으로 많다면, 그 값들을 평균으로 치환되면서 분포가 조금 더 부드러워짐.

#### 왜 그렇게 처리 했는가?

- 고객 행동 데이터에서는 이상치처럼 보이는 값도 실제로 중요한 "특이 행동 패턴"일 수 있기 때문.
  - **예시**: 거래가 갑자기 폭증 → 사용률이 비정상적으로 높음 → 오히려 이탈 신호일 수도 있음.
- 따라서 단순히 통계 기준(IQR, z-score)만으로 강하게 자르기보다는 모델이 스스로 학습하도록 두고, 추후 모델 성능/Feature Importance를 보며 필요할 때만 추가로 처리하는 방향을 선택.

### 2.3. 기타 전처리 방법

#### 2.3.1. 범주형 변수별 이탈률 분석

**코드:**
```python
# 범주형 변수별 이탈률 계산 및 시각화
categorical_vars = ['Dependent_count', 'Gender', 'Education_Level', 
                    'Marital_Status', 'Income_Category', 'Card_Category']

for var in categorical_vars:
    churn_by_var = df.groupby(var)['Attrition_Binary'].agg(['mean', 'count']).reset_index()
    churn_by_var.columns = [var, 'Churn_Rate', 'Count']
    churn_by_var = churn_by_var.sort_values('Churn_Rate', ascending=False)
    # 시각화...
```

**결과:**
![범주형 변수별 이탈률 분석](images/categorical_churn_analysis.png)

**주요 발견사항:**
- **Card_Category**: Platinum 카드 고객의 이탈률이 25.0%로 가장 높음 (샘플 수 20개)
- **Gender**: 여성(F) 고객의 이탈률(17.36%)이 남성(M) 고객(14.62%)보다 높음
- **Education_Level**: Doctorate 학력 고객의 이탈률이 21.06%로 가장 높음
- **Income_Category**: $120K+ 고객의 이탈률이 17.33%로 높음

#### 2.3.2. 연속형 변수 통계적 비교

**코드:**
```python
from scipy import stats

continuous_vars = ['Total_Trans_Amt', 'Total_Trans_Ct', 'Total_Revolving_Bal',
                   'Customer_Age', 'Months_on_book', 'Credit_Limit',
                   'Avg_Utilization_Ratio', 'Total_Relationship_Count']

for var in continuous_vars:
    churned = df[df['Attrition_Binary'] == 1][var].dropna()
    retained = df[df['Attrition_Binary'] == 0][var].dropna()
    # t-test 수행
    t_stat, p_value = stats.ttest_ind(churned, retained)
    # 효과 크기 (Cohen's d) 계산
```

**결과:**
![연속형 변수 통계적 비교](images/continuous_statistical_comparison.png)

**주요 발견사항:**
- **Total_Trans_Ct**: 이탈 고객의 평균 거래 횟수가 유지 고객보다 약 23.7회 낮음 (p < 0.001, Cohen's d = -1.09)
- **Total_Revolving_Bal**: 이탈 고객의 평균 리볼빙 잔액이 약 583.8달러 낮음 (p < 0.001)
- 모든 주요 연속형 변수에서 이탈 고객과 유지 고객 간 통계적으로 유의한 차이 발견

#### 2.3.3. 소득 범주별 및 연령별 이탈률 분석

**소득 범주별 이탈률:**
![소득 범주별 이탈률](images/churn_rate_by_income.png)

**코드:**
```python
grouped_churn = df.groupby('Income_Category')['Churn'].mean().reset_index()
grouped_churn = grouped_churn.sort_values(by='Churn', ascending=False)
sns.barplot(x='Income_Category', y='Churn', data=grouped_churn)
```

**연령별 이탈 분포:**
![연령별 이탈 분포](images/churn_distribution_by_age.png)

**코드:**
```python
sns.histplot(
    data=df, 
    x='Customer_Age', 
    hue='Attrition_Flag', 
    kde=True,
    stat="density"
)
```

**주요 발견사항:**
- 전체 평균 이탈률: 16.07%
- 소득이 높을수록($120K+) 이탈률이 높은 경향
- 연령대별로 이탈 고객과 유지 고객의 분포가 유사하지만, 일부 연령대에서 차이 존재

#### 2.3.4. Feature들과 target_col의 상관관계

**전체 상관관계 히트맵:**
![상관관계 히트맵](images/pearson_correlation_heatmap.png)

**코드:**
```python
numeric_df = df.select_dtypes(include=[np.number])
corr_matrix = numeric_df.corr(method="pearson")
sns.heatmap(corr_matrix, cmap="coolwarm", annot=True, fmt=".2f")
```

**타겟과의 상관관계 Top 5:**
![타겟과의 상관관계 Top 5](images/top5_pearson_correlations_attrition.png)

**코드:**
```python
attrition_corr = corr_matrix["Attrition_Binary"].drop("Attrition_Binary")
top5_corr = attrition_corr.reindex(
    attrition_corr.abs().sort_values(ascending=False).index
).head(5)
```

**결과:**
```
Attrition_Binary와의 Pearson 상관계수 Top 5
Total_Trans_Ct          -0.371403
Total_Ct_Chng_Q4_Q1     -0.290054
Total_Revolving_Bal     -0.263053
Contacts_Count_12_mon    0.204491
Avg_Utilization_Ratio   -0.178410
```

**분석 결과:**
- **음의 상관관계**: Total_Trans_Ct, Total_Ct_Chng_Q4_Q1, Total_Revolving_Bal, Avg_Utilization_Ratio
  - 거래 횟수가 적을수록, 거래 변화율이 낮을수록, 리볼빙 잔액이 낮을수록 이탈 가능성이 높음
- **양의 상관관계**: Contacts_Count_12_mon
  - 콜센터 접촉 횟수가 많을수록 이탈 가능성이 높음 (불만 고객일 가능성)

**이 결과를 참고해서:**
- Engagement_Score, Inactivity_Score 같은 파생변수를 설계
- 거래 관련 변수를 묶어주는 방향의 Feature Engineering을 진행

#### 2.3.5. 상관관계 계수가 높은 feature 처리방법

- **두 Feature 간 상관계수가 |corr| > 0.85처럼 매우 높으면**: 둘 다 거의 같은 정보를 가지고 있다고 볼 수 있음.
  - 의미 해석이 더 쉬운 컬럼 하나만 남기거나
  - 둘을 합쳐서 요약 Feature (점수/비율)로 바꾸는 전략 사용.

#### 2.3.6. target_col과의 상관관계 계수 절대값이 높은 feature 처리방법

- 타겟과의 상관관계 절대값이 높은 Feature는 기본적으로 모델에 포함하되, 교차 검증을 통해 과적합 여부를 확인하고, 필요한 경우 정규화 및 규제를 통해 안정성을 확보.

#### 2.3.7. 범주형 데이터 인코딩 방법

- 범주형 Feature는 `pd.get_dummies(drop_first=True)`를 통해 One-Hot Encoding 하였으며, 기준 카테고리를 하나 제거(k-1 인코딩)하여 다중공선성을 완화.
- 인코딩 이후에는 원본 범주형 컬럼 대신 인코딩된 더미 컬럼을 사용.

## 2.4. 적용한 Feature Engineering 방식

  * 비율 (Ratio): 숫자가 크거나 작거나의 결과값으로 추정하는것이 아니라, 숫자가 얼마만큼 변했는지를 보는 것이 중요할때 사용. 

  * 조합기반 (Combination): 두 개 이상의 컬럼을 곱셈·가중합 형태로 결합해 고객의 활동 특성을 단일 점수로 표현하는 방식. 

  * 구간 나누어 주기 (Binning): 숫자를 그대로 쓰는것이 아니라, 각자 라벨링을 줘서 등급을 표기하는 방식. 

### 2.4.1. Feature Scaling
  * Linear 모델(Logistic, SVM) : StandardScaler / MinMaxScaler 적용.
  * Tree 계열 모델(XGBoost/RandomForest) : Scaling 불필요.

### 2.4.2. Feature Selection
  * 상관계수 기반 필터링.
  * XGBoost Feature Importance 기반 선정.

### 2.4.3. Feature Engineering

#### 파생 변수 생성 코드

```python
# 1) 거래 변화율 (거래 횟수 대비 분기 변화 비율)
df["Trans_Change_Ratio"] = (
    df["Total_Trans_Ct"] / (df["Total_Ct_Chng_Q4_Q1"] + 1)
)

# 2) 비활동 기반 리스크 스코어
df["Inactivity_Score"] = (
    df["Months_Inactive_12_mon"] * df["Avg_Utilization_Ratio"]
)

# 3) 고객 참여도 스코어 (Engagement Score)
df["Engagement_Score"] = (
    df["Total_Trans_Amt"] * 0.4 +
    df["Total_Trans_Ct"] * 0.4 -
    df["Months_Inactive_12_mon"] * 0.2
)

# 4) Utilization 기반 위험 구간화
df["Utilization_Risk_Level"] = pd.cut(
    df["Avg_Utilization_Ratio"].fillna(0),
    bins=[0, 0.3, 0.6, 1.0],
    labels=[0, 1, 2],
    include_lowest=True
).astype(int)

print("✓ Feature Engineering 완료!")
print(f"생성된 파생 변수: Trans_Change_Ratio, Inactivity_Score, Engagement_Score, Utilization_Risk_Level")
print(f"최종 데이터 shape: {df.shape}")
```

**출력 결과:**
```
파생 변수 생성 중...

✓ 거래 변화율 생성
✓ 비활동 기반 리스크 스코어 생성
✓ 고객 참여도 스코어 생성
✓ Utilization 기반 위험 구간화 생성
✓ Feature Engineering 완료!
생성된 파생 변수: Trans_Change_Ratio, Inactivity_Score, Engagement_Score, Utilization_Risk_Level
최종 데이터 shape: (10127, 25)
```

#### 생성된 파생 변수 설명

1. **거래 변화율 지표 (Trans_Change_Ratio)**
   - 거래 급증 또는 급감 패턴이 이탈률과 유의한 상관관계를 가지는 점을 반영.
   - 고객의 총 거래 횟수 대비 분기 변화율을 정규화한 지표.
   - 공식: `Total_Trans_Ct / (Total_Ct_Chng_Q4_Q1 + 1)`

2. **비활동 기간 리스크 (Inactivity_Score)**
   - 고객의 금융 행동 안정성 / 불균형 신호를 모델이 인식하도록 유도.
   - 비활동 개월 수 × 카드 사용률.
   - 공식: `Months_Inactive_12_mon * Avg_Utilization_Ratio`

3. **고객 활동 점수 (Engagement_Score)**
   - 고객의 적극적인 활용성 점수가 낮다는 경향 발견을 반영.
   - 거래 금액 + 거래 횟수 - 비활동 패널티.
   - 공식: `Total_Trans_Amt * 0.4 + Total_Trans_Ct * 0.4 - Months_Inactive_12_mon * 0.2`

4. **카드 사용률 기반 위험 등급 (Utilization_Risk_Level)**
   - 비선형적 위험 구간을 "범주형 라벨"로 모델에 제공.
   - 구간: 0-0.3 (낮음), 0.3-0.6 (중간), 0.6-1.0 (높음)