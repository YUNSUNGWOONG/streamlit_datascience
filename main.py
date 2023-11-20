import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import poisson
import pandas as pd

st.write('# Made by 윤성웅(2020117744)')

st.write('## 푸아송 분포(Poisson distribution)')
st.write('푸아송 분포는 일정한 시간 또는 공간에서 발생하는 이산적인 사건의 수를 모델링하는 확률분포입니다.'
         '주로 특정 기간 내에 사건이 발생하는 횟수를 다루는덷 사용됩니다. 이 분포를 통해 다양한 정보를 얻을 수 있습니다.')
st.write('### 1.평균 사건 발생률:')
st.write('푸아송 분포의 매개변수인  λ(람다)는 단위 시간 또는 단위 공간당 평균 사건 발생률을 나타냅니다.'
         '분포의 모양은 이 평균 발생률을 중심으로 형성됩니다.')

st.write('### 2.확률 계산:')
st.write('주어진 시간 동안 발생하는 사건의 수에 대한 확률을 계산할 수 있습니다. 예를 들어,'
         '어떤 이벤트가 시간당 평균 2번 발생하는 경우, 1시간 동안 이벤트가 정확히 3번 발생할 확률을'
         '계산할 수 있습니다.')

st.write('### 3.사건 발생 간격:')
st.write('푸아송 분포를 사용하면 사건 발생 간격에 대한 정보를 얻을 수 있습니다. 즉, 한 사건과'
         '다음 사건 사이의 시간 간격에 대한 확률을 모델링할 수 있습니다.')

st.write('### 4.사건의 예측:')
st.write('푸아송 분포를 사용하여 미래에 발생할 사건의 수를 예측할 수 있습니다. 예를 들어,'
         '고정된 시간 동안의 사건 발생률을 알고 있다면, 다음 시간 동안의 사건 수에 대한 예측을 수행할'
         '수 있습니다.')

st.write('### 5.대기열 이론:')
st.write('푸아송 분포는 대기열 이론(queuing theory)에서 널리 사용됩니다. 대기열에서도'
         '도착하는 고객의 수나 요청의 수 등을 모델링하는 데 활용됩니다.')
st.write('')
st.write('(푸아송 분포는 특히 독립적인 사건들이 고정된 비율로 발생할 때 유용하며, 큰 표본에 대해'
         '정규 분포에 수렴하는 중심극한정리와 관련이 있습니다.)')


#diagram(1)
plt.rc('font', family='Malgun Gothic')

# Load data
df = pd.read_csv('lotto-3.csv', encoding='cp949', index_col=False)
df1 = df[['Drawing date', 'First place winners', 'Second place winners',
          'Third place winners', 'Fourth place winners', 'Fifth place winners']]
df2 = df[['1', '2', '3', '4', '5', '6', 'Bonus']]

kimdaejoong = df1.loc[1:16]
nomoohyun = df1.loc[17:273]
leemyungbak = df1.loc[274:534]
bakgunhae = df1.loc[535:753]
moonjaein = df1.loc[754:1014]
yoonsukyeol = df1.loc[1015:1091]

governments = {
    'kimdaejoong': kimdaejoong,
    'nomoohyun': nomoohyun,
    'leemyungbak': leemyungbak,
    'bakgunhae': bakgunhae,
    'moonjaein': moonjaein,
    'yoonsukyeol': yoonsukyeol
}

# Create Streamlit app
st.write('### ※로또 당첨자 수 분석')
government_view = pd.DataFrame(columns=['Government', 'Average Winners'])
for government_name, government_data in governments.items():
    temp = government_data
    t1 = temp.loc[:, 'First place winners']
    avg_winners = t1.mean().round(0)

    # Append data to government_view DataFrame
    government_view = government_view.append({'Government': government_name, 'Average Winners': avg_winners},
                                             ignore_index=True)
    #st.write(f"{government_name} 정권의 1등 당첨자수 평균: {avg_winners}")

st.write(government_view)

#diagram(2)
st.write('### ※평균 1등 당첨자 수가 3명, 5명, 6명, 8명, 10명일때의 푸아송분포')
# Set up the plot
fig, ax = plt.subplots()
ax.set_ylim(0, 0.3)
ax.set_xlim(0, 15)
ax.set_title("Poisson distribution")
ax.set_xlabel("x")
ax.set_ylabel("p(x)")

# Values of lambda
lambdas = [3, 5, 6, 8, 10, 13]

# Plot Poisson distributions for different lambdas
for i in range(5):
    x = np.arange(0, 15)
    y = poisson.pmf(x, lambdas[i])

    ax.plot(x, y, marker='o', linestyle='-', color=plt.cm.rainbow(i / 5.0))

    # Add vertical line at lambda
    ax.axvline(x=lambdas[i], color=plt.cm.rainbow(i / 5.0), linestyle='--')

    # Add label for lambda
    ax.text(lambdas[i], np.max(y) + 0.01, f"lambda={lambdas[i]}", color=plt.cm.rainbow(i / 5.0))

# Show the plot in Streamlit app
st.pyplot(fig)
