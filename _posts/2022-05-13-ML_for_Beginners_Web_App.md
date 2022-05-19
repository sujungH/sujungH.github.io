# ML 모델을 사용하는 웹 앱 빌드

## 1.데이터 정리

이 단원에서는 NUFORC (National UFO Reporting Center)에서 수집한 80,000번의 UFO 목격 데이터를 사용합니다. 이 데이터에는 UFO 목격에 대한 몇 가지 흥미로운 설명이 있습니다. 예를 들면 다음과 같습니다.

1. pandas, matplotlib및 numpy이전 수업에서 했던 것처럼 ufos 스프레드시트를 가져옵니다. 샘플 데이터 세트를 살펴볼 수 있습니다.


```python
import pandas as pd
import numpy as np

ufos = pd.read_csv('C:/Users/PC/Desktop/ufos.csv')
ufos.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>datetime</th>
      <th>city</th>
      <th>state</th>
      <th>country</th>
      <th>shape</th>
      <th>duration (seconds)</th>
      <th>duration (hours/min)</th>
      <th>comments</th>
      <th>date posted</th>
      <th>latitude</th>
      <th>longitude</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10/10/1949 20:30</td>
      <td>san marcos</td>
      <td>tx</td>
      <td>us</td>
      <td>cylinder</td>
      <td>2700.0</td>
      <td>45 minutes</td>
      <td>This event took place in early fall around 194...</td>
      <td>4/27/2004</td>
      <td>29.883056</td>
      <td>-97.941111</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10/10/1949 21:00</td>
      <td>lackland afb</td>
      <td>tx</td>
      <td>NaN</td>
      <td>light</td>
      <td>7200.0</td>
      <td>1-2 hrs</td>
      <td>1949 Lackland AFB&amp;#44 TX.  Lights racing acros...</td>
      <td>12/16/2005</td>
      <td>29.384210</td>
      <td>-98.581082</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10/10/1955 17:00</td>
      <td>chester (uk/england)</td>
      <td>NaN</td>
      <td>gb</td>
      <td>circle</td>
      <td>20.0</td>
      <td>20 seconds</td>
      <td>Green/Orange circular disc over Chester&amp;#44 En...</td>
      <td>1/21/2008</td>
      <td>53.200000</td>
      <td>-2.916667</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10/10/1956 21:00</td>
      <td>edna</td>
      <td>tx</td>
      <td>us</td>
      <td>circle</td>
      <td>20.0</td>
      <td>1/2 hour</td>
      <td>My older brother and twin sister were leaving ...</td>
      <td>1/17/2004</td>
      <td>28.978333</td>
      <td>-96.645833</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10/10/1960 20:00</td>
      <td>kaneohe</td>
      <td>hi</td>
      <td>us</td>
      <td>light</td>
      <td>900.0</td>
      <td>15 minutes</td>
      <td>AS a Marine 1st Lt. flying an FJ4B fighter/att...</td>
      <td>1/22/2004</td>
      <td>21.418056</td>
      <td>-157.803611</td>
    </tr>
  </tbody>
</table>
</div>



2. ufos 데이터를 새로운 제목의 작은 데이터 프레임으로 변환합니다. 필드 의 고유 값을 확인하십시오.


```python
ufos = pd.DataFrame({'Seconds': ufos['duration (seconds)'], 'Country': ufos['country'],'Latitude': ufos['latitude'],'Longitude': ufos['longitude']})

ufos.Country.unique()
```




    array(['us', nan, 'gb', 'ca', 'au', 'de'], dtype=object)



3. 이제 null 값을 삭제하고 1-60초 사이의 목격만 가져와서 처리해야 하는 데이터의 양을 줄일 수 있습니다.


```python
ufos.dropna(inplace=True)

ufos = ufos[(ufos['Seconds'] >= 1) & (ufos['Seconds'] <= 60)]

ufos.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 25863 entries, 2 to 80330
    Data columns (total 4 columns):
     #   Column     Non-Null Count  Dtype  
    ---  ------     --------------  -----  
     0   Seconds    25863 non-null  float64
     1   Country    25863 non-null  object 
     2   Latitude   25863 non-null  float64
     3   Longitude  25863 non-null  float64
    dtypes: float64(3), object(1)
    memory usage: 1010.3+ KB
    

4. LabelEncoder국가의 텍스트 값을 숫자로 변환하려면 Scikit-learn의 라이브러리를 가져오세요 .


```python
from sklearn.preprocessing import LabelEncoder

ufos['Country'] = LabelEncoder().fit_transform(ufos['Country'])

ufos.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Seconds</th>
      <th>Country</th>
      <th>Latitude</th>
      <th>Longitude</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>20.0</td>
      <td>3</td>
      <td>53.200000</td>
      <td>-2.916667</td>
    </tr>
    <tr>
      <th>3</th>
      <td>20.0</td>
      <td>4</td>
      <td>28.978333</td>
      <td>-96.645833</td>
    </tr>
    <tr>
      <th>14</th>
      <td>30.0</td>
      <td>4</td>
      <td>35.823889</td>
      <td>-80.253611</td>
    </tr>
    <tr>
      <th>23</th>
      <td>60.0</td>
      <td>4</td>
      <td>45.582778</td>
      <td>-122.352222</td>
    </tr>
    <tr>
      <th>24</th>
      <td>3.0</td>
      <td>3</td>
      <td>51.783333</td>
      <td>-0.783333</td>
    </tr>
  </tbody>
</table>
</div>



데이터는 다음과 같습니다.

	Seconds	Country	Latitude	Longitude
2	20.0	3		53.200000	-2.916667
3	20.0	4		28.978333	-96.645833
14	30.0	4		35.823889	-80.253611
23	60.0	4		45.582778	-122.352222
24	3.0		3		51.783333	-0.783333

## 2.모델 구축

이제 데이터를 훈련 및 테스트 그룹으로 나누어 모델을 훈련할 준비를 할 수 있습니다.

1.X 벡터로 학습하려는 세 가지 기능을 선택하면 y 벡터는 Country를 입력하고 국가 ID를 반환할 수 있기를 원합니다 .


```python
from sklearn.model_selection import train_test_split

Selected_features = ['Seconds','Latitude','Longitude']

X = ufos[Selected_features]
y = ufos['Country']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```

2. 로지스틱 회귀를 사용하여 모델 훈련:


```python
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print(classification_report(y_test, predictions))
print('Predicted labels: ', predictions)
print('Accuracy: ', accuracy_score(y_test, predictions))
```

                  precision    recall  f1-score   support
    
               0       1.00      1.00      1.00        41
               1       0.83      0.23      0.36       250
               2       1.00      0.75      0.86         8
               3       0.98      1.00      0.99       131
               4       0.96      1.00      0.98      4743
    
        accuracy                           0.96      5173
       macro avg       0.95      0.80      0.84      5173
    weighted avg       0.96      0.96      0.95      5173
    
    Predicted labels:  [4 4 4 ... 3 4 4]
    Accuracy:  0.9601778465107288
    

    C:\Users\PC\anaconda3\lib\site-packages\sklearn\linear_model\_logistic.py:763: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    

정확도는 당연히 나쁘지 않습니다 ( 약 95%) .

## 3. 모델 '피클' 

이제 모델을 피클 할 시간입니다! 몇 줄의 코드로 이를 수행할 수 있습니다. 피클 되면 피클된 모델을 로드하고 초, 위도 및 경도 값을 포함하는 샘플 데이터 배열에 대해 테스트합니다.


```python
import pickle
model_filename = 'ufo-model.pkl'
pickle.dump(model, open(model_filename,'wb'))

model = pickle.load(open('ufo-model.pkl','rb'))
print(model.predict([[50,44,-12]]))
```

    [1]
    

## 4. Flask 앱 빌드

이제 Flask 앱을 ​​빌드하여 모델을 호출하고 유사한 결과를 반환하지만 시각적으로 더 만족스러운 방식으로 반환할 수 있습니다.

1.ufo-model.pkl 파일이 있는 notebook.ipynb 파일 옆에 web-app 이라는 폴더를 생성하여 시작 합니다.

2.그 폴더에 세 개의 폴더를 더 만듭니다. static , 안에 css 폴더가 있고 템플릿 . 이제 다음 파일과 디렉터리가 있어야 합니다.


```python
web-app/
  static/
    css/
  templates/
notebook.ipynb
ufo-model.pkl
```


3.web-app 폴더 에 생성할 첫 번째 파일 은 requirements.txt 파일입니다. JavaScript 앱의 package.json 과 마찬가지로 이 파일은 앱에 필요한 종속성을 나열합니다. requirements.txt 에 다음 행을 추가하십시오.


```python
scikit-learn
pandas
numpy
flask
```

4.이제 web-app 으로 이동하여 이 파일을 실행합니다 .


```python
cd web-app
```
    
5.터미널 유형 pip install에서 requirements.txt 에 나열된 라이브러리를 설치하려면 다음을 입력하세요 .


```python
pip install -r requirements.txt
```
    
6.이제 앱을 완성하기 위해 세 개의 파일을 더 만들 준비가 되었습니다.

-1)루트에 app.py 를 만듭니다 .
-2)템플릿 디렉토리 에 index.html 을 만듭니다 .
-3)static/css 디렉토리 에 styles.css 를 생성 합니다.

7.몇 가지 스타일로 styles.css 파일을 빌드 합니다.


```python
body {
	width: 100%;
	height: 100%;
	font-family: 'Helvetica';
	background: black;
	color: #fff;
	text-align: center;
	letter-spacing: 1.4px;
	font-size: 30px;
}

input {
	min-width: 150px;
}

.grid {
	width: 300px;
	border: 1px solid #2d2d2d;
	display: grid;
	justify-content: center;
	margin: 20px auto;
}

.box {
	color: #fff;
	background: #2d2d2d;
	padding: 12px;
	display: inline-block;
}
```

8.다음으로 index.html 파일 을 빌드 합니다.


```python
<!DOCTYPE html>
<html>
  <head>
    <meta charset="UTF-8">
    <title>🛸 UFO Appearance Prediction! 👽</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='css/styles.css') }}">
  </head>

  <body>
    <div class="grid">

      <div class="box">

        <p>According to the number of seconds, latitude and longitude, which country is likely to have reported seeing a UFO?</p>

        <form action="{{ url_for('predict')}}" method="post">
          <input type="number" name="seconds" placeholder="Seconds" required="required" min="0" max="60" />
          <input type="text" name="latitude" placeholder="Latitude" required="required" />
          <input type="text" name="longitude" placeholder="Longitude" required="required" />
          <button type="submit" class="btn">Predict country where the UFO is seen</button>
        </form>

        <p>{{ prediction_text }}</p>

      </div>

    </div>

  </body>
</html>
```


마지막으로, 모델 소비와 예측 표시를 구동하는 Python 파일을 빌드할 준비가 되었습니다.

9.추가 app.py:


```python
import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

model = pickle.load(open("./ufo-model.pkl", "rb"))


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = prediction[0]

    countries = ["Australia", "Canada", "Germany", "UK", "US"]

    return render_template(
        "index.html", prediction_text="Likely country: {}".format(countries[output])
    )


if __name__ == "__main__":
    app.run(debug=True)
```
10. 실제 구현 이후
주소:http://127.0.0.1:5000/

![화면 캡처 2022-05-19 131016](https://user-images.githubusercontent.com/103700013/169202546-35c4d364-4f66-4fff-8ff6-1505e428fba5.png)

11.실제 구현 파이참 환경

![화면 캡처 2022-05-19 131139](https://user-images.githubusercontent.com/103700013/169202707-5b5daeee-d5c9-41c0-9a3a-913286eba594.png)

    
