# 🌾 AgriRec - AI-Powered Crop Recommendation System  
**Decision Tree Based | Web Integrated | ML-Driven Agriculture**

---

## 📌 Project Overview

**AgriRec** is a machine learning-based crop recommendation system that empowers farmers, researchers, and agricultural experts to make smart, data-driven decisions. With just seven input parameters related to soil and environmental conditions, users receive instant predictions on the most suitable crop for cultivation.

Built with a responsive and intuitive web interface, AgriRec connects users to a trained ML backend for real-time predictions.

---

## 🌿 Dataset Information

- **Source:** Crop Recommendation Dataset  
- **Total Records:** 2,200 entries  
- **Target Classes:** 22 unique crops

### 🔢 Input Features:
- Nitrogen (N)
- Phosphorus (P)
- Potassium (K)
- Temperature (°C)
- Humidity (%)
- pH
- Rainfall (mm)

### 🛠️ Preprocessing Steps:
- Filled missing values
- Removed duplicate entries
- Standardized input features
- Ensured balanced target labels

---

## 🤖 Machine Learning Models Used

| Model               | Performance   |
|---------------------|---------------|
| Logistic Regression | Moderate      |
| SVM                 | High accuracy |
| KNN                 | Slower, okay  |
| ✅ Decision Tree     | **Best (98.18%)** |

### 🏆 Final Model Chosen:
**Decision Tree Classifier**  
- Accuracy: **98.18%**  
- High interpretability and ideal for deployment

---

## 💻 Tech Stack

- **Backend:** Python, Flask, scikit-learn, NumPy, Pandas  
- **Frontend:** HTML, CSS, JavaScript  
- **Model Development:** Jupyter Notebook / Google Colab

---

## 📁 Project Structure

```
AgriRec/
├── app.py                   # Flask app backend
├── decision_tree_model.pkl  # Trained ML model
├── templates/
│   └── index.html           # Frontend HTML form
```

---


## 🌐 Web Interface Overview

The frontend provides a modern, mobile-friendly UI with a simple form for entering soil and weather data.

### 🧾 index.html

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Agrirec: Crop Recommendation System</title>
  <style>
    body {
      background-color: #e6f5e6;
      font-family: 'Segoe UI';
      color: #1e4d2b;
      margin: 0;
      padding: 0;
    }
    .container {
      max-width: 500px;
      margin: auto;
      padding: 40px 20px;
    }
    h2 {
      text-align: center;
      color: #2e8b57;
    }
    input[type="number"] {
      padding: 12px;
      border-radius: 8px;
      border: 2px solid #6cbf6c;
      background-color: #f4fff4;
      width: 100%;
      box-sizing: border-box;
    }
    .buttons {
      display: flex;
      justify-content: center;
      gap: 20px;
      margin-top: 20px;
    }
    .buttons input {
      padding: 12px 24px;
      border-radius: 10px;
      background-color: #4CAF50;
      color: white;
      border: none;
      font-size: 1.1em;
      cursor: pointer;
    }
    .buttons input:hover {
      background-color: #45a049;
    }
    .output {
      margin-top: 30px;
      text-align: center;
      background-color: #dff0d8;
      padding: 25px;
      border-radius: 15px;
      font-size: 1.3em;
      font-weight: 600;
    }
  </style>
</head>
<body>
  <div class="container">
    <h2>Agrirec: Crop Recommendation System</h2>
    <form action="/predict" method="POST">
      <input type="number" step="any" name="N" placeholder="Nitrogen (N)">
      <input type="number" step="any" name="P" placeholder="Phosphorus (P)">
      <input type="number" step="any" name="K" placeholder="Potassium (K)">
      <input type="number" step="any" name="temperature" placeholder="Temperature (°C)">
      <input type="number" step="any" name="humidity" placeholder="Humidity (%)">
      <input type="number" step="any" name="ph" placeholder="pH Level">
      <input type="number" step="any" name="rainfall" placeholder="Rainfall (mm)">
      <div class="buttons">
        <input type="submit" value="Predict">
        <input type="button" value="Clear" onclick="clearForm()">
      </div>
    </form>
    {% if prediction %}
    <div class="output">
      🌾 <strong>Recommended Crop:</strong> {{ prediction }} <br>
      📊 <strong>Crop Level:</strong> {{ level }}
    </div>
    {% endif %}
  </div>
  <script>
    function clearForm() {
      document.querySelectorAll('input[type=number]').forEach(input => input.value = '');
      document.querySelector('.output')?.remove();
    }
  </script>
</body>
</html>

```


## 🔁 Backend Integration

### 🧠 app.py

```python
from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('decision_tree_model.pkl', 'rb'))

crop_mapping = {
    0: 'apple', 1: 'banana', 2: 'blackgram', 3: 'chickpea', 4: 'coconut',
    5: 'coffee', 6: 'cotton', 7: 'grapes', 8: 'jute', 9: 'kidneybeans',
    10: 'lentil', 11: 'maize', 12: 'mango', 13: 'mothbeans', 14: 'mungbean',
    15: 'muskmelon', 16: 'orange', 17: 'papaya', 18: 'pigeonpeas',
    19: 'pomegranate', 20: 'rice', 21: 'watermelon'
}

def get_crop_level(crop_name):
    if crop_name in ["apple", "banana", "blackgram", "chickpea", "coconut", "coffee", "cotton", "grapes", "jute"]:
        return 1
    elif crop_name in ["kidneybeans", "lentil", "maize", "mango", "mothbeans", "mungbean", "muskmelon", "orange"]:
        return 2
    else:
        return 3

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = np.array([[float(request.form['N']), float(request.form['P']), float(request.form['K']),
                          float(request.form['temperature']), float(request.form['humidity']),
                          float(request.form['ph']), float(request.form['rainfall'])]])
        prediction_idx = model.predict(data)[0]
        prediction = crop_mapping[prediction_idx]
        level = get_crop_level(prediction)
        return render_template('index.html', prediction=prediction.capitalize(), level=level)
    except Exception as e:
        return render_template('index.html', prediction=f"Error: {str(e)}", level="-")

if __name__ == "__main__":
    app.run(debug=True)

```
---

## 🧪 Example Output

**Input Values:**

- Nitrogen (N): 90  
- Phosphorus (P): 42  
- Potassium (K): 43 
- Temperature: 20.87974371°  
- Humidity: 82.00274423%  
- pH Level: 6.502985292000001
- Rainfall: 202.9355362 mm  

**Predicted Output:**

🌾 **Recommended Crop:** Rice  
📊 **Crop Level:** 3

---

## 🔮 Future Scope

- 🌦 **Integrate real-time weather API** for dynamic prediction based on local climate.
- 🌍 **Add multilingual support** to make it accessible to regional users and farmers.
- 📱 **Build a mobile app** or a Progressive Web App (PWA) for better reach.
- 📊 **Enable export of predictions** as PDF reports for offline use.
- 📡 **Connect with IoT-based sensors** to fetch real-time soil and weather data.
- 🧠 **Improve model** using deep learning for multi-crop prediction and recommendation rankings.

---

## 🌱 Conclusion

AgriRec is a powerful yet simple-to-use AI system that brings precision agriculture to life. With a user-friendly interface, reliable machine learning model, and end-to-end functionality, it bridges the gap between traditional farming and smart agriculture.

By making intelligent crop decisions easily accessible, AgriRec contributes toward sustainable farming, increased productivity, and informed agricultural planning.

> 🌾 **Grow smart, farm smart — with AgriRec!**

