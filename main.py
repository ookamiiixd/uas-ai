from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score
import pandas as pd

# Muat data
data = pd.read_csv('balance-scale.csv', delimiter=',', header=0)

# Hapus kolom Class
X = data.drop('Class', axis=1)
y = data['Class']

# Bagi data menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standarisasi data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model SVC
svc_model = SVC()
svc_model.fit(X_train_scaled, y_train)

# Prediksi data uji
y_pred_svc = svc_model.predict(X_test_scaled)

# Tampilkan akurasi dan presisi SVC
accuracy_svc = accuracy_score(y_test, y_pred_svc)
precision_svc = precision_score(y_test, y_pred_svc, average='weighted')
print(f"Akurasi SVC: {(accuracy_svc * 100):.2f}%")
print(f"Presisi: {precision_svc * 100:.2f}%")

# Model ANN
ann_model = MLPClassifier(
    hidden_layer_sizes=(100,), 
    max_iter=1000,            
    random_state=42
)
ann_model.fit(X_train_scaled, y_train)

# Prediksi data uji
y_pred_ann = ann_model.predict(X_test_scaled)

# Tampilkan akurasi dan presisi ANN
accuracy_ann = accuracy_score(y_test, y_pred_ann)
precision_ann = precision_score(y_test, y_pred_ann, average='weighted')
print(f"Akurasi ANN: {(accuracy_ann * 100):.2f}%")
print(f"Presisi: {precision_ann * 100:.2f}%")
