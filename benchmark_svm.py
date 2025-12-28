import json
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import generate_data
import os

def benchmark_accuracy():
    print("--- Benchmarking Accuracy: CNN vs SVM ---")
    
    # 1. Load Data
    # generate_data.get_processed_data() fetches from DB
    X, y = generate_data.get_processed_data()
    
    # Check if we have enough data
    if len(X) < 100:
        print("Not enough data in DB to benchmark. Generating temporary batch.")
        # Generate 200 samples on the fly if DB is empty/small
        X_temps = []
        y_temps = []
        for _ in range(100):
             X_temps.append(generate_data.apply_fft_processing(generate_data.generate_raw_signal('normal')))
             y_temps.append(0)
             X_temps.append(generate_data.apply_fft_processing(generate_data.generate_raw_signal('faulty')))
             y_temps.append(1)
        X = np.array(X_temps)
        y = np.array(y_temps)
    
    # 2. Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. SVM Baseline
    print("Training SVM...")
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    svm = SVC()
    svm.fit(X_train_flat, y_train)
    y_pred_svm = svm.predict(X_test_flat)
    acc_svm = accuracy_score(y_test, y_pred_svm)
    print(f"SVM Accuracy: {acc_svm:.4f}")
    
    # 4. Load Pre-trained CNN
    model_path = 'model.keras'
    if os.path.exists(model_path):
        print("Loading existing CNN...")
        model = tf.keras.models.load_model(model_path)
        # Evaluate
        loss, acc_cnn = model.evaluate(X_test, y_test, verbose=0)
        print(f"CNN Accuracy: {acc_cnn:.4f}")
    else:
        print("CNN model not found. Scaling accuracy to 0.")
        acc_cnn = 0.0

    # 5. Save Results
    results = {
        "svm_accuracy": round(acc_svm, 4),
        "cnn_accuracy": round(acc_cnn, 4),
        "improvement_pct": round(((acc_cnn - acc_svm)/acc_svm)*100, 2) if acc_svm > 0 else 0
    }
    
    with open("accuracy_metrics.json", "w") as f:
        json.dump(results, f, indent=4)
        print("Benchmarks saved to accuracy_metrics.json")

if __name__ == "__main__":
    benchmark_accuracy()
