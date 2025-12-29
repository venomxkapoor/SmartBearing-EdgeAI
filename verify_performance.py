import time
import os
import json
import numpy as np
import tensorflow as tf
import generate_data

def verify_performance():
    print("--- Measuring Edge Inference Performance ---")
    
    # 1. Load Data for Inference Test
    # We'll just generate one sample or get one from DB
    raw_sample = generate_data.generate_raw_signal('normal')
    processed_sample = generate_data.apply_fft_processing(raw_sample)
    # Shape for TFLite: (1, 500, 1) usually, but check input details
    input_data = processed_sample.reshape(1, len(processed_sample), 1).astype(np.float32)

    # 2. Measure TFLite Latency
    tflite_path = "model.tflite"
    if not os.path.exists(tflite_path):
        print("Error: model.tflite not found.")
        return

    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Warmup
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    # Benchmark Loop
    num_runs = 100
    start_time = time.time()
    for _ in range(num_runs):
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
    end_time = time.time()
    
    avg_latency_ms = ((end_time - start_time) / num_runs) * 1000
    print(f"Average Inference Latency: {avg_latency_ms:.4f} ms")
    
    # 3. Measure Model Size
    file_size_kb = os.path.getsize(tflite_path) / 1024
    print(f"Model File Size: {file_size_kb:.2f} kB")
    
    # 4. Save Results
    results = {
        "avg_latency_ms": round(avg_latency_ms, 3),
        "model_size_kb": round(file_size_kb, 2),
        "latency_target_met": avg_latency_ms < 5,
        "size_target_met": file_size_kb < 50
    }
    
    with open("performance_metrics.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print("Metrics saved to performance_metrics.json")
    
if __name__ == "__main__":
    verify_performance()
