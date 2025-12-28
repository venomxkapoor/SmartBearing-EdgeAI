import tensorflow as tf
import os

def convert_model():
    model_path = 'model.keras'
    tflite_path = 'model.tflite'
    
    # 1. Load Keras model
    print(f"Loading {model_path}...")
    model = tf.keras.models.load_model(model_path)
    
    # 2. Convert to TFLite
    print("Converting to TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Enable optimizations to reduce size (Dynamic Range Quantization)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    tflite_model = converter.convert()
    
    # 3. Save
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
        
    # 4. Check size
    size_bytes = os.path.getsize(tflite_path)
    size_kb = size_bytes / 1024
    print(f"TFLite Model Saved to {tflite_path}")
    print(f"File Size: {size_kb:.2f} kB")
    
    if size_kb < 50:
        print("Success: Model is under 50kB!")
    else:
        print("Note: Model is larger than 50kB (likely due to Dense layer parameters).")

if __name__ == "__main__":
    convert_model()
