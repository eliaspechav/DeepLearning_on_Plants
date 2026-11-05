import sys
import tensorflow as tf
import numpy as np
import pandas as pd
import cv2

print("=== ENTORNO CORRECTAMENTE CONFIGURADO ===")
print(f"Versión de Python:     {sys.version.split()[0]}")
print(f"Versión de TensorFlow: {tf.__version__}")
print(f"Versión de NumPy:      {np.__version__}")

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"\n¡Éxito! TensorFlow reconoció tu tarjeta gráfica: {gpus[0].name}")
else:
    print("\nOjo: TensorFlow no detectó la GPU (revisa los drivers de NVIDIA si planeas usarla).")
