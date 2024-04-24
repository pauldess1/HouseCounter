from ultralytics import YOLO
import matplotlib.pyplot as plt
from PIL import Image

# Définir le chemin vers les données d'entraînement et le nombre d'époques
data_path = '/home/pdessain/Bureau/test/data.yaml'
epochs = 200

# Créer une instance YOLO et lancer l'entraînement
yolo = YOLO()
train_results = yolo.train(data=data_path, epochs=epochs)

# Valider les résultats après l'entraînement
valid_results = yolo.val()
print(valid_results)

# Sauvegarder les poids à la fin de l'entraînement
yolo.save('yolov8n_newtrained.pt')

