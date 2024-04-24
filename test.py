from ultralytics import YOLO
import matplotlib.pyplot as plt
from PIL import Image

# Charger le modèle entraîné pour l'inférence
yolo_loaded = YOLO('/home/pdessain/Bureau/test/runs/detect/train/weights/best.pt')

# Définir une fonction pour exécuter YOLO sur une image spécifique
def run_yolo(yolo, image_url, conf=0.25, iou=0.7):
    results = yolo(image_url, conf=conf, iou=iou)
    res = results[0].plot()[:, :, [2,1,0]]
    return Image.fromarray(res)

# Définir l'URL de l'image que vous souhaitez tester
image_urls = ['/home/pdessain/Bureau/test/letest.png', '/home/pdessain/Bureau/test/testt.png', '/home/pdessain/Bureau/test/eltest.png']

# Exécuter YOLO sur l'image et afficher les résultats
for image_url in image_urls : 
    result_image = run_yolo(yolo_loaded, image_url)
    result_image.show()

# Afficher les courbes de loss et de métriques pendant l'entraînement
    plt.show()
