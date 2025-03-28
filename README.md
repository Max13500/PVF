## Apprentissage supervisé sur le dataset [PVF-10](https://drive.google.com/file/d/1SQq0hETXi8I3Kdq9tDAEVyZgIsRCbOah/view?usp=sharing)

Ce repo explore des techniques d'**apprentissage supervisé** (Machine Learning / Deep Learning) sur le dataset PVF-10.

PVF-10 regroupe plus de **5000 images IR de panneaux photovoltaïques** acquises par drone et classées suivant 10 catégories de défauts.
La construction du dataset est explicitée dans l'article [PVF-10: A high-resolution unmanned aerial vehicle thermal infrared image dataset for fine-grained photovoltaic fault classification](https://www.sciencedirect.com/science/article/pii/S0306261924015708)

![image](https://github.com/user-attachments/assets/bd2da503-b1a5-4b5b-b1a9-b18ab939d834)

### Premiers essais
Les 2 notebooks suivants ont été développés pour se faire une première idée de l'exploitabilité du dataset par des techniques d'apprentissage supervisé
- *[essai_ML.ipynb](essai_ML.ipynb)* : Essai avec les modèles SVM et KNN après extraction HOG des images (Histogram of Oriented Gradient)
- *[essai_DL.ipynb](essai_DL.ipynb)* : Essai avec le modèle ResNet18
  
Ces premiers essais sont prometteurs, le modèle SVM présentant une accuracy dépassant les 80% et le modèle ResNet18 dépassant les 90%, avec des bons f1-score dans l'ensemble des classes. :thumbsup:

### Utilisation du repo
1. **Dataset PVF-10**

   Après [téléchargement](https://drive.google.com/file/d/1SQq0hETXi8I3Kdq9tDAEVyZgIsRCbOah/view?usp=sharing), extraire le zip à la racine du repo
   
   Le répertoire PVF-10 doit se trouver au même niveau que les notebooks
   

3. **Dépendances**

   Les dépendances python sont listées dans [requirements.txt](requirements.txt)
   
   Rappel : pour installer ces librairies dans l'environnement de développement :
   ```bash
    pip install -r requirements.txt
   
4. **Si vous avez une GPU...**

   Le notebook [essai_DL.ipynb](essai_DL.ipynb) permet d'utiliser CUDA afin d'optimiser les calculs lors de l'entraînement du modèle de Deep Learning.
   
   Effectuer l'installation express via https://developer.nvidia.com/cuda-downloads

   Dans un terminal, taper :
   ```bash
   nvcc --version
   ```
   et noter le numero de version.
   
   Aller sur https://pytorch.org/get-started/locally/ pour récupérer la ligne de commande d’intallation des packages Pytorch adaptée à la version de CUDA installée.
   
   ![image](https://github.com/user-attachments/assets/56091f87-b26b-41a4-8477-56ea092f52d1)

   Lancer la ligne de commande dans l'environnement de développement.
   
   Vérifier dans un notebook :
   ```python
   import torch
   print(torch.cuda.is_available())
   print(torch.cuda.get_device_name(0))
   ```
   La sortie doit renvoyer True puis "NVIDIA GeForce GTX 1070" par exemple
