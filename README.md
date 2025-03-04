# Architectural Style Classifier

Computer Vision Fundamentals, Transfer Learning

Start with established CV techniques and pre-trained models



Data Soruces:

1. https://www.kaggle.com/datasets/jungseolin/international-architectural-styles-combined/data
2. https://www.kaggle.com/datasets/dumitrux/architectural-styles-dataset


## Example Run
First Train Run:
    python train_model.py --data_dir data/processed --batch_size 32 --num_epochs 10 --learning_rate 0.001 --architecture mobilenet

Fine Tune Run:
    python train_model.py --load_model outputs/run_datetime/best_model.pth --fine_tune --learning_rate 1e-4 --num_epochs 30 --batch_size 16 --backbone_lr_multiplier 0.1