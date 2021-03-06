## Hand Gesture Recognition

### Pyenv installation
```bash
pyenv install 3.6.0
pyenv virtualenv 3.6.0 hand-gesture-recognition
pyenv activate hand-gesture-recognition
pip install -r requirement.txt
```

### Create your own hand dataset with command:
```bash
python create_ds.py --pose stop --size 128

change poses.txt to your hand class

create your own augmentation in augment.py
```

### Train your model with command:
```bash
python cnn\model_training.py --model hand_detector --epoch 15
```

### Inference:
```bash
- load your model in the inference.py
- load your images to do inference
- python inference.py
```

### If you want to inference it with web cam on flask:
Do refer [previous post](https://github.com/quietrex/hand_gesture_recognition_flask)

