Music Classification Project
Setup

with conda..

"Install Natten"
https://github.com/mir-aidj/all-in-one

pip install -r requirements.txt

may install goes to complicated.. especially natten and allin1




python gtzan.py -> download gtzan It will make Data folder
python main.py -> downbeat tracking gtzan It will make gtzan_analysis folder
python process.py -> segment gtzan It will make gtzan_preprocessed folder


Method 1: 30-second Audio

python method1_train.py --feature mel --model simple_cnn --epochs 50
python method1_train.py --feature cqt --model simple_cnn --epochs 50
python method1_train.py --feature audio --model mert --epochs 50

Method 2: 5-second Segments


python train.py --method method2_5sec --feature mel --model simple_cnn --epochs 50
python train.py --method method2_5sec --feature cqt --model simple_cnn --epochs 50
python train.py --method method2_5sec --feature audio --model mert --epochs 50

Method 3: 2-bar Segments

python train.py --method method3_2bar --feature cqt --model simple_cnn --epochs 50
python train.py --method method3_2bar --feature audio --model mert --epochs 50

Quick Options
Features: mel, cqt
Models: convnext, simple_cnn, mert -> Convnext train failed haha. we have to use convnext with pretrained=False, and it occurs very slow converges.. 
Args: --epochs, --batch_size, --learning_rate

