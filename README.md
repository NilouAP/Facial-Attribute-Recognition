# Facial-Attribute-Recognition

## Project Overview
This repository is dedicated to facial attribute recognition using multi-head neural network architectures with PyTorch implemantation.

## Installation

### Prerequisites
- Python 3.8+
- PyTorch 1.9+
- torchvision
- tqdm
- matplotlib
- tensorboardx
- CUDA

### Setup

```bash
conda create -y -n att python=3.8
conda activate att
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https:/download.pytorch.org/whl/torch_stable.html
```

Install the required Python packages:
```
pip install -r requirements.txt
```

## Dataset
You can download the aligned version of Celeba dataset from [here](https://drive.google.com/file/d/1uGU4MBlsGJlSVA0CYDBJOY9TTfPfJhAD/view?usp=sharing). 

The train, validation, and test text splits of CelebA, along with the corresponding attribute labels, are available [here](https://drive.google.com/drive/folders/1H6BzFY7rBcBTx9CvHOAR1_Y9oSK8NR2Y?usp=sharing)

## Usage
### Training a Model
Use the 'main.py' script. Here's an example of how to run the training process:
```
python main.py --data ./dataset/Celeba \
    --epochs 10 --batch_size 1500 --learning_rate 0.05 \
    --workers 8
    --test-batch 1500
```

## Contributing
Contributions to this project are welcome. Please fork the repository, make your changes, and submit a pull request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
"# Face-Morphing-Attack-Detection-Benchmark" 
"# Face-Morphing-Attack-Detection-Benchmark" 
