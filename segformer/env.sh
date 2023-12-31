conda create -n segformer python=3.10
conda activate segformer
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install timm einops segmentation-models-pytorch opencv-python matplotlib pandas albumentations

wget https://thinkautonomous-segmentation.s3.eu-west-3.amazonaws.com/archive.zip && unzip archive.zip
rm archive.zip 
wget https://thinkautonomous-segmentation.s3.eu-west-3.amazonaws.com/segformers.zip && unzip segformers.zip
rm segformers.zip
rm segformers/utils.py
mv segformers/* .
rm -r segformers


conda remove -n segformer --all
