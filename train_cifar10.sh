docker run --gpus all -it --rm \
    -v `pwd`:/scratch --workdir /scratch -e HOME=/scratch \
    edm2:latest \
    python train_edm2.py \
    --outdir=.training-runs/baseline-edm2-cifar10 \
    --data=.datasets/cifar_train.zip \
    --preset=edm2-cifar10 \
    --cond=False \
    --dropout=0.13 \
    --snapshot=512Ki \
    --checkpoint=2Mi