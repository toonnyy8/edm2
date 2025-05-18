sudo docker run --gpus all -it --rm --user $(id -u):$(id -g) \
    -v `pwd`:/scratch --workdir /scratch -e HOME=/scratch \
    toonnyy8/edm2:latest \
    python train_edm2.py \
    --outdir=.training-runs/baseline-edm2-img64-xs \
    --data=.datasets/img32.zip \
    --preset=edm2-img64-xs