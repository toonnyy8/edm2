# expr_name=edm2-cifar10-00023

for expr_name in edm2-cifar10.new.2
do
    # for kimg in 10485 15204 20447 25165 30408 35127 40370
    for kimg in 23068
    do
        # for ema_rate in 0.010 0.030 0.050 0.070 0.090 0.110 0.130 0.150 0.170 0.190 0.210 0.230 0.250
        for ema_rate in 0.045
        do
            (uv run python reconstruct_phema.py --indir=./.training-runs/${expr_name} \
             --outdir=.phema/${expr_name} --outstd=${ema_rate} --batch=8 --outkimg=${kimg})
            for steps in 6 8 10 12 16 16 # 8 4 # 32 # 8 4
            do
                (uv run python generate_images.py --net=.phema/${expr_name}/phema-00${kimg}-${ema_rate}.pkl --outdir=.out-base/${expr_name}/phema-00${kimg}-${ema_rate}/${steps}s --subdirs --seeds=0-49999 --steps=${steps})
                (uv run python calculate_metrics.py calc --images=.out-base/${expr_name}/phema-00${kimg}-${ema_rate}/${steps}s \
                 --ref=.fid-refs/cifar_train.pkl \
                 --batch=256 --num=50000 \
                 --outfile=.out-base/${expr_name}/metrics.txt \
                 --name=phema-00${kimg}-${ema_rate}-${steps}s)
            done
        done
    done
done
