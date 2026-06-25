outdir=.out-euler #-sg-0.8-es-1.005-new

for expr_name in edm2-cifar10.new.11
do
    for kimg in 23068
    do
        for ema_rate in 0.045
        do
            (uv run python reconstruct_phema.py --indir=./.training-runs/${expr_name} \
             --outdir=.phema/${expr_name} --outstd=${ema_rate} --batch=8 --outkimg=${kimg})
            for steps in 17 19 21 23 25 27 29 31
            # for steps in 3 5 7 9 11 13 15
            do
                (uv run python generate_images.py --net=.phema/${expr_name}/phema-00${kimg}-${ema_rate}.pkl \
                 --outdir=${outdir}/${expr_name}/phema-00${kimg}-${ema_rate}/${steps}s \
                 --subdirs --seeds=0-49999 --steps=${steps} \
                 --batch=512 \
                 --eps_scaler=1. \
                 --sg=0. \
                 --sampler=euler)
                (uv run python calculate_metrics.py calc --images=${outdir}/${expr_name}/phema-00${kimg}-${ema_rate}/${steps}s \
                 --ref=.fid-refs/cifar_train.pkl \
                 --batch=256 --num=50000 \
                 --outfile=${outdir}/${expr_name}/metrics.jsonl \
                 --name=phema-00${kimg}-${ema_rate}-${steps}s \
                 --metrics=fid,fd_dinov2)
            done
        done
    done
done

