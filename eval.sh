expr_name=baseline-edm2-cifar10
outdir=".out5"

# for kimg in 10485 15204 20447 25165 30408 35127 40370
for kimg in 0023068 # 0847249 # 0746586 0771751 0796917 0822083
do
    # (python reconstruct_phema.py --indir=./.training-runs/${expr_name} \
    #  --outdir=.phema/${expr_name} --outstd=0.01,0.015 --batch=8 --outkimg=${kimg})
    # for ema_rate in 0.010 0.030 0.050 0.070 0.090 0.110 0.130 0.150 0.170 0.190 0.210 0.230 0.250
    for ema_rate in 0.045 # 0.029 0.028 0.027 0.026
    do
        for steps in 4 8 10 12 16 # 23 31 15 # 8 # 8 # 4
        # for steps in 7 15 19 23 31 # 23 31 15 # 8 # 8 # 4
        do
            # echo "kimg: ${kimg}, ema_rate: ${ema_rate}, steps: ${steps}"
            (python generate_images.py --net=.phema/${expr_name}/phema-${kimg}-${ema_rate}.pkl --outdir=${outdir}/${expr_name}/phema-${kimg}-${ema_rate}/${steps}s --subdirs --seeds=0-49999 --steps=${steps})
            (python calculate_metrics.py calc --images=${outdir}/${expr_name}/phema-${kimg}-${ema_rate}/${steps}s \
             --ref=.fid-refs/cifar_train.pkl --outlog=${outdir}/metrics.csv \
             --batch=256 --num=50000)
        done
    done
done
