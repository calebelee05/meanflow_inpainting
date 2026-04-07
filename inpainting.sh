module load conda
conda activate easy_meanflow
conda install tqdm
python /home/lee4929/meanflow_inpainting/inpainting_onestep_mf.py --outdir=/home/lee4929/meanflow_inpainting/image_experiment/out --seeds=0-3 \
   --cifar_zip=/home/lee4929/meanflow_inpainting/cifar10-32x32.zip \
   --network='/home/lee4929/meanflow_inpainting/logs/mf/MF01/00000-cifar10-32x32-uncond-ddpmpp-mf-gpus1-batch64-fp32/network-snapshot-007507.pkl'