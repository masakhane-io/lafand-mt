## mBART fine-tuning
The following is the sample script provided by [Machel Reid](https://machelreid.github.io/) for continue pre-training of mBART50

```
mv am th_TH; mv ha pt_XX; mv zu te_IN; mv yo ta_IN; mv rw hi_IN; mv sn et_EE; mv ig it_IT; mv mg sv_SE; mv ny gu_IN; mv om ro_RO; mv pcm ne_NP; mv so zh_CN; mv st pl_PL

CUDA_VISIBLE_DEVICES=0 $HOME/miniconda3/envs/py3/bin/fairseq-train \
$SHARD_STR \
--task=multilingual_denoising \
--arch=mbart_large \
--layernorm-embedding \
--decoder-learned-pos \
--encoder-normalize-before \
--decoder-normalize-before \
--encoder-learned-pos \
--langs="ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN,af_ZA,az_AZ,bn_IN,fa_IR,he_IL,hr_HR,id_ID,ka_GE,km_KH,mk_MK,ml_IN,mn_MN,mr_IN,pl_PL,ps_AF,pt_XX,sv_SE,sw_KE,ta_IN,te_IN,th_TH,tl_XX,uk_UA,ur_PK,xh_ZA,gl_ES,sl_SI" \
--training-langs="ar_AR,en_XX,fr_XX,af_ZA,sw_KE,xh_ZA,th_TH,pt_XX,it_IT,sv_SE,gu_IN,ro_RO,ne_NP,hi_IN,et_EE,zh_CN,pl_PL,ta_IN,te_IN" \
--tokens-per-sample=256 \
--max-tokens=3072 \
--max-tokens-valid=3072 \
--tensorboard-logdir=$OUTDIR/log \
--attention-dropout=0.1 \
--no-progress-bar \
--criterion=cross_entropy \
--lr-scheduler=polynomial_decay \
--skip-invalid-size-inputs-valid-test \
--update-freq=[8] \
--stop-min-lr=-1 \
--optimizer=adam \
--adam-betas="(0.9, 0.98)" \
--lr=1e-4 \
--warmup-updates=10000 \
--share-decoder-input-output-embed \
--dropout=0.05 \
--weight-decay=0.01 \
--train-subset=train \
--valid-subset=valid \
--max-update=100000 \
--save-dir=$OUTDIR/ \
--mask=0.3 \
--mask-random=0.1 \
--poisson-lambda=3.5 \
--permute-sentences=0 \
--mask-length=span-poisson \
--replace-length=1 \
--share-all-embeddings \
--layernorm-embedding \
--log-interval=10 \
--log-format=json \
--seed=1 \
--min-loss-scale=0.0001 \
--bucket-cap-mb=25 \
--optimizer-overrides={} \
--save-interval-updates=500 \
--keep-interval-updates=2 \
--validate-interval=25 \
--keep-last-epochs=-1 \
--keep-best-checkpoints=-1 \
--no-epoch-checkpoints \
--no-last-checkpoints \
--best-checkpoint-metric=loss \
--patience=-1 \
--adam-eps=1e-06 \
--power=1 \
--total-num-update=100000 \
--num-workers=2 \
--no-progress-bar \
--sample-break-mode=none \
--fp16 \
--add-lang-token \
--disable-validation    \
--encoder-embed-dim=1024 \
--encoder-ffn-embed-dim=4096 \
--encoder-layers=12 \
--encoder-attention-heads=16 \
--decoder-layers=12 --decoder-attention-heads=16 \
--decoder-embed-dim=1024 \
--decoder-ffn-embed-dim=4096 \
--reset-dataloader \
--reset-optimizer \
--reset-meters \
--reset-lr-scheduler \
--multilang-sampling-alpha 0.3 \
--restore-file=$RESTORE_FILE \

```

## Another Alternative
