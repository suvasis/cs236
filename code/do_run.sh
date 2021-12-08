python runseq2seq_flax.py \
        --train_file /home/minasm/suvasis/tools/cs236_project/code/encoding/encodedfile_fortesting/train.tsv \
        --validation_file /home/minasm/suvasis/tools/cs236_project/code/encoding/encodedfile_fortesting/validation.tsv \
        --len_train \
        --len_eval 130000 \
        --eval_steps 1000 \
        --normalize_text \
        --output_dir output \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 1 \
        --preprocessing_num_workers 1 \
        --warmup_steps 5 \
        --gradient_accumulation_steps 2 \
        --do_train \
        --do_eval \
        --adafactor \
        --num_train_epochs 6 \
        --log_model \
        --learning_rate 0.005
        
