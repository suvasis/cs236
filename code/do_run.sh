python runseq2seq_flax.py \
        --train_file /home/minasm/suvasis/tools/cs236_project/code/encoding/encodedfile_fortesting/train.tsv \
        --validation_file /home/minasm/suvasis/tools/cs236_project/code/encoding/encodedfile_fortesting/validation.tsv \
        --len_train 1858400\
        --len_eval 130000 \
        --eval_steps 1000 \
        --normalize_text \
        --output_dir output \
        --per_device_train_batch_size 56 \
        --per_device_eval_batch_size 56 \
        --preprocessing_num_workers 80 \
        --warmup_steps 5000 \
        --gradient_accumulation_steps 8 \
        --do_train \
        --do_eval \
        --adafactor \
        --num_train_epochs 6 \
        --log_model \
        --learning_rate 0.005
        
