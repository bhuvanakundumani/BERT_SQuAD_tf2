


python run_squad.py --train_file squad_dataset/train-v1.1.json --predict_file squad_dataset/dev-v1.1.json --output_dir models_jun24 --overwrite_output_dir  --model_name_or_path bert-base-uncased --do_train --do_eval --do_lower_case --learning_rate 3e-5 --num_train_epochs 1 --max_seq_length 384 --doc_stride 128 
--multi-gpu