#!/bin/bash

python eval/eval_medvqa.py --save_path outputs/ --num_workers 4 --eval_jsonl /home/yuexi/projects/Med-VLM-R1/data/OmniMedVQA/QA_information/Open-access/omnimedvqa_test_3k.jsonl --image_root /home/yuexi/projects/Med-VLM-R1/data/OmniMedVQA/
python eval/eval_medvqa.py --save_path outputs/ --num_workers 4 --eval_jsonl /data/yuexi/datasets/SLAKE/slake_test.jsonl
python eval/eval_medvqa.py --save_path outputs/ --num_workers 4 --eval_jsonl /data/yuexi/datasets/ImageCLEF-2019/VQAMed2019Test/VQAMed2019_Test_Questions_w_Ref_Answers.jsonl
python eval/eval_medvqa.py --save_path outputs/ --num_workers 4 --eval_jsonl /data/yuexi/datasets/VQA_RAD/vqa_rad_test.jsonl