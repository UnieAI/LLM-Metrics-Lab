# Read the first command-line argument into a variable
model_path="$1"
echo "$model_path"

port="$2"
echo "$port"

python3 b.py python3 benchmark_serving.py --backend openai --percentile-metrics ttft,tpot,itl,e2el --dataset-name=random  --model $model_path --ignore_eos  --served-model-name test  --host localhost --port $port --random-input-len=40960 --random-output-len=2048 --metric-percentiles 99,100 --max-concurrency=1 --num-prompts=1       | tee r_1.log
python3 b.py python3 benchmark_serving.py --backend openai --percentile-metrics ttft,tpot,itl,e2el --dataset-name=random  --model $model_path --ignore_eos  --served-model-name test  --host localhost --port $port --random-input-len=40960 --random-output-len=2048 --metric-percentiles 99,100 --max-concurrency=10 --num-prompts=10     | tee r_10.log
python3 b.py python3 benchmark_serving.py --backend openai --percentile-metrics ttft,tpot,itl,e2el --dataset-name=random  --model $model_path --ignore_eos  --served-model-name test  --host localhost --port $port --random-input-len=40960 --random-output-len=2048 --metric-percentiles 99,100 --max-concurrency=50 --num-prompts=50     | tee r_50.log
python3 b.py python3 benchmark_serving.py --backend openai --percentile-metrics ttft,tpot,itl,e2el --dataset-name=random  --model $model_path --ignore_eos  --served-model-name test  --host localhost --port $port --random-input-len=40960 --random-output-len=2048 --metric-percentiles 99,100 --max-concurrency=100 --num-prompts=100   | tee r_100.log
python3 b.py python3 benchmark_serving.py --backend openai --percentile-metrics ttft,tpot,itl,e2el --dataset-name=random  --model $model_path --ignore_eos  --served-model-name test  --host localhost --port $port --random-input-len=40960 --random-output-len=2048 --metric-percentiles 99,100 --max-concurrency=256 --num-prompts=256   | tee r_256.log
python3 b.py python3 benchmark_serving.py --backend openai --percentile-metrics ttft,tpot,itl,e2el --dataset-name=random  --model $model_path --ignore_eos  --served-model-name test  --host localhost --port $port --random-input-len=40960 --random-output-len=2048 --metric-percentiles 99,100 --max-concurrency=512 --num-prompts=512   | tee r_512.log
python3 b.py python3 benchmark_serving.py --backend openai --percentile-metrics ttft,tpot,itl,e2el --dataset-name=random  --model $model_path --ignore_eos  --served-model-name test  --host localhost --port $port --random-input-len=40960 --random-output-len=2048 --metric-percentiles 99,100 --max-concurrency=1024 --num-prompts=1024 | tee r_1024.log

echo "################ hard ################"
python h.py --tokenizer=$model_path --concurrency=1 --port=$port    | tee h_1.log
python h.py --tokenizer=$model_path --concurrency=10 --port=$port   | tee h_10.log
python h.py --tokenizer=$model_path --concurrency=50 --port=$port   | tee h_50.log
python h.py --tokenizer=$model_path --concurrency=100 --port=$port  | tee h_100.log
python h.py --tokenizer=$model_path --concurrency=256 --port=$port  | tee h_256.log
python h.py --tokenizer=$model_path --concurrency=512 --port=$port  | tee h_512.log
python h.py --tokenizer=$model_path --concurrency=1024 --port=$port | tee h_1024.log

echo "################ summarization ################"
python s.py --tokenizer=$model_path --concurrency=1 --port=$port    | tee s_1.log
python s.py --tokenizer=$model_path --concurrency=10 --port=$port   | tee s_10.log
python s.py --tokenizer=$model_path --concurrency=50 --port=$port   | tee s_50.log
python s.py --tokenizer=$model_path --concurrency=100 --port=$port  | tee s_100.log
python s.py --tokenizer=$model_path --concurrency=256 --port=$port  | tee s_256.log
python s.py --tokenizer=$model_path --concurrency=512 --port=$port  | tee s_512.log
python s.py --tokenizer=$model_path --concurrency=1024 --port=$port | tee s_1024.log

echo "################ generation ################"
python g.py --tokenizer=$model_path --concurrency=1 --port=$port    | tee g_1.log
python g.py --tokenizer=$model_path --concurrency=10 --port=$port   | tee g_10.log
python g.py --tokenizer=$model_path --concurrency=50 --port=$port   | tee g_50.log
python g.py --tokenizer=$model_path --concurrency=100 --port=$port  | tee g_100.log
python g.py --tokenizer=$model_path --concurrency=256 --port=$port  | tee g_256.log
python g.py --tokenizer=$model_path --concurrency=512 --port=$port  | tee g_512.log
python g.py --tokenizer=$model_path --concurrency=1024 --port=$port | tee g_1024.log

