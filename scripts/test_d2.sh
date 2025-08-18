# Read the first command-line argument into a variable
model_path="$1"
echo "$model_path"

port="$2"
echo "$port"

python3 b.py python3 benchmark_serving.py --backend openai --percentile-metrics ttft,tpot,itl,e2el --dataset-name=random  --model $model_path --ignore_eos  --served-model-name test  --host localhost --port $port --random-input-len=1 --random-output-len=2048 --metric-percentiles 99,100 --max-concurrency=1 --num-prompts=1       | tee r_1.log
python3 b.py python3 benchmark_serving.py --backend openai --percentile-metrics ttft,tpot,itl,e2el --dataset-name=random  --model $model_path --ignore_eos  --served-model-name test  --host localhost --port $port --random-input-len=1 --random-output-len=2048 --metric-percentiles 99,100 --max-concurrency=10 --num-prompts=10     | tee r_10.log
python3 b.py python3 benchmark_serving.py --backend openai --percentile-metrics ttft,tpot,itl,e2el --dataset-name=random  --model $model_path --ignore_eos  --served-model-name test  --host localhost --port $port --random-input-len=1 --random-output-len=2048 --metric-percentiles 99,100 --max-concurrency=50 --num-prompts=50     | tee r_50.log
python3 b.py python3 benchmark_serving.py --backend openai --percentile-metrics ttft,tpot,itl,e2el --dataset-name=random  --model $model_path --ignore_eos  --served-model-name test  --host localhost --port $port --random-input-len=1 --random-output-len=2048 --metric-percentiles 99,100 --max-concurrency=100 --num-prompts=100   | tee r_100.log
python3 b.py python3 benchmark_serving.py --backend openai --percentile-metrics ttft,tpot,itl,e2el --dataset-name=random  --model $model_path --ignore_eos  --served-model-name test  --host localhost --port $port --random-input-len=1 --random-output-len=2048 --metric-percentiles 99,100 --max-concurrency=256 --num-prompts=256   | tee r_256.log
python3 b.py python3 benchmark_serving.py --backend openai --percentile-metrics ttft,tpot,itl,e2el --dataset-name=random  --model $model_path --ignore_eos  --served-model-name test  --host localhost --port $port --random-input-len=1 --random-output-len=2048 --metric-percentiles 99,100 --max-concurrency=512 --num-prompts=512   | tee r_512.log
python3 b.py python3 benchmark_serving.py --backend openai --percentile-metrics ttft,tpot,itl,e2el --dataset-name=random  --model $model_path --ignore_eos  --served-model-name test  --host localhost --port $port --random-input-len=1 --random-output-len=2048 --metric-percentiles 99,100 --max-concurrency=1024 --num-prompts=1024 | tee r_1024.log

