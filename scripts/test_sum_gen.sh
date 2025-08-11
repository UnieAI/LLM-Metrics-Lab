# Read the first command-line argument into a variable
model_path="$1"
echo "$model_path"

port="$2"
echo "$port"

echo "################ summarization ################"
python s.py --tokenizer=$model_path --concurrency=1 --port=$port    | tee s_1.log
python s.py --tokenizer=$model_path --concurrency=2 --port=$port    | tee s_2.log
python s.py --tokenizer=$model_path --concurrency=4 --port=$port    | tee s_4.log
python s.py --tokenizer=$model_path --concurrency=8 --port=$port    | tee s_8.log
python s.py --tokenizer=$model_path --concurrency=16 --port=$port   | tee s_16.log
python s.py --tokenizer=$model_path --concurrency=32 --port=$port   | tee s_32.log
python s.py --tokenizer=$model_path --concurrency=64 --port=$port   | tee s_64.log
python s.py --tokenizer=$model_path --concurrency=128 --port=$port  | tee s_128.log
python s.py --tokenizer=$model_path --concurrency=256 --port=$port  | tee s_256.log
python s.py --tokenizer=$model_path --concurrency=512 --port=$port  | tee s_512.log
python s.py --tokenizer=$model_path --concurrency=1024 --port=$port | tee s_1024.log

echo "################ generation ################"
python g.py --tokenizer=$model_path --concurrency=1 --port=$port    | tee g_1.log
python g.py --tokenizer=$model_path --concurrency=2 --port=$port    | tee g_2.log
python g.py --tokenizer=$model_path --concurrency=4 --port=$port    | tee g_4.log
python g.py --tokenizer=$model_path --concurrency=8 --port=$port    | tee g_8.log
python g.py --tokenizer=$model_path --concurrency=16 --port=$port   | tee g_16.log
python g.py --tokenizer=$model_path --concurrency=32 --port=$port   | tee g_32.log
python g.py --tokenizer=$model_path --concurrency=64 --port=$port   | tee g_64.log
python g.py --tokenizer=$model_path --concurrency=128 --port=$port  | tee g_128.log
python g.py --tokenizer=$model_path --concurrency=256 --port=$port  | tee g_256.log
python g.py --tokenizer=$model_path --concurrency=512 --port=$port  | tee g_512.log
python g.py --tokenizer=$model_path --concurrency=1024 --port=$port | tee g_1024.log

