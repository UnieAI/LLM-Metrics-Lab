# Read the first command-line argument into a variable
model_path="$1"
echo "$model_path"

port="$2"
echo "$port"

echo "################ decode ################"
python d.py --tokenizer=$model_path --concurrency=1 --port=$port    | tee d_1.log
python d.py --tokenizer=$model_path --concurrency=10 --port=$port   | tee d_10.log
python d.py --tokenizer=$model_path --concurrency=50 --port=$port   | tee d_50.log
python d.py --tokenizer=$model_path --concurrency=100 --port=$port  | tee d_100.log
python d.py --tokenizer=$model_path --concurrency=256 --port=$port  | tee d_256.log
python d.py --tokenizer=$model_path --concurrency=512 --port=$port  | tee d_512.log
python d.py --tokenizer=$model_path --concurrency=1024 --port=$port | tee d_1024.log

