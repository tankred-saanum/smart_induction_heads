python src/find_learning_heads.py --model_name Qwen/Qwen2.5-0.5B --markov_order=2 --batch_size=8
python src/find_learning_heads.py --model_name Qwen/Qwen2.5-0.5B --markov_order=3 --batch_size=8

python src/find_learning_heads.py --model_name Qwen/Qwen2.5-1.5B --markov_order=2 --batch_size=4
python src/find_learning_heads.py --model_name Qwen/Qwen2.5-1.5B --markov_order=3 --batch_size=4

python src/find_learning_heads.py --model_name Qwen/Qwen2.5-3B --markov_order=2 --batch_size=2
python src/find_learning_heads.py --model_name Qwen/Qwen2.5-3B --markov_order=3 --batch_size=2


python src/trace_nback2.py --model_name Qwen/Qwen2.5-0.5B --markov_order=2 --batch_size=8 --module=heads
python src/trace_nback2.py --model_name Qwen/Qwen2.5-0.5B --markov_order=3 --batch_size=8 --module=heads

python src/trace_nback2.py --model_name Qwen/Qwen2.5-1.5B --markov_order=2 --batch_size=4 --module=heads
python src/trace_nback2.py --model_name Qwen/Qwen2.5-1.5B --markov_order=3 --batch_size=4 --module=heads

python src/trace_nback2.py --model_name Qwen/Qwen2.5-3B --markov_order=2 --batch_size=2 --module=heads
python src/trace_nback2.py --model_name Qwen/Qwen2.5-3B --markov_order=3 --batch_size=2 --module=heads
