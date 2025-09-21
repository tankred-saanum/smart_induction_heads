#!/bin/bash
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


python src/ablate_heads.py --model_name Qwen/Qwen2.5-0.5B --markov_order=2 --batch_size=8 --ablation_style=induction
python src/ablate_heads.py --model_name Qwen/Qwen2.5-0.5B --markov_order=3 --batch_size=8

python src/ablate_heads.py --model_name Qwen/Qwen2.5-1.5B --markov_order=2 --batch_size=4 --ablation_style=induction
python src/ablate_heads.py --model_name Qwen/Qwen2.5-1.5B --markov_order=3 --batch_size=4

python src/ablate_heads.py --model_name Qwen/Qwen2.5-3B --markov_order=2 --batch_size=2 --ablation_style=induction
python src/ablate_heads.py --model_name Qwen/Qwen2.5-3B --markov_order=3 --batch_size=2




python src/ablate_heads.py --model_name Qwen/Qwen2.5-0.5B --markov_order=3 --batch_size=2 --ablation_style=one_back --threshold=0.9
python src/ablate_heads.py --model_name Qwen/Qwen2.5-0.5B --markov_order=2 --batch_size=2 --ablation_style=one_back --threshold=0.9
python src/ablate_heads.py --model_name Qwen/Qwen2.5-0.5B --markov_order=3 --batch_size=2 --ablation_style=random --threshold=0.9
python src/ablate_heads.py --model_name Qwen/Qwen2.5-0.5B --markov_order=2 --batch_size=2 --ablation_style=random --threshold=0.9


python src/ablate_heads.py --model_name Qwen/Qwen2.5-1.5B --markov_order=3 --batch_size=2 --ablation_style=one_back --threshold=0.85
python src/ablate_heads.py --model_name Qwen/Qwen2.5-1.5B --markov_order=2 --batch_size=2 --ablation_style=one_back --threshold=0.85
python src/ablate_heads.py --model_name Qwen/Qwen2.5-1.5B --markov_order=3 --batch_size=2 --ablation_style=random --threshold=0.85
python src/ablate_heads.py --model_name Qwen/Qwen2.5-1.5B --markov_order=2 --batch_size=2 --ablation_style=random --threshold=0.85




python src/ablate_heads.py --model_name Qwen/Qwen2.5-0.5B --markov_order=3 --batch_size=2 --ablation_style=one_back --threshold=0.8
python src/ablate_heads.py --model_name Qwen/Qwen2.5-0.5B --markov_order=2 --batch_size=2 --ablation_style=one_back --threshold=0.8
python src/ablate_heads.py --model_name Qwen/Qwen2.5-0.5B --markov_order=3 --batch_size=2 --ablation_style=random --threshold=0.8
python src/ablate_heads.py --model_name Qwen/Qwen2.5-0.5B --markov_order=2 --batch_size=2 --ablation_style=random --threshold=0.8
python src/ablate_heads.py --model_name Qwen/Qwen2.5-0.5B --markov_order=3 --batch_size=2 --ablation_style=one_back --threshold=0.75
python src/ablate_heads.py --model_name Qwen/Qwen2.5-0.5B --markov_order=2 --batch_size=2 --ablation_style=one_back --threshold=0.75
python src/ablate_heads.py --model_name Qwen/Qwen2.5-0.5B --markov_order=3 --batch_size=2 --ablation_style=random --threshold=0.75
python src/ablate_heads.py --model_name Qwen/Qwen2.5-0.5B --markov_order=2 --batch_size=2 --ablation_style=random --threshold=0.75


python plotting/plot_ablation3.py --ablation_style=random --model_name=Qwen/Qwen2.5-0.5B --threshold=0.8 --markov_order=2
python plotting/plot_ablation3.py --ablation_style=random --model_name=Qwen/Qwen2.5-0.5B --threshold=0.8 --markov_order=3
python plotting/plot_ablation3.py --ablation_style=one_back --model_name=Qwen/Qwen2.5-0.5B --threshold=0.8 --markov_order=2
python plotting/plot_ablation3.py --ablation_style=one_back --model_name=Qwen/Qwen2.5-0.5B --threshold=0.8 --markov_order=3

python plotting/plot_ablation3.py --ablation_style=random --model_name=Qwen/Qwen2.5-0.5B --threshold=0.75 --markov_order=2
python plotting/plot_ablation3.py --ablation_style=random --model_name=Qwen/Qwen2.5-0.5B --threshold=0.75 --markov_order=3
python plotting/plot_ablation3.py --ablation_style=one_back --model_name=Qwen/Qwen2.5-0.5B --threshold=0.75 --markov_order=2
python plotting/plot_ablation3.py --ablation_style=one_back --model_name=Qwen/Qwen2.5-0.5B --threshold=0.75 --markov_order=3



python src/ablate_heads.py --model_name Qwen/Qwen2.5-1.5B --markov_order=2 --batch_size=2 --ablation_style=induction --threshold=0.4
python src/ablate_heads.py --model_name Qwen/Qwen2.5-1.5B --markov_order=2 --batch_size=2 --ablation_style=random_induction --threshold=0.4
python src/ablate_heads.py --model_name Qwen/Qwen2.5-1.5B --markov_order=3 --batch_size=2 --ablation_style=induction --threshold=0.4
python src/ablate_heads.py --model_name Qwen/Qwen2.5-1.5B --markov_order=3 --batch_size=2 --ablation_style=random_induction --threshold=0.4

python src/ablate_heads.py --model_name Qwen/Qwen2.5-3B --markov_order=2 --batch_size=2 --ablation_style=induction --threshold=0.4
python src/ablate_heads.py --model_name Qwen/Qwen2.5-3B --markov_order=2 --batch_size=2 --ablation_style=random_induction --threshold=0.4
python src/ablate_heads.py --model_name Qwen/Qwen2.5-3B --markov_order=3 --batch_size=2 --ablation_style=induction --threshold=0.4
python src/ablate_heads.py --model_name Qwen/Qwen2.5-3B --markov_order=3 --batch_size=2 --ablation_style=random_induction --threshold=0.4