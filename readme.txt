Our fantastic induction head project!


We have a canonical set of parameters for running the experiments with a particular model (say Qwen2.5-1.5B)


$
def get_config():
    parser = ArgumentParser()

    parser.add_argument('--n_reps', default=8, type=int)
    parser.add_argument('--batch_size', default=4, type=int) # can be changed, but must be power of two
    parser.add_argument('--total_batch_size', default=32, type=int)
    parser.add_argument('--n_permute', default=4, type=int)
    parser.add_argument('--chunk_size', default=8, type=int)
    parser.add_argument('--markov_order', default=2, type=int) # can be changed for different experimental conditions
    parser.add_argument('--n_permute_primitive', default=4, type=int)
    parser.add_argument('--threshold', default=0.4, type=float) 
    parser.add_argument('--model_name', default='Qwen/Qwen2.5-1.5B', type=str)   
    args, _ = parser.parse_known_args()
    args.iters = args.total_batch_size//args.batch_size
    if args.markov_order==3:
        args.chunk_size = args.chunk_size//2

    return args
$

This gives us sequences consisting of 8 randomly ordered chunks, where the chunk properties depend on markov_order

If markov_order = 2, a chunk is a randomly permuted sequence drawn from the vocabulary
If markov_order = 3, a chunk is a randomly permuted sequence of chunks, each of which is a randomly permuted sequence of tokens