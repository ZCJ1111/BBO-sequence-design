import numpy as np

from algorithm import algorithm_collection, get_algorithm
from landscape import get_landscape, landscape_collection, protein_alphabet
from model import get_model, model_collection
from model.ensemble import ensemble_rules
from utils.eval_utils import Runner
from utils.os_utils import get_arg_parser


def get_args():
    parser = get_arg_parser()

    parser.add_argument("--device", help="device", type=str, default="cpu")

    # landscape arguments
    parser.add_argument(
        "--landscape",
        help="fitness landscape",
        type=str,
        default="custom",
        dest="landscape",
        choices=landscape_collection.keys(),
    )

    # algorithm arguments
    parser.add_argument(
        "--alg",
        help="exploration algorithm",
        type=str,
        default="pex",
        choices=algorithm_collection.keys(),
    )
    parser.add_argument("--name", help="algorithm name", type=str, default="pexmufac")
    parser.add_argument("--runs", help="random runs tag", type=int, default=1)
    parser.add_argument("--seed", help="random seed", type=int, default=42)

    parser.add_argument(
        "--out-dir", help="Output directory", type=str, dest="out_dir", default="./expresult"
    )
    parser.add_argument("--gplayer", help="add gplayer or not", default=False, action="store_true")

    parser.add_argument(
        "--num_rounds", help="number of query rounds", type=int, default=2
    )  # rounds have to be smaller or equal than samples
    parser.add_argument(
        "--num_queries_per_round",
        help="number of black-box queries per round",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--num_model_queries_per_round",
        help="number of model predictions per round",
        type=int,
        default=10,
    )

    # model arguments
    parser.add_argument(
        "--net",
        help="surrogate model architecture",
        type=str,
        default="esm1b",
        choices=model_collection.keys(),
    )
    parser.add_argument("--lr", help="learning rate", type=np.float32, default=1e-3)
    parser.add_argument("--batch_size", help="batch size", type=int, default=256)
    parser.add_argument(
        "--max-epochs",
        help="Stop training after this limit",
        type=int,
        default=100,
        dest="max_epochs",
    )
    parser.add_argument(
        "--patience",
        help="number of epochs without improvement to wait before terminating training",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--ensemble_size", help="number of model instances in ensemble", type=int, default=3
    )
    parser.add_argument(
        "--ensemble_rule",
        help="rule to aggregate the ensemble predictions",
        type=str,
        default="mean",
        choices=ensemble_rules.keys(),
    )
    
    args, _ = parser.parse_known_args()

    # Custom landscape arguments
    if args.landscape == "custom":
        parser.add_argument(
            "--fitness-data",
            help="CSV file of the sequences and the fitness values",
            type=str,
            dest="fitness_data",
            required=True,
        )
        parser.add_argument(
            "--sequence-column",
            help="Column in the CSV file with the amino acid sequence",
            type=str,
            dest="seq_col",
            default="CDR3",
        )
        parser.add_argument(
            "--fitness-column",
            help="Column in the CSV file with fitness values",
            type=str,
            dest="fitness_col",
            default="Energy",
        )
        parser.add_argument(
            "--starting-seq",
            help="Seed sequence to initiate the search",
            type=str,
            dest="starting_sequence",
        )
        parser.add_argument(
            "--invert-score",
            help="Use this flag if the score should be minimized",
            action="store_true",
            dest="invert_score",
        )

    # PEX arguments
    if args.alg in ("pex", "pexcons"):
        parser.add_argument(
            "--num_random_mutations",
            help="number of amino acids to mutate per sequence",
            type=int,
            default=2,
        )
        parser.add_argument(
            "--frontier_neighbor_size", help="size of the frontier neighbor", type=int, default=5
        )
    
    # MuFacNet arguments
    if args.net == "mufacnet":
        parser.add_argument(
            "--latent_dim", help="dimension of latent mutation embedding", type=int, default=32
        )
        parser.add_argument(
            "--context_radius", help="the radius of context window", type=int, default=10
        )
    
    
    if args.net == "GPmufacnet":
        parser.add_argument(
            "--latent_dim", help="dimension of latent mutation embedding", type=int, default=32
        )
        parser.add_argument(
            "--context_radius", help="the radius of context window", type=int, default=10
        )

    if args.net == "esm1b":
        parser.add_argument(
            "--torch-hub-cache",
            help="Cache directory for the ESM model checkpoint",
            type=str,
            dest="torch_hub_cache",
            default="/home/tianyu/code/pretrained_esm",
        )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    print("algorithm is:", args.alg)
    landscape, starting_sequence = get_landscape(args)
    model = get_model(args, alphabet=protein_alphabet, starting_sequence=starting_sequence)
    print(f'model is {model}')
    
    explorer = get_algorithm(
        args, model=model, alphabet=protein_alphabet, starting_sequence=starting_sequence
    )
    runner = Runner(args)
    runner.run(landscape, starting_sequence, model, explorer, args.name, args.runs, args.out_dir)



## simplified example:
    # import model
    # import algorithm
    

    # num_input_channel = 20
    # net = model.cnn.ConvolutionalNetworkModel(args,alphabet=protein_alphabet, starting_sequence=starting_sequence)
    # explorer = algorithm.pex.ProximalExploration( args, model=net, alphabet=protein_alphabet, starting_sequence=starting_sequence)

    # runner = Runner(args)
    # runner.run(landscape, starting_sequence, net, explorer, args.name, args.runs, args.out_dir)