import numpy as np

from algorithm import algorithm_collection, get_algorithm
from landscape import get_landscape, landscape_collection, task_collection
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
        choices=task_collection.keys(),
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
    parser.add_argument("--datasetrange", help="bound of dataset", type=int, default=104541)
    parser.add_argument("--gplayer", help="add gplayer or not", default=False, action="store_true")

    parser.add_argument(
        "--num_rounds", help="number of query rounds", type=int, default=100
    )  ##rounds have to be smaller or equal than samples
    parser.add_argument(
        "--num_queries_per_round",
        help="number of black-box queries per round",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--num_model_queries_per_round",
        help="number of model predictions per round",
        type=int,
        default=2000,
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

    # PEX arguments
    if args.alg == "pex":
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

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    print('algorithm is:',args.alg)
    landscape, alphabet, starting_sequence = get_landscape(args)
    model = get_model(args, alphabet=alphabet, starting_sequence=starting_sequence)
    explorer = get_algorithm(args, model=model, alphabet=alphabet, starting_sequence=starting_sequence)
    runner = Runner(args)
    runner.run(landscape, starting_sequence, model, explorer, args.name, args.runs,args.task)
