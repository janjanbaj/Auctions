import argparse
import json
import logging
import os
import json
from collections import defaultdict
from functools import partial

# SET JAVA FOR PYCHARM
# os.environ['PATH'] += ':{}/jdk-16.0.1/bin'.format(os.path.expanduser('~'))
# os.environ['JAVA_HOME'] = '{}/jdk-16.0.1'.format(os.path.expanduser('~'))
os.environ["LD_LIBRARY_PATH"] = (
    "/opt/ibm/ILOG/CPLEX_Studio_Community2212/cplex/bin/x86-64_linux"
)
import numpy as np
from mlca_src.mlca_util import create_value_model, problem_instance_info
from sklearn.preprocessing import MinMaxScaler

network_type_to_layer_type = {"MVNN": "CALayerReLUProjected", "NN": "PlainNN"}


INCUMBENTS_to_num_train_data = {
    'GSVM': {'MVNN': 50, 'NN': 20},
    'LSVM': {'MVNN': 50, 'NN': 100},
    'MRVM': {'MVNN': 10, 'NN': 300},
    'SRVM': {'MVNN': 100, 'NN': 100}
}



import sys


def main(
    domain: str,
    num_train_data: int,
    layer_type: str,
    seed: int,
    qinit: int,
    qround: int,
    qmax: int,
):
    logging.basicConfig(
        datefmt="%H:%M:%S",
        format="%(asctime)s: %(message)s",
        handlers=[
            logging.FileHandler(
                f"{domain}-{seed}-{layer_type}.log",
            ),  # Logs to a file
        ],
    )

    value_model = create_value_model(domain)
    print("Created Value Model")

    hpo_results = json.load(open("prediction_performance_hpo_results.json", "r"))

    print("Loaded HPOs")
    # Reformatting the hpo experiments
    NN_parameters = defaultdict(dict)
    for bidder_type in problem_instance_info[value_model.name.upper()].bidder_types:
        for key, value in hpo_results[args.domain][str(num_train_data)][
            bidder_type.lower()
        ][layer_type].items():
            NN_parameters[bidder_type][key] = value
        NN_parameters[bidder_type]["layer_type"] = layer_type

        NN_parameters[bidder_type]["num_hidden_units"] = int(
            max(
                1,
                np.round(
                    NN_parameters[bidder_type]["num_neurons"]
                    / NN_parameters[bidder_type]["num_hidden_layers"]
                ),
            )
        )
        NN_parameters[bidder_type].pop("num_neurons")

    NN_parameters = value_model.parameters_to_bidder_id(NN_parameters)

    scaler = MinMaxScaler(feature_range=(0, 500))

    # Qmax, Qround, Qinit = (
    #     problem_instance_info[value_model.name.upper()].Qmax,
    #     problem_instance_info[value_model.name.upper()].Qround,
    #     problem_instance_info[value_model.name.upper()].Qinit,
    # )
    Qmax, Qround, Qinit = qmax, qround, qinit

    MIP_parameters = {
        "bigM": 2000000,
        "mip_bounds_tightening": "IA",
        "warm_start": False,
        "time_limit": 200 if domain == "MRVM" else 300,
        "relative_gap": 5e-2 if domain == "MRVM" else 1e-2,
        "integrality_tol": 1e-6,
        "attempts_DNN_WDP": 5,
    }
    res_path = os.path.join(
        "mlca_results", domain, layer_type, str(num_train_data), str(seed)
    )
    os.makedirs(res_path, exist_ok=True)
    kwargs = {
        "SATS_domain_name": value_model.name.upper(),
        "Qinit": Qinit,
        "Qmax": Qmax,
        "Qround": Qround,
        "NN_parameters": NN_parameters,
        "MIP_parameters": MIP_parameters,
        "scaler": scaler,
        "init_bids_and_fitted_scaler": [None, None],
        "return_allocation": False,
        "return_payments": True,
        "calc_efficiency_per_iteration": True,
        "isLegacy": False,
        "local_scaling_factor": None if layer_type == "PlainNN" else 1.0,
        "res_path": os.path.join(res_path, "logs.json"),
    }
    from mlca_src.mlca import mlca_mechanism

    print("Done importing MLCA Mech")
    mlca_func = partial(mlca_mechanism, **kwargs)

    # Evaluate the MLCA
    print(f"MLCA Func called for {int(seed)}")

    _, logs = mlca_func(int(seed))

    from mlca_src.mlca import NumpyEncoder
    with open(os.path.join(res_path, "logs.json"), "w") as f:
        json.dump(logs, f, cls=NumpyEncoder)

    
    logging.debug("MLCA Efficiency: {}".format(logs["MLCA Efficiency"]))
    logging.debug(
        "MLCA rel. revenue: {}".format(logs["Statistics"]["Relative Revenue"])
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prediction Performance Evaluation")
    parser.add_argument(
        "--domain",
        type=str,
        default="GSVM",
        help="SATS domain",
        choices=["GSVM", "LSVM", "SRVM", "MRVM"],
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=10000,
        help="SATS auction instance seed.",
    )
    parser.add_argument(
        "--network_type",
        type=str,
        default="MVNN",
        choices=["MVNN", "NN"],
        help="Evaluate either MVNN or NN.",
    )
    parser.add_argument(
        "--qinit",
        type=int,
        default=5,
        help="SATS Qinit",
    )
    parser.add_argument(
        "--qround",
        type=int,
        default=5,
        help="SATS Qround: the number of bundles (queries) elicited from each bidder in each round.",
    )
    parser.add_argument(
        "--qmax",
        type=int,
        default=10,
        help="SATS Qmax",
    )
    args = parser.parse_args()

    main(
        domain=args.domain,
        num_train_data=INCUMBENTS_to_num_train_data[args.domain][args.network_type],
        layer_type=network_type_to_layer_type[args.network_type],
        seed=args.seed,
        qinit=args.qinit,
        qround=args.qround,
        qmax=args.qmax,
    )
