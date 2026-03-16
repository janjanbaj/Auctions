# libs
import json
import logging
import os
import random
from collections import OrderedDict
from datetime import datetime

from joblib import Parallel, delayed

import numpy as np
import torch
from numpyencoder import NumpyEncoder
from tqdm import tqdm

# own modules
from mlca.mlca_economies import MLCA_Economies
from pysatsGurobi.pysats import PySats


# %% MECHANISM
def mechanism(
    SATS_auction_instance_seed=None,
    SATS_domain_name=None,
    Qinit=None,
    Qmax=None,
    Qround=None,
    separate_economy_training=True,
    new_query_option="MIP",
    acquisition="uUB",
    balanced_global_marginals=False,
    parallelize_training=False,
    local_scaling_factor=None,
    NN_parameters=None,
    MIP_parameters=None,
    DE_parameters=None,
    RS_parameters=None,
    scaler=None,
    init_bids_and_fitted_scaler=[None, None],
    calc_efficiency_per_iteration=False,
    isLegacy=False,
    res_path=None,
):

    print("Hello World from the mechanism")
    # Save config dict
    if res_path is not None:
        config_dict = dict(locals())

        if config_dict[
            "scaler"
        ]:  # Object of type MinMaxScaler is not JSON serializable
            config_dict["scaler"] = (
                f"MinMaxScaler(feature_range={scaler.feature_range})"
            )

        json.dump(
            config_dict,
            open(os.path.join(res_path, "config.json"), "w"),
            indent=4,
            sort_keys=False,
            separators=(", ", ": "),
            ensure_ascii=False,
            cls=NumpyEncoder,
        )

    start = datetime.now()
    # SEEDING ------------------
    np.random.seed(SATS_auction_instance_seed)
    torch.manual_seed(SATS_auction_instance_seed)
    random.seed(SATS_auction_instance_seed)
    # ---------------------------

    logging.warning("START MLCA:")
    logging.warning("-----------------------------------------------")
    logging.warning("Model: %s", SATS_domain_name)
    logging.warning("Seed SATS Instance: %s", SATS_auction_instance_seed)
    logging.warning("Qinit: %s", Qinit) if (
        init_bids_and_fitted_scaler[0] is None
    ) else logging.warning(
        "Qinit: %s", [v[0].shape[0] for k, v in init_bids_and_fitted_scaler[0].items()]
    )
    logging.warning("Qmax: %s", Qmax)
    logging.warning("Qround: %s", Qround)
    logging.warning(f"Separate economy training: {separate_economy_training}")
    logging.warning(f"Balanced global marginals: {balanced_global_marginals}")
    logging.warning(f"New query option: {new_query_option}")
    logging.warning(f"Acquisition: {acquisition}")
    if new_query_option == "RS":
        for k, v in RS_parameters.items():
            logging.warning(f"{k} in RS:{v}")
    if new_query_option == "DE":
        for k, v in DE_parameters.items():
            logging.warning(f"{k} in DE:{v}")
    logging.warning("")

    # Instantiate Economies
    logging.warning("Instantiate SATS Instance")
    if SATS_domain_name == "LSVM":
        SATS_auction_instance = PySats.getInstance().create_lsvm(
            seed=SATS_auction_instance_seed, isLegacyLSVM=isLegacy
        )  # create SATS auction instance
        logging.warning("####### ATTENTION #######")
        logging.warning("isLegacyLSVM: %s", SATS_auction_instance.isLegacy)
        logging.warning("#########################\n")
    if SATS_domain_name == "GSVM":
        SATS_auction_instance = PySats.getInstance().create_gsvm(
            seed=SATS_auction_instance_seed, isLegacyGSVM=isLegacy
        )  # create SATS auction instance
        logging.warning("####### ATTENTION #######")
        logging.warning("isLegacyGSVM: %s", SATS_auction_instance.isLegacy)
        logging.warning("#########################\n")
    if SATS_domain_name == "MRVM":
        SATS_auction_instance = PySats.getInstance().create_mrvm(
            seed=SATS_auction_instance_seed
        )  # create SATS auction instance
    if SATS_domain_name == "SRVM":
        SATS_auction_instance = PySats.getInstance().create_srvm(
            seed=SATS_auction_instance_seed
        )  # create SATS auction instance

    # create economy instance
    E = MLCA_Economies(
        SATS_auction_instance=SATS_auction_instance,
        SATS_auction_instance_seed=SATS_auction_instance_seed,
        Qinit=Qinit,
        Qmax=Qmax,
        Qround=Qround,
        scaler=scaler,
        separate_economy_training=separate_economy_training,
        new_query_option=new_query_option,
        acquisition=acquisition,
        balanced_global_marginals=balanced_global_marginals,
        parallelize_training=parallelize_training,
        local_scaling_factor=local_scaling_factor,
        start_time=start,
    )

    E.set_NN_parameters(parameters=NN_parameters)  # set NN parameters
    E.set_MIP_parameters(parameters=MIP_parameters)  # set MIP parameters
    if new_query_option == "RS":
        E.set_RS_parameters(parameters=RS_parameters)  # set RS parameters
    if new_query_option == "DE":
        E.set_DE_parameters(parameters=DE_parameters)  # set DE parameters

    # Set initial bids | Line 1-3
    init_bids, init_fitted_scaler = init_bids_and_fitted_scaler
    # use self defined inital bids | Line 1
    if init_bids is not None:
        E.set_initial_bids(initial_bids=init_bids, fitted_scaler=init_fitted_scaler)
    # create inital bundle-value pairs, uniformly sampling at random from admissible bundle space | Line 2 (now depending on isLegacy with correct sampling)
    else:
        E.set_initial_bids(seed=SATS_auction_instance_seed, include_full_bundle=True)

    # Calculate efficient allocation given current elicited bids
    if calc_efficiency_per_iteration:
        E.calculate_efficiency_per_iteration()

    # Global while loop: check if for all bidders one addtitional auction round is feasible | Line 4
    Rmax = max(E.get_number_of_elicited_bids().values())
    CHECK = Rmax <= (E.Qmax - E.Qround)
    pbar = tqdm(total=E.Qmax, desc="Auction Rounds.", initial=Rmax)
    while CHECK:
        E.mlca_iteration += 1
        # log info
        E.get_info()

        # Reset Attributes | Line 18
        logging.info("RESET: Auction Round Query Profile S=(S_1,...,S_n)")
        E.reset_current_query_profile()
        logging.info("RESET: Status of Economies")
        E.reset_economy_status()
        logging.info("RESET: NN Models")
        E.reset_NN_models()
        logging.info("RESET: Argmax Allocation")
        E.reset_argmax_allocations()
        logging.info("RESET: Allocations and SCWs")
        E.reset_scws_and_allocations()
        logging.info("RESET: Allocation Cache for Reusing\n")
        E.reset_allocation_cache_for_reusing()

        # Check Current Query Profile: | Line 5
        logging.debug("Current query profile S=(S_1,...,S_n):")
        for k, v in E.current_query_profile.items():
            logging.debug(k + ":  %s", v)

        # Fit MVNNs only once in each iteration (not for each economy separately)
        if not separate_economy_training:
            E.estimation_step()

        # Parallelized Marginal and Main Economy queries
        # ----------------------------------------------------------------------------
        
        # 1. Collect all sampled economies for all bidders
        bidder_to_marginals = OrderedDict()
        all_sampled_economies = set()
        for bidder in E.bidder_names:
            sampled_marginals = E.sample_marginal_economies_for_bidder(active_bidder=bidder)
            bidder_to_marginals[bidder] = sampled_marginals
            all_sampled_economies.update(sampled_marginals)
        
        # 2. Identify and Solve "Main" optimizations for all unique sampled economies + Main Economy
        unique_economies = list(all_sampled_economies) + ["Main Economy"]
        logging.info("PHASE 1: PARALLEL MAIN OPTIMIZATIONS for unique economies")
        E.solve_optimizations_parallel([(econ, None) for econ in unique_economies], model_type=acquisition)
        
        # 3. PHASE 2: Parallelize per-bidder query finalization
        # This handles intra-round collisions sequentially per bidder, but in parallel across bidders.
        logging.info("PHASE 2: PARALLEL PER-BIDDER QUERY FINALIZATION")
        # Use a list for jobs to ensure deterministic order if needed, though Parallel doesn't strictly need it.
        bidder_tasks = []
        for bidder in E.bidder_names:
            economies_for_bidder = bidder_to_marginals[bidder] + ["Main Economy"]
            bidder_tasks.append((bidder, economies_for_bidder))
        
        # We use the same backend="threading" to allow access to shared E state easily
        # Results are ignored as finalize_bidder_queries updates E state directly
        Parallel(n_jobs=-1, backend="threading")(
            delayed(E.finalize_bidder_queries)(bidder, econs, acquisition)
            for bidder, econs in bidder_tasks
        )

        for bidder in E.bidder_names:
            logging.info("Current query profile for %s:", bidder)
            if E.current_query_profile[bidder] is not None:
                for k in range(E.current_query_profile[bidder].shape[0]):
                    logging.info(E.current_query_profile[bidder][k, :])
            logging.info("")

        # Update Elicited Bids With Current Query Profile and check uniqueness | Line 15-16
        if not E.update_elicited_bids():
            raise RuntimeError(
                f"UNIQUENESS CHECK OF ELICITED BIDS FAILED IN ITERATION {E.mlca_iteration}, STOP MLCA!"
            )

        # Calculate efficient allocation given current elicited bids
        if calc_efficiency_per_iteration:
            efficiency = E.calculate_efficiency_per_iteration()
            if np.isclose(efficiency, 1.0, rtol=5 * 1e-5):
                logging.info("EARLY STOPPING - 100% efficiency reached.")
                break

        # Increment iteration
        pbar.update(E.Qround)
        # PER ITERATION SAVING OF RESULTS
        if res_path is not None:
            # Calculate timings
            E.calc_time_spent()
            E.save_results(res_path)

        # Set marginal efficiencies if they were calculated (only for new_query_option = 'RS' and 'DE')
        # Note E.mlca_iteration-1 since for new_query_option = 'RS' or 'DE' it used R from the previous iteration
        E.set_calc_efficiency_per_iteration_all_economies(
            mlca_iteration=E.mlca_iteration - 1
        )

        # Update while condition
        Rmax = max(E.get_number_of_elicited_bids().values())
        CHECK = Rmax <= (E.Qmax - E.Qround)

    # allocation & payments # | Line 20
    logging.info("")
    logging.info("CALCULATE ALLOCATION")
    logging.info("---------------------------------------------")
    logging.info("RESET: Allocations and SCWs\n")
    E.reset_scws_and_allocations()
    E.calculate_mlca_allocation()
    E.mlca_allocation_efficiency = E.calculate_efficiency_of_allocation(
        E.mlca_allocation, E.mlca_scw, verbose=1
    )
    # return_payments:  # | Line 21
    logging.info("")
    logging.info("CALCULATE PAYMENTS")
    logging.info("---------------------------------------------")
    E.calculate_vcg_payments()
    # Set calculated marginal efficiencies if they were calculated (only for new_query_option = 'RS' or 'DE')
    E.set_calc_efficiency_per_iteration_all_economies(mlca_iteration=E.mlca_iteration)
    # Calculate timings
    E.calc_time_spent()
    # Final Info
    E.get_info(final_summary=True)

    # FINAL SAVING OF RESULTS
    if res_path is not None:
        E.save_results(res_path)

    return
