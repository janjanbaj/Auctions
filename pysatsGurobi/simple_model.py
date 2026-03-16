import warnings
import os
import json
import hashlib
import uuid
import gurobipy as gp
from gurobipy import GRB
import re

from jnius import (
    JavaClass,
    JavaMethod,
    JavaMultipleMethod,
    MetaJavaClass,
    autoclass,
    cast,
)

SizeBasedUniqueRandomXOR = autoclass(
    "org.spectrumauctions.sats.core.bidlang.xor.SizeBasedUniqueRandomXOR"
)
JavaUtilRNGSupplier = autoclass(
    "org.spectrumauctions.sats.core.util.random.JavaUtilRNGSupplier"
)
Random = autoclass("java.util.Random")
HashSet = autoclass("java.util.HashSet")
LinkedList = autoclass("java.util.LinkedList")
Bundle = autoclass("org.marketdesignresearch.mechlib.core.Bundle")
BundleEntry = autoclass("org.marketdesignresearch.mechlib.core.BundleEntry")
InstanceHandler = autoclass(
    "org.spectrumauctions.sats.core.util.instancehandling.InstanceHandler"
)
InMemoryInstanceHandler = autoclass(
    "org.spectrumauctions.sats.core.util.instancehandling.InMemoryInstanceHandler"
)
JSONInstanceHandler = autoclass(
    "org.spectrumauctions.sats.core.util.instancehandling.JSONInstanceHandler"
)
LinkedHashMap = autoclass("java.util.LinkedHashMap")
Price = autoclass("org.marketdesignresearch.mechlib.core.price.Price")
LinearPrices = autoclass("org.marketdesignresearch.mechlib.core.price.LinearPrices")


class SimpleModel(JavaClass):
    def __init__(self, seed, mip_path: str, store_files=False):
        super().__init__()
        if seed:
            rng = JavaUtilRNGSupplier(seed)
        else:
            rng = JavaUtilRNGSupplier()

        self.population = {}
        self.goods = {}
        self.seed = seed
        self.mip_path = mip_path
        self.efficient_allocation = None
        # The following sets the instance handler to InMemory (i.e., no files are stored), if store_files is false
        # Note that since InstanceHandler is a singleton, if you're running experiments in parallel, it may lead
        # to troubles to have this flag set to True in one experiment and False in another one. This is best used
        # in a consistent way, which is probably the idea anyway in most cases.
        InstanceHandler.setDefaultHandler(
            JSONInstanceHandler.getInstance()
            if store_files
            else InMemoryInstanceHandler.getInstance()
        )
        self.prepare_world()
        world = self.createWorld(rng)
        self._bidder_list = self.createPopulation(world, rng)

        # Store bidders
        bidderator = self._bidder_list.iterator()
        count = 0
        while bidderator.hasNext():
            bidder = bidderator.next()
            self.population[count] = bidder
            count += 1

        # Store goods
        goods_iterator = world.getLicenses().iterator()
        count = 0
        while goods_iterator.hasNext():
            good = goods_iterator.next()
            assert good.getLongId() == count
            count += 1
            self.goods[good.getLongId()] = good

        # Python maintains insertion order since 3.7, so it's fine to fill these dictionaries this way
        # -> https://stackoverflow.com/a/40007169

    def prepare_world(self):
        """
        Here, child classes will set the parameters for the world creation, e.g. the
        number of bidders
        """
        raise NotImplementedError("Child class has to implement this method")

    def get_model_name(self):
        raise NotImplementedError("Child class has to implement this method")

    def get_bidder_ids(self):
        return list(self.population.keys())

    def get_good_ids(self):
        return list(self.goods.keys())

    def calculate_value(self, bidder_id, goods_vector):
        bidder = self.population[bidder_id]
        bundle = self._vector_to_bundle(goods_vector)
        return bidder.calculateValue(bundle).doubleValue()

    def calculate_values(self, bidder_id, goods_vector_2D):
        bidder = self.population[bidder_id]
        values = []
        for goods_vector in goods_vector_2D:
            bundle = self._vector_to_bundle(goods_vector)
            values.append(bidder.calculateValue(bundle).doubleValue())
        return values

    def get_best_bundles(
        self, bidder_id, price_vector, max_number_of_bundles, allow_negative=False
    ):
        assert len(price_vector) == len(self.goods.keys())
        bidder = self.population[bidder_id]
        prices_map = LinkedHashMap()
        index = 0
        for good in self.goods.values():
            prices_map.put(good, Price.of(price_vector[index]))
            index += 1
        bundles = bidder.getBestBundles(
            LinearPrices(prices_map), max_number_of_bundles, allow_negative
        )
        result = []
        for bundle in bundles:
            assert bundle.areSingleQuantityGoods()
            bundle_vector = []
            for i in range(len(price_vector)):
                if bundle.contains(self.goods[i]):
                    bundle_vector.append(1)
                else:
                    bundle_vector.append(0)
            result.append(bundle_vector)
        return result

    def get_goods_of_interest(self, bidder_id):
        bidder = self.population[bidder_id]
        goods_of_interest = []
        for good_id, good in self.goods.items():
            good_set = HashSet()
            good_set.add(good)
            bundle = Bundle.of(good_set)
            if bidder.getValue(bundle, True).doubleValue() > 0:
                goods_of_interest.append(good_id)
        return goods_of_interest

    def get_uniform_random_bids(self, bidder_id, number_of_bids, seed=None):
        bidder = self.population[bidder_id]
        goods = LinkedList()
        for good in self.goods.values():
            goods.add(good)
        if seed:
            random = Random(seed)
        else:
            random = Random()

        bids = []
        for i in range(number_of_bids):
            bid = []
            bundle = bidder.getAllocationLimit().getUniformRandomBundle(random, goods)
            for good_id, good in self.goods.items():
                if bundle.contains(good):
                    bid.append(1)
                else:
                    bid.append(0)
            bid.append(bidder.getValue(bundle).doubleValue())
            bids.append(bid)
        return bids

    def get_random_bids(
        self,
        bidder_id,
        number_of_bids,
        seed=None,
        mean_bundle_size=9,
        standard_deviation_bundle_size=4.5,
    ):
        bidder = self.population[bidder_id]
        if seed:
            rng = JavaUtilRNGSupplier(seed)
        else:
            rng = JavaUtilRNGSupplier()
        valueFunction = cast(
            "org.spectrumauctions.sats.core.bidlang.xor.SizeBasedUniqueRandomXOR",
            bidder.getValueFunction(SizeBasedUniqueRandomXOR, rng),
        )
        valueFunction.setDistribution(mean_bundle_size, standard_deviation_bundle_size)
        valueFunction.setIterations(number_of_bids)
        xorBidIterator = valueFunction.iterator()
        bids = []
        while xorBidIterator.hasNext():
            bundleValue = xorBidIterator.next()
            bid = []
            for good_id, good in self.goods.items():
                if bundleValue.getBundle().contains(good):
                    bid.append(1)
                else:
                    bid.append(0)
            bid.append(bundleValue.getAmount().doubleValue())
            bids.append(bid)
        return bids

    def _get_cache_path(self):
        # Generate a unique hash for the model parameters
        params = {
            "model_name": self.get_model_name(),
            "seed": self.seed,
            "mip_path": self.mip_path,
        }
        # Add model-specific parameters if any
        for attr in ["number_of_national_bidders", "number_of_regional_bidders", "number_of_local_bidders", "isLegacy"]:
            if hasattr(self, attr):
                params[attr] = getattr(self, attr)
        
        param_str = json.dumps(params, sort_keys=True)
        param_hash = hashlib.md5(param_str.encode()).hexdigest()
        
        cache_dir = os.path.join(os.path.dirname(__file__), "cache")
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)
            
        return os.path.join(cache_dir, f"{self.get_model_name()}_{param_hash}.json")

    def get_efficient_allocation(self, display_output=False):
        if self.efficient_allocation:
            return self.efficient_allocation, sum(
                [
                    self.efficient_allocation[bidder_id]["value"]
                    for bidder_id in self.efficient_allocation.keys()
                ]
            )

        # Check Cache
        cache_path = self._get_cache_path()
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    cached_data = json.load(f)
                # JSON keys are strings, convert bidder_id back to int if helpful
                self.efficient_allocation = {int(k): v for k, v in cached_data["allocation"].items()}
                return self.efficient_allocation, cached_data["total_value"]
            except Exception as e:
                print(f"Failed to load cache from {cache_path}: {e}")

        mip_wrapper = autoclass(self.mip_path)(self._bidder_list)
        imip = mip_wrapper.getMIP()
        imip.setObjectiveMax(True)

        # Gurobi Wrapper :
        # Intercept MIP to solve using Gurobi instead of CPLEX.

        # Export IMIP to an LP file: CPLEX -> IMIP (Still Requires CPLEX) but no license.
        solver = autoclass(
            "edu.harvard.econcs.jopt.solver.server.cplex.CPlexMIPSolver"
        )()
        export_lp = os.path.abspath(f"sats_export_{uuid.uuid4().hex}.lp")
        java_path = autoclass("java.nio.file.Paths").get(export_lp)
        solver.exportToDisk(imip, java_path)

        # Solve with Gurobi
        m = gp.read(export_lp)
        m.Params.OutputFlag = 1 if display_output else 0
        m.optimize()

        # Parse allocation
        self.efficient_allocation = {}
        for bidder_id, bidder in self.population.items():
            self.efficient_allocation[bidder_id] = {"good_ids": [], "value": 0.0}

        if m.Status == GRB.OPTIMAL:
            # Propose Gurobi's solution back into PySats variables to let Java compute the values
            j_vars_map = imip.getVars()
            for vname in j_vars_map.keySet():
                g_var = m.getVarByName(vname)
                if g_var:
                    # Gurobi solution value
                    val = g_var.X
                    # Propose it back to the jopt model so the evaluator can read it
                    jvar = imip.getVar(vname)
                    # Use integer proposing for binary variables to avoid tolerance issues
                    imip.proposeValue(jvar, int(round(val)))

            # Now that the solution is loaded back into the Java `imip`,
            # we can ask SATS to evaluate the allocation natively in Java.
            # Reverse the LP mangling to populate the solution map
            HashMap = autoclass("java.util.HashMap")
            Double_J = autoclass("java.lang.Double")
            solution_map = HashMap()

            # Create a reverse lookup from de-mangled name -> original Java variable Name

            java_keys = set(j_vars_map.keySet())

            for g_var in m.getVars():
                # CPLEX LP exporter replaces brackets with parentheses and appends #NUM
                clean_name = g_var.varName.split("#")[0]
                clean_name = clean_name.replace("(", "[").replace(")", "]")

                if clean_name in java_keys:
                    solution_map.put(clean_name, Double_J(float(g_var.X)))

            allocation = mip_wrapper.adaptMIPResult(
                autoclass("edu.harvard.econcs.jopt.solver.mip.PoolSolution")(
                    float(m.ObjVal),
                    0.0,  # Relative/Absolute gap
                    solution_map,
                )
            )

            for bidder_id, bidder in self.population.items():
                self.efficient_allocation[bidder_id] = {"good_ids": []}
                bidder_allocation = allocation.allocationOf(bidder)
                good_iterator = (
                    bidder_allocation.getBundle().getSingleQuantityGoods().iterator()
                )
                while good_iterator.hasNext():
                    self.efficient_allocation[bidder_id]["good_ids"].append(
                        good_iterator.next().getLongId()
                    )
                self.efficient_allocation[bidder_id]["value"] = (
                    bidder_allocation.getValue().doubleValue()
                )

            total_value = allocation.getTotalAllocationValue().doubleValue()

        else:
            raise Exception(
                f"Gurobi failed to find an optimal allocation. Status: {m.Status}"
            )

        # Cleanup
        if os.path.exists(export_lp):
            os.remove(export_lp)

        # Save to Cache
        try:
            with open(cache_path, 'w') as f:
                json.dump({
                    "allocation": self.efficient_allocation,
                    "total_value": total_value
                }, f)
        except Exception as e:
            print(f"Failed to save cache to {cache_path}: {e}")

        return (
            self.efficient_allocation,
            total_value,
        )

    def _vector_to_bundle(self, vector):
        assert len(vector) == len(self.goods.keys())
        bundleEntries = HashSet()
        for i in range(len(vector)):
            if vector[i] == 1:
                bundleEntries.add(BundleEntry(self.goods[i], 1))
        return Bundle(bundleEntries)
