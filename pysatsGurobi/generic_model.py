from jnius import autoclass, cast

from .simple_model import SimpleModel

LinkedHashMap = autoclass("java.util.LinkedHashMap")
Price = autoclass("org.marketdesignresearch.mechlib.core.price.Price")
LinearPrices = autoclass("org.marketdesignresearch.mechlib.core.price.LinearPrices")


class GenericModel(SimpleModel):
    def __init__(
        self, seed, mip_path: str, generic_definition_path: str, store_files=False
    ):
        super().__init__(seed, mip_path, store_files)
        self.generic_definition_path = generic_definition_path

    def get_random_bids(
        self,
        bidder_id,
        number_of_bids,
        seed=None,
        mean_bundle_size=49,
        standard_deviation_bundle_size=24.5,
    ):
        """
        Keeping this one because of the different default values that were set.
        """
        return super().get_random_bids(
            bidder_id,
            number_of_bids,
            seed,
            mean_bundle_size,
            standard_deviation_bundle_size,
        )

    def get_efficient_allocation(self, display_output=False):
        """
        The efficient allocation is calculated on a generic definition. It is then "translated" into individual licenses that are assigned to bidders.
        Note that this does NOT result in a consistent allocation, since a single license can be assigned to multiple bidders.
        The value per bidder is still consistent, which is why this method can still be useful.
        """
        if self.efficient_allocation:
            return self.efficient_allocation, sum(
                [
                    self.efficient_allocation[bidder_id]["value"]
                    for bidder_id in self.efficient_allocation.keys()
                ]
            )

        mip_wrapper = autoclass(self.mip_path)(self._bidder_list)
        imip = mip_wrapper.getMIP()
        imip.setObjectiveMax(True)

        # Gurobi Wrapper :
        # Intercept MIP to solve using Gurobi instead of CPLEX.
        import os
        import uuid

        import gurobipy as gp
        from gurobipy import GRB

        # Export IMIP to an LP file: CPLEX -> IMIP (Still Requires CPLEX code path for export but no full license)
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
            import re

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

            available = self.get_good_ids()

            for bidder_id, bidder in self.population.items():
                self.efficient_allocation[bidder_id] = {}
                self.efficient_allocation[bidder_id]["good_ids"] = []
                if allocation.getWinners().contains(bidder):
                    bidder_allocation = allocation.allocationOf(bidder)
                    bundle_entry_iterator = (
                        bidder_allocation.getBundle().getBundleEntries().iterator()
                    )
                    while bundle_entry_iterator.hasNext():
                        bundle_entry = bundle_entry_iterator.next()
                        count = bundle_entry.getAmount()
                        licenses_iterator = (
                            cast(self.generic_definition_path, bundle_entry.getGood())
                            .containedGoods()
                            .iterator()
                        )
                        given = 0
                        while licenses_iterator.hasNext() and given < count:
                            lic_id = licenses_iterator.next().getLongId()
                            if lic_id in available:
                                self.efficient_allocation[bidder_id]["good_ids"].append(
                                    lic_id
                                )
                                available.remove(lic_id)
                                given += 1
                        assert given == count

                self.efficient_allocation[bidder_id]["value"] = (
                    bidder_allocation.getValue().doubleValue()
                    if allocation.getWinners().contains(bidder)
                    else 0.0
                )

            total_value = allocation.getTotalAllocationValue().doubleValue()

        else:
            raise Exception(
                f"Gurobi failed to find an optimal allocation. Status: {m.Status}"
            )

        # Cleanup
        if os.path.exists(export_lp):
            os.remove(export_lp)

        return (
            self.efficient_allocation,
            total_value,
        )

    def get_best_bundles(self, bidder_id, price_vector, max_number_of_bundles):
        """
        Careful: So far, it seems it was possible to sneak through PySats without caring about the fact that
        these models are generic. Generic means that there are subsets of goods that are complete comlements.
        From a set of goods (A{3}, B{2}) a bidder doesn't care if she gets the first, second or third A.
        For generic models, asking values is easy - if asked for A.1 or A.2, it will just return the same value.
        That's why it was OKish so far. For the efficient allocation above, the generic allocation was broken down
        into concrete licenses again, since it did not really matter - the value of this allocation was still consistent.

        When asking demand queries, this model breaks. A bidder of a generic world expects a price vector of length 2 in
        the example above, with a price for good A and a price for good B. If we're in a concrete-licenses world, we
        would have to ask with a price vector [price(A), price(A), price(A), price(B), price(B)]. I'm currently not sure
        where this inconsistent use of generic and concrete models would cause problems, but my fear is that it's in
        the core idea itself, while it would technically run through. So think twice if you want to use PySats for generic
        demand queries - and if you need it, we'll need to carefully think about this and implement it.

        We're not even talking about the fact that breaking down a generic model into concrete goods is highly inefficient
        and it's going a step away from the original idea of generic models - losing much of its power...
        """
        raise NotImplementedError(
            "Generic models are not yet supported for demand queries."
        )
