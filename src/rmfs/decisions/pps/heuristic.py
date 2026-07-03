"""PPS heuristic decision policies."""

from __future__ import annotations
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from model.inventory import Inventory
    from model.pod import Pod

logger = logging.getLogger(__name__)


def find_best_pod(
    universe: Inventory,
    sku_to_quantity: dict,
    relevant_skus: list,
    mode: str = "pile_on",
) -> tuple[Pod | None, int]:
    """Find the best pod candidate based on pile-on or demand mode."""
    # Step 1: Collect pod candidates from relevant skus
    pod_candidates: set[Pod] = set()
    for sku in relevant_skus:
        pod_candidates.update(universe.pod_manager.sku_to_pods.get(sku, []))

    # Step 2: Filter only idle pods
    pod_candidates = {
        pod
        for pod in pod_candidates
        if universe.pod_manager.is_idle(pod.pod_id)
        and not getattr(pod, "is_awaiting_replenishment", False)
        and not getattr(pod, "must_replenish_before_pick", False)
    }

    logger.debug("Checking candidates for mode=%s skus=%s", mode, relevant_skus)
    logger.debug("pod_candidates=%s", pod_candidates)

    if not pod_candidates:
        return None, -1

    # Step 3: Score function
    def score_pod(pod: Pod) -> int:
        score = 0
        for sku, req_qty in sku_to_quantity.items():
            if sku in pod.skus:
                current_total = pod.skus[sku]["current_qty"]
                score += min(current_total, req_qty)
        return score

    # Step 4: Rank pods by score
    ranked_pods = sorted(
        [(pod, score_pod(pod)) for pod in pod_candidates],
        key=lambda x: x[1],
        reverse=True
    )

    logger.debug("ranked_pods (mode=%s) = %s", mode, ranked_pods)
    best_pod, best_score = ranked_pods[0]
    # Reject zero-value candidates: never dispatch a pod merely because it holds
    # the SKU key when its usable quantity contributes nothing (score <= 0).
    if best_score <= 0:
        return None, best_score
    return best_pod, best_score


def find_pod_with_the_highest_pile_on(universe: Inventory, sku_to_quantity: dict) -> tuple[Pod, int]:
    """Find the pod candidate with the highest pile-on score."""
    sku_list = sku_to_quantity.keys()
    pod_candidates: set[Pod] = set()
    for sku in sku_list:
        pod_candidates.update(universe.pod_manager.sku_to_pods.get(sku, []))

    logger.debug("checking candidate for sku %s", sku_list)
    logger.debug("pod_candidates %s", pod_candidates)

    def pile_on_score(pod: Pod) -> int:
        if (
            universe.pod_manager.is_idle(pod.pod_id)
            and not getattr(pod, "is_awaiting_replenishment", False)
            and not getattr(pod, "must_replenish_before_pick", False)
        ):
            score = 0
            for sku, req_qty in sku_to_quantity.items():
                if sku in pod.skus:
                    current_total = pod.skus[sku]["current_qty"]
                    score += min(current_total, req_qty)
        else:
            score = -1
        return score

    if not pod_candidates:
        return None, -1

    ranked_pods = sorted(
        [(pod, pile_on_score(pod)) for pod in pod_candidates],
        key=lambda x: x[1],
        reverse=True
    )
    logger.debug("ranked_pods %s", ranked_pods)
    best_pod, best_score = ranked_pods[0]
    # Reject zero-value (and ineligible-only, score == -1) candidates.
    if best_score <= 0:
        return None, best_score
    return best_pod, best_score


def find_pod_with_the_highest_demand(
    universe: Inventory,
    sku_to_quantity: dict,
    station_unfinished_skus: list,
) -> tuple[Pod | None, int]:
    """Find the pod candidate with the highest demand score."""
    pod_candidates: set[Pod] = set()
    for sku in station_unfinished_skus:
        pod_candidates.update(universe.pod_manager.sku_to_pods.get(sku, []))

    pod_candidates = {
        po
        for po in pod_candidates
        if universe.pod_manager.is_idle(po.pod_id)
        and not getattr(po, "is_awaiting_replenishment", False)
        and not getattr(po, "must_replenish_before_pick", False)
    }
    logger.debug("checking candidate for sku %s", station_unfinished_skus)
    logger.debug("pod_candidates %s", pod_candidates)

    if not pod_candidates:
        return None, -1

    def demand_score(pod: Pod) -> int:
        if (
            universe.pod_manager.is_idle(pod.pod_id)
            and not getattr(pod, "is_awaiting_replenishment", False)
            and not getattr(pod, "must_replenish_before_pick", False)
        ):
            score = 0
            for sku, req_qty in sku_to_quantity.items():
                if sku in pod.skus:
                    current_total = pod.skus[sku]["current_qty"]
                    score += min(current_total, req_qty)
        else:
            score = -1
        return score

    ranked_pods = sorted(
        [(pod, demand_score(pod)) for pod in pod_candidates],
        key=lambda x: x[1],
        reverse=True
    )
    logger.debug("ranked_pods %s", ranked_pods)
    best_pod, best_score = ranked_pods[0]
    # Reject zero-value candidates so no pod is dispatched with no usable stock.
    if best_score <= 0:
        return None, best_score
    return best_pod, best_score
