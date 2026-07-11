from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from src.rmfs.app import netlogo_api


class FakeOrder:
    def __init__(self, order_id, demand):
        self.order_id = order_id
        self._demand = dict(demand)

    def get_remaining_skus(self):
        return dict(self._demand)


class FakePodManager:
    def __init__(self, pods):
        self.pods = pods
        self._idle = {pod.pod_id: True for pod in pods}

    def is_idle(self, pod_id):
        return self._idle.get(pod_id, False)


def station(station_id, demand, *, incoming=0, capacity=2, x=0, y=0):
    return SimpleNamespace(
        station_id=station_id,
        orders=[FakeOrder(f"order-{station_id}", demand)] if demand else [],
        incoming_pod=[object()] * incoming,
        max_robots=capacity,
        pos_x=x,
        pos_y=y,
    )


def pod(pod_id, skus, *, x=0, y=0, replenishment=False):
    return SimpleNamespace(
        pod_id=pod_id,
        skus={sku: {"current_qty": qty} for sku, qty in skus.items()},
        pos_x=x,
        pos_y=y,
        is_awaiting_replenishment=replenishment,
        must_replenish_before_pick=False,
    )


def universe(pods, stations):
    return SimpleNamespace(
        pod_manager=FakePodManager(pods),
        station_manager=SimpleNamespace(picking_stations=stations),
        pps_counters={},
    )


def test_feasibility_uses_all_skus_and_masks_wrong_or_full_station():
    candidate = pod(1, {900: 5})  # SKU 900 is outside the PPO's first 500 columns.
    wrong = station("picker-1", {20: 2})
    correct = station("picker-2", {900: 3})
    full = station("picker-3", {900: 3}, incoming=2, capacity=2)
    wh = universe([candidate], [wrong, correct, full])

    options = netlogo_api._pps_feasible_station_options(wh, candidate)

    assert [row[1].station_id for row in options] == ["picker-2"]


def test_candidate_filter_excludes_replenishment_blocked_pods():
    ready = pod(1, {900: 5})
    blocked = pod(2, {900: 5}, replenishment=True)
    wh = universe([ready, blocked], [station("picker-1", {900: 2})])

    assert netlogo_api._pps_rl_candidate_pods(wh, limit=None) == [ready]


def test_constrained_mode_corrects_zero_action_to_valid_assignment():
    candidate = pod(1, {900: 5})
    wh = universe([candidate], [station("picker-1", {900: 2})])

    with patch.object(netlogo_api, "_pps_queue_assignment", return_value=True):
        accepted = netlogo_api._execute_pps_constrained_actions(
            wh, np.array([0]), [candidate], [candidate],
        )

    assert accepted == 1
    assert wh.pps_counters["pps_raw_actions_zero"] == 1
    assert wh.pps_counters["pps_actions_corrected"] == 1
    assert wh.pps_counters["pps_constrained_assignments_accepted"] == 1


def test_constrained_mode_waits_when_no_valid_work_exists():
    candidate = pod(1, {900: 5})
    wh = universe([candidate], [station("picker-1", {20: 2})])

    accepted = netlogo_api._execute_pps_constrained_actions(
        wh, np.array([0]), [candidate], [candidate],
    )

    assert accepted == 0
    assert wh.pps_counters.get("pps_zero_progress_rounds", 0) == 0
