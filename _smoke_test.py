"""Smoke test for revised mdc_mdp_env.py — verify info schema + reproducibility.

Run with: python3 _smoke_test.py
Delete after verification (this file is not part of the project).
"""
from mdc_mdp_env import MDCMDPEnv
import numpy as np


def test_info_keys():
    env = MDCMDPEnv()
    env.reset(seed=42)
    s, r, term, trunc, info = env.step(0)
    expected = {"delay", "energy", "is_admitted", "is_dropped",
                "agent_drop", "env_drop", "completed_count"}
    missing = expected - info.keys()
    assert not missing, f"missing info keys: {missing}"
    assert term is False, "terminated must be False (continuing MDP)"
    print("[OK] info keys:", sorted(info.keys()))
    print("     sample info:", info)


def rollout(seed, n=200):
    env = MDCMDPEnv()
    env.reset(seed=seed)
    rs, env_drops, completed = [], 0, 0
    for t in range(n):
        s, r, _, _, info = env.step(t % 4)
        rs.append(r)
        env_drops += info["env_drop"]
        completed += info["completed_count"]
    return tuple(rs), env_drops, completed


def test_reproducible():
    a = rollout(42)
    b = rollout(42)
    assert a == b, "same seed should produce identical trajectory"
    c = rollout(123)
    assert c[0] != a[0], "different seed should diverge"
    print(f"[OK] reproducible: env_drops={a[1]}, completed={a[2]} (seed=42, 200 steps)")
    print(f"[OK] different seed diverges (seed=123 reward[0]={c[0][0]:.3f} vs seed=42 {a[0][0]:.3f})")


def test_env_drop_positive():
    env = MDCMDPEnv()
    env.reset(seed=42)
    total = 0
    for _ in range(2000):
        _, _, _, _, info = env.step(3)
        total += info["env_drop"]
    print(f"[OK] 2000 steps of all-drop: env_drops total = {total} (expect >> 0 due to bg arrivals)")
    assert total > 0


def test_reward_types():
    for rt in ["standard", "sparse", "cliff"]:
        env = MDCMDPEnv(reward_type=rt)
        env.reset(seed=1)
        s, r, _, _, info = env.step(0)
        print(f"[OK] reward_type={rt}: reward={r:.4f}")


if __name__ == "__main__":
    test_info_keys()
    test_reproducible()
    test_env_drop_positive()
    test_reward_types()
    print("\nAll smoke tests passed.")
