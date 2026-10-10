from dataclasses import fields

import numpy as np

from candidate.agent.belief.generate import classify
from candidate.agent.belief.motion import SuccessorBatch, _FIELD_NOTES
from candidate.agent.geometry import bitboard
from candidate.tests.belief_support import belief_state_of, situation_of
from candidate.tests.test_reference_rules import RULES, loop_world


def real_batch():
    world = loop_world()
    state = belief_state_of(world, RULES)
    return classify(state, situation_of(state, world)).batch


def test_every_field_of_the_batch_has_a_note_and_the_description_lists_all_of_them():
    batch = real_batch()
    assert {f.name for f in fields(SuccessorBatch)} == set(_FIELD_NOTES)
    text = batch.describe()
    for f in fields(SuccessorBatch):
        assert f"  {f.name} " in text


def test_the_description_gives_structure_but_no_values():
    batch = real_batch()
    text = batch.describe()
    assert text.startswith("SuccessorBatch: 1 hypotheses x 4 moves")
    assert "ndarray(1, 4) bool" in text and "ndarray(1, 50) int64" in text and "Observation (turn 0, window" in text
    assert "int" in text.splitlines()[1] or "n " in text
    # nothing from the arrays leaks into it: no array dumps, no cell numbers of this scene
    assert "[" not in text.replace("(1, 4)", "").replace("(1, 50)", "").replace("(50,)", "").replace("(1,)", "")
    assert str(int(batch.pos[0])) not in text.replace("1 hypotheses", "")
    assert len(text.splitlines()) == len(fields(SuccessorBatch)) + 1


def test_the_constant_model_prints_the_description_not_the_batch(capsys):
    from candidate.agent.belief.motion import ConstantMotionModel

    ConstantMotionModel().score(real_batch())
    out = capsys.readouterr().out
    assert out.startswith("SuccessorBatch: 1 hypotheses") and "array([" not in out


def test_a_selection_keeps_only_the_chosen_hypotheses_and_shares_the_situation():
    from candidate.agent.data.hypotheses import HypothesisStore

    world = loop_world()
    state = belief_state_of(world, RULES)
    second = state.single(RULES.opp.respawn, RULES.opp.start_bb, bitboard.empty(), 9)
    state.store = HypothesisStore.concatenate([state.store, second])
    full = classify(state, situation_of(state, world)).batch
    assert full.n == 2
    one = full.select(np.array([1]))
    assert one.n == 1 and one.pos.tolist() == [int(full.pos[1])] and one.legal.shape == (1, 4)
    assert np.array_equal(one.dest[0], full.dest[1]) and np.array_equal(one.trail_bb[0], full.trail_bb[1])
    assert one.my_pos == full.my_pos and one.obs is full.obs and one.opp_land_bb is full.opp_land_bb
    assert full.n == 2                       # the original is untouched
    assert full.select(np.zeros(0, dtype=np.int64)).n == 0
