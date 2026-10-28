"""The action-state behaviours: what the agent does once a state has been chosen, and when a state may or must be.

This package knows nothing about the Q model. From one frame (the observation, the memory and the belief, as a
TurnView) it works out what each of the six states would do, which of them make sense right now, and the rules that
override any scoring (after a death, PATROL; with a thin safety margin, RETURN; never a step that leaves the margin
negative). The Q model (candidate/agent/q) sits on top: it puts a value on each state, and QStrategy hands those values
to the chooser here. Imports go one way only, q -> behavior, so the states can be tested, and played with any source
of values (the hand-set stand-in, a trained network, fixed numbers in a test), without the network or its inputs.

    states.py        the six states, in the order of Q's outputs
    config.py        every constant of this layer (BehaviorConfig)

  what the board says (built once per frame; the states and Q's inputs both read it)
    snapshot.py      distance maps, the way home, the possibility mask, D1..D8, the safety margins, the world map
    worldmap.py      the whole belief as per-cell probabilities: ownership and trail layers, confidence, positions
    votes.py         per-cell probabilities from the belief's hypotheses
    paths.py         distance maps, paths and the area a path would capture
    maps.py          small (50, 50) array helpers (box sums, neighbour shifts)
    frame.py         one frame: the view, its snapshot, and the hints the strategy keeps between frames
    exits.py         where on the edge of my territory it is best to go out (PATROL walks there, EXPAND leaves from it)

  the six states: each module has valid(frame) -> bool and move(frame) -> Direction | None
    expand.py        EXPAND: out from a good nearby edge of my land, on a side whose straight trail keeps away
                     from my land, then straight on
    expand2.py       EXPAND2: the turn, sideways, keeping one side
    return_home.py   RETURN: back to my nearest land by the shortest way
    hunt.py          HUNT: to the nearest cell that probably holds the enemy's trail (the world map)
    track.py         TRACK TARGET: where a look would tell me the most, while the enemy is out of view
    patrol.py        PATROL: reposition inside my land, at the best exit
    common.py        the small helpers they share
    moves.py         the table over them: valid_states(frame) (the mask) and move_for(state, frame)

  choosing and staying safe
    chooser.py       the masks, the switching cost and the brief's two hard rules
    safety.py        the look-ahead: the safety margin a step would leave

  variety for training games
    style.py         BehaviorConfig.randomness > 0: each game draws its own constants, exit preference and EXPAND jogs
                     (0, the default and a real match's setting, changes nothing)
"""
