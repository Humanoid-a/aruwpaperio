"""The Q model: the value it puts on each of the six action-states, and everything around it.

The states themselves (how each one moves, when it is valid, the hard rules and the switching cost) are in
candidate/agent/behavior, which knows nothing of Q. This package imports from there, never the other way round.

    config.py        every constant of the Q model (QConfig)
    features.py      input 1: the 22 scalars
    grid.py          input 2: the 7 x 21 x 21 egocentric grid
    probmap.py       input 3: the 10 x 10 x 10 probability map of the whole world
    inputs.py        the three built together (QInputs)
    heuristic.py     HeuristicQ, the hand-set stand-in used until a network exists
    network.py       the network (PyTorch) and NetworkQ, the trained Q as the strategy uses it
    record.py        GameRecorder: every frame of a game written down for training
    version.py       inputs_version(): which code and settings produced a record's inputs
    strategy.py      QStrategy, the Strategy the controller runs: Q's values -> behavior's chooser -> the state's move
"""
