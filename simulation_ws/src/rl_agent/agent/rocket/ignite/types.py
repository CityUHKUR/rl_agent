from collections import namedtuple


Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'reward_to_go', 'next_state', 'logp'))
