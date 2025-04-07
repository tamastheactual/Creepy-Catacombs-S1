from dataclasses import dataclass

@dataclass
class EnvParams:
    """
    Basic configuration parameters for Creepy Catacombs environment.
    You can add or remove fields as needed.
    """
    width: int = 10
    height: int = 10
    tile_size: int = 32
    corridor_width: int = 3
    n_zombies: int = 2
