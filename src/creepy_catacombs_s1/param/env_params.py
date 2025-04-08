from dataclasses import dataclass

@dataclass
class EnvParams:
    """
    Basic configuration parameters for Creepy Catacombs environment.
    You can add or remove fields as needed.
    """
    width: int = 10
    height: int = 20
    tile_size: int = 64
    corridor_width: int = 8
    n_zombies: int = 0
