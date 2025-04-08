from dataclasses import dataclass

@dataclass
class EnvParams:
    """
    Basic configuration parameters for Creepy Catacombs environment.
    You can add or remove fields as needed.
    """
    width: int = 7
    height: int = 8
    tile_size: int = 128
    corridor_width: int = 5
    n_zombies: int = 0
    zombie_movement: str = "random"  # Options: "random", "towards_player"