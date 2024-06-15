import math as m
import sys
from abc import ABC, abstractmethod
from typing import NamedTuple

MOVES = {0: 'UP', 1: 'LEFT', 2: 'DOWN', 3: 'RIGHT'}
DEBUG = True

#----------------------------------------------------------------------------------------
class Minigame(ABC):
    """Abstract class representing a generic minigame framework"""

    def __init__(self, player_idx: int, nb_games: int) -> None:
        self.player_idx = player_idx
        self.nb_games = nb_games
        self.gpu = ""
        self.reg: list[int] = []
        self.weights : dict[str, float] = {}

    @abstractmethod
    def advantage(self) -> float:
        """Calculate the advantage against the other players"""
        pass

    @abstractmethod
    def calculate_weights(self) -> None:
        """Method for calculating the weights of each move"""
        pass

    @abstractmethod
    def obtain_game_specific_parameters(self) -> None:
        pass

    def game_loop(self) -> None:
        """Reads all the inputs for the game and returns the weighted moves"""
        inputs = input().split()
        self.gpu = inputs[0]
        self.reg = [int(x) for x in inputs[1:]]
        self.obtain_game_specific_parameters()
        self.calculate_weights()
        if DEBUG:
            print(f"Gpu: {self.gpu}, Reg: {self.reg}, Weights: {self.weights}", file=sys.stderr, flush=True)

#----------------------------------------------------------------------------------------

class HurdleGame(Minigame):
    def __init__(self, *args, **kwargs) -> None:
        self.current_position: int
        self.stun_timer = 0
        super().__init__(*args, **kwargs)

    def obtain_game_specific_parameters(self) -> None:
        """Calculate specific variables relatives to the game"""
        self.current_position = self.reg[self.player_idx]
        self.stun_timer = self.reg[3 + self.player_idx]

    def count_obstacles(self) -> int:
        """Count the number of obstacles in the race track"""
        return self.gpu.count('#')
    
    def close2winning(self) -> float:
        """Check how close the player is to winning"""
        distance2win = max(self.reg[i] for i in range(2)) - self.current_position
        return 1 / (1 + distance2win)
    
    def advantage(self) -> float:
        """Calculate the advantage against the other players"""
        advantage = self.current_position - max(self.reg[i] for i in range(2) if i != self.player_idx)
        average_steps_per_turn = 2
        return advantage / average_steps_per_turn

    def calculate_spaces2obs(self) -> int:
        """Calculates the spaces until the next obstacle"""
        if self.gpu[self.current_position] != '#':
            obs_position = self.gpu[self.current_position:].find('#')
            if obs_position >= 0:
                result = obs_position - 1
            else:
                result = len(self.gpu) - (self.current_position + 1)
        else:
            obs_position = self.gpu[self.current_position+1:].find('#')
            if obs_position >= 0:
                result = obs_position
            else:
                result = len(self.gpu) - self.current_position
        return result

    def normalize_weights(self) -> None:
        """Make it so the weights are in the interval [0, 1] so we can compare them between games"""
        self.weights = {move : (weight + 8) / 11 for move, weight in self.weights.items()}

    def calculate_weights(self) -> None:
        """Calculates the weigthed moves"""
        if self.gpu != "GAME_OVER" and self.stun_timer == 0:
            spaces2obs = self.calculate_spaces2obs()
            optimal_steps = min(spaces2obs, 3)
            optimal_move = MOVES[optimal_steps]
            if optimal_move == "DOWN":
                self.weights = { "UP": 2, "LEFT": 1, "DOWN": 2, "RIGHT": -8 }
            elif optimal_move == "UP":
                self.weights = { "UP": 2, "LEFT": -8, "DOWN": -8, "RIGHT": -8 }
            elif optimal_move == "RIGHT":
                self.weights = { "UP": 2, "LEFT": 1, "DOWN": 2, "RIGHT": 3 }
            elif optimal_move == "LEFT":
                self.weights = { "UP": -8, "LEFT": 1, "DOWN": -8, "RIGHT": -8 }
        else:
            self.weights = { "UP": 0, "LEFT": 0, "DOWN": 0, "RIGHT": 0 }
        self.normalize_weights()

#----------------------------------------------------------------------------------------

class Coordinates(NamedTuple):
    x: int
    y: int

class Archery(Minigame):
    def __init__(self, *args, **kwargs) -> None:
        self.pos = Coordinates(0, 0)
        super().__init__(*args, **kwargs)

    def obtain_game_specific_parameters(self) -> None:
        """Calculate specific variables relatives to the game"""
        self.pos = Coordinates(self.reg[2 * self.player_idx], self.reg[2 * self.player_idx + 1])
        if self.gpu != "GAME_OVER":
            self.wind_strength = int(self.gpu[0])

    def distance2center(self, pos: Coordinates) -> float:
        """Calculate the distance of a position to the origin of coordinates"""
        return m.sqrt(pos.x**2 + pos.y**2)

    def close2winning(self) -> float:
        """Check how close the player is to winning"""
        player_positions = [Coordinates(self.reg[2 * i], self.reg[2 * i + 1]) for i in range(2)]
        distances2center = [self.distance2center(pos) for pos in player_positions]
        distance2win = min(distances2center) - self.distance2center(self.pos)
        return 1 / (1 + distance2win)
    
    def advantage(self) -> float:
        """Calculate the advantage against the other players"""
        player_positions = [Coordinates(self.reg[2 * i], self.reg[2 * i + 1]) for i in range(2)]
        distances2center = [self.distance2center(pos) for i, pos in enumerate(player_positions) if i != self.player_idx]
        advantage = min(distances2center) - self.distance2center(self.pos)
        average_wind_power = 4
        return advantage / average_wind_power

    def calculate_weights(self) -> None:
        """Calculates the weigthed moves"""
        if self.gpu != "GAME_OVER":
            potential_moves = {
                "UP": Coordinates(self.pos.x, self.pos.y - self.wind_strength),
                "DOWN": Coordinates(self.pos.x, self.pos.y + self.wind_strength),
                "LEFT": Coordinates(self.pos.x - self.wind_strength, self.pos.y),
                "RIGHT": Coordinates(self.pos.x + self.wind_strength, self.pos.y)
            }
            self.weights = {move: self.distance2center(new_pos) for move, new_pos in potential_moves.items()}
            self.normalize_weights()
        else:
            self.weights = { "UP": 0, "LEFT": 0, "DOWN": 0, "RIGHT": 0 }

    def normalize_weights(self) -> None:
        """Make it so the weights are in the interval [0, 1] so we can compare them between games"""
        self.weights = {move: (1 / (1 + weight)) for move, weight in self.weights.items()}

#----------------------------------------------------------------------------------------

class DummyMinigame(Minigame):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def obtain_game_specific_parameters(self) -> None:
        """Dummy implementation of obtaining game-specific parameters"""
        pass

    def close2winning(self) -> float:
        """Dummy implementation always returns 0"""
        return 0.0
    
    def advantage(self) -> float:
        return 0.0

    def calculate_weights(self) -> None:
        """Dummy implementation sets all weights to 0"""
        self.weights = { "UP": 0.0, "LEFT": 0.0, "DOWN": 0.0, "RIGHT": 0.0 }

#----------------------------------------------------------------------------------------

def read_game_info() -> tuple[int, int]:
    """Reads and returns the basic information about the games"""
    player_idx = int(input())
    nb_games = int(input())
    return player_idx, nb_games

def decide_move(games: list[Minigame], games2win: set[int]) -> str:
    """Returns the move to play by calculating the total weigths for all the games"""
    total_weights = {}
    for i, game in enumerate(games):
        game.game_loop()
        for move in game.weights:
            if move not in total_weights:
                total_weights[move] = 0.0
            if i in games2win:
                total_weights[move] += game.weights[move] * 1 / (1 + 2 * abs(game.advantage()))
            else:
                total_weights[move] += 0
    if DEBUG:
        print(f"Total weights: {total_weights}", file=sys.stderr, flush=True)
    return max(total_weights, key=lambda move: total_weights[move])

def decide_games2win(games: list[Minigame]) -> set[int]:
    evals = []
    for i, game in enumerate(games):
        game_eval = game.advantage()
        evals.append((game_eval, i))
    evals = sorted(evals)
    return {evals[-1][1], evals[-2][1]}

#----------------------------------------------------------------------------------------

def main() -> None:
    player_idx, nb_games = read_game_info()
    games = [HurdleGame(player_idx, nb_games), Archery(player_idx, nb_games), 
             DummyMinigame(player_idx, nb_games), DummyMinigame(player_idx, nb_games)]
    turn = 1
    while True:
        for _ in range(3):
            score_info = input()
        # if turn == 1:
        #     games2win = {i for i in range(nb_games)}
        # else:
        #     games2win = decide_games2win(games)
        # print(decide_move(games, games2win))
        print(decide_move(games, {0, 1}))
        turn += 1

#----------------------------------------------------------------------------------------

main()