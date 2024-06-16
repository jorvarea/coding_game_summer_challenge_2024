import math as m
import sys
from abc import ABC, abstractmethod
from typing import NamedTuple

MOVES = {0: 'UP', 1: 'LEFT', 2: 'DOWN', 3: 'RIGHT'}
MAPPING = { "UP": "U", "DOWN": "D", "LEFT": "L", "RIGHT": "R"}
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
        self.turn = 1

    def game_loop(self) -> None:
        """Reads all the inputs for the game and returns the weighted moves"""
        inputs = input().split()
        self.gpu = inputs[0]
        self.reg = [int(x) for x in inputs[1:]]
        self.obtain_game_specific_parameters()
        if self.gpu != "GAME_OVER":
            self.calculate_weights()
            self.normalize_weights()
        else:
            self.weights = { "UP": 0, "LEFT": 0, "DOWN": 0, "RIGHT": 0 }
        self.update_turn_count()
        if DEBUG:
            print(f"Gpu: {self.gpu}, Reg: {self.reg}, Weights: {self.weights}", file=sys.stderr, flush=True)

    def update_turn_count(self) -> None:
        """Updates the turn count"""
        self.turn += 1

    @abstractmethod
    def obtain_game_specific_parameters(self) -> None:
        pass

    @abstractmethod
    def calculate_weights(self) -> None:
        """Method for calculating the weights of each move"""
        pass
    
    @abstractmethod
    def normalize_weights(self) -> None:
        """Make it so the weights are in the interval [0, 1] so we can compare them between games"""
        pass

    @abstractmethod
    def relative_advantage(self) -> float:
        """Calculate the relative advantage against the other players"""
        pass

#----------------------------------------------------------------------------------------

class HurdleGame(Minigame):
    def __init__(self, *args, **kwargs) -> None:
        self.current_position: int
        self.stun_timer = 0
        super().__init__(*args, **kwargs)

    def obtain_game_specific_parameters(self) -> None:
        self.current_position = self.reg[self.player_idx]
        self.stun_timer = self.reg[3 + self.player_idx]

    def count_obstacles(self) -> int:
        """Count the number of obstacles in the race track"""
        return self.gpu.count('#')

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

    def calculate_weights(self) -> None:
        if self.stun_timer == 0:
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

    def normalize_weights(self) -> None:
        self.weights = {move : (weight + 8) / 11 for move, weight in self.weights.items()}

    def relative_advantage(self) -> float:
        advantage = self.current_position - max(self.reg[i] for i in range(2) if i != self.player_idx)
        try:
            relative_advantage = advantage / max(self.reg[i] for i in range(2))
        except ZeroDivisionError:
            relative_advantage = advantage
        return relative_advantage

#----------------------------------------------------------------------------------------

class Coordinates(NamedTuple):
    x: int
    y: int

class Archery(Minigame):
    def __init__(self, *args, **kwargs) -> None:
        self.pos = Coordinates(0, 0)
        super().__init__(*args, **kwargs)

    def obtain_game_specific_parameters(self) -> None:
        self.pos = Coordinates(self.reg[2 * self.player_idx], self.reg[2 * self.player_idx + 1])
        if self.gpu != "GAME_OVER":
            self.wind_strength = int(self.gpu[0])

    def distance2center(self, pos: Coordinates) -> float:
        """Calculate the distance of a position to the origin of coordinates"""
        return m.sqrt(pos.x**2 + pos.y**2)

    def calculate_weights(self) -> None:
        potential_moves = {
            "UP": Coordinates(self.pos.x, self.pos.y - self.wind_strength),
            "DOWN": Coordinates(self.pos.x, self.pos.y + self.wind_strength),
            "LEFT": Coordinates(self.pos.x - self.wind_strength, self.pos.y),
            "RIGHT": Coordinates(self.pos.x + self.wind_strength, self.pos.y)
        }
        self.weights = {move: self.distance2center(new_pos) for move, new_pos in potential_moves.items()}

    def normalize_weights(self) -> None:
        self.weights = {move: (1 / (1 + weight)) for move, weight in self.weights.items()}

    def relative_advantage(self) -> float:
        player_positions = [Coordinates(self.reg[2 * i], self.reg[2 * i + 1]) for i in range(2)]
        distances2center = [self.distance2center(pos) for i, pos in enumerate(player_positions) if i != self.player_idx]
        advantage = min(distances2center) - self.distance2center(self.pos)
        try:
            relative_advantage = advantage / min(self.distance2center(pos) for pos in player_positions)
        except ZeroDivisionError:
            relative_advantage = advantage
        return relative_advantage

#----------------------------------------------------------------------------------------

class Diving(Minigame):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def obtain_game_specific_parameters(self) -> None:
        self.points = self.reg[self.player_idx]
        self.combo = self.reg[self.player_idx + 3]

    def calculate_weights(self) -> None:
        self.weights = {}
        for move in MOVES.values():
            if MAPPING[move] == self.gpu[0]:
                self.weights[move] = self.combo + 1
            else:
                self.weights[move] = - self.combo

    def normalize_weights(self) -> None:
        self.weights = {move: (weight + self.combo) / (max(self.weights.values()) + self.combo) for move, weight in self.weights.items()}

    def relative_advantage(self) -> float:
        points = [self.reg[i] for i in range(2) if i != self.player_idx]
        advantage = self.reg[self.player_idx] - max(points)
        try:
            relative_advantage = advantage /  max(self.reg[i] for i in range(2))
        except ZeroDivisionError:
            relative_advantage = 0
        return relative_advantage

#----------------------------------------------------------------------------------------

class RollerSpeedSkating(Minigame):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def obtain_game_specific_parameters(self) -> None:
        self.risk = self.reg[self.player_idx + 3]
        self.space_travelled = self.reg[self.player_idx]

    def calculate_weights(self) -> None:
        if self.risk >= 0:
            for move in MOVES.values():
                index = self.gpu.find(MAPPING[move])
                stun_penalty = 2
                risk_limit = 5
                if index == 0:
                    self.weights[move] = 1 - stun_penalty * min((self.risk - 1), 0) / risk_limit
                elif index == 1:
                    self.weights[move] = 2 - stun_penalty * self.risk / risk_limit
                elif index == 2:
                    self.weights[move] = 2 - stun_penalty * max((self.risk + 1), risk_limit) / risk_limit  
                else:
                    self.weights[move] = 3 - stun_penalty * max(self.risk + 2, risk_limit) / risk_limit
        else:
            self.weights = { "UP": 0, "LEFT": 0, "DOWN": 0, "RIGHT": 0 }

    def normalize_weights(self) -> None:
        min_value = min(self.weights.values())
        max_value = max(self.weights.values())
        range_weights = max_value - min_value
        if range_weights != 0:
            self.weights = {move: (weight - min_value) / range_weights for move, weight in self.weights.items()}
    
    def relative_advantage(self) -> float:
        spaces_travelled = [self.reg[i] for i in range(2) if i != self.player_idx]
        advantage = self.space_travelled - max(spaces_travelled)
        try:
            relative_advantage = advantage /  max(self.reg[i] for i in range(2))
        except ZeroDivisionError:
            relative_advantage = 0
        return relative_advantage

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
                advantage = game.relative_advantage()
                if advantage >= 0:
                    total_weights[move] += game.weights[move] * 1 / (1 + abs(advantage))
                else:
                    total_weights[move] += game.weights[move] * 1 / (1 + m.sqrt(abs(advantage)))
    if DEBUG:
        print(f"Total weights: {total_weights}", file=sys.stderr, flush=True)
    return max(total_weights, key=lambda move: total_weights[move])

#----------------------------------------------------------------------------------------

def main() -> None:
    player_idx, nb_games = read_game_info()
    games = [HurdleGame(player_idx, nb_games), Archery(player_idx, nb_games), 
             RollerSpeedSkating(player_idx, nb_games), Diving(player_idx, nb_games)]
    turn = 1
    while True:
        for _ in range(3):
            score_info = input()
        print(decide_move(games, {2}))
        turn += 1

#----------------------------------------------------------------------------------------

main()