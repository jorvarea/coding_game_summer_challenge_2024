import math as m
import sys
from abc import ABC, abstractmethod
from typing import NamedTuple

MOVES = {0: 'UP', 1: 'LEFT', 2: 'DOWN', 3: 'RIGHT'}
MAPPING = { "UP": "U", "DOWN": "D", "LEFT": "L", "RIGHT": "R" }
MAX_ADVANTAGE = { "Hurdle": 30, "Archery": 40, "Diving": 120, "RollerSpeedSkating": 20 }
DEBUG = True

#----------------------------------------------------------------------------------------
class Minigame(ABC):
    """Abstract class representing a generic minigame framework"""

    def __init__(self, player_idx: int, nb_games: int) -> None:
        self.name: str
        self.player_idx = player_idx
        self.nb_games = nb_games
        self.gpu = ""
        self.reg: list[int] = []
        self.weights : dict[str, float] = {}
        self.advantage = 0.0
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
            self.calculate_advantage()
            self.normalize_advantage()
        else:
            self.weights = { "UP": 0, "LEFT": 0, "DOWN": 0, "RIGHT": 0 }
        self.update_turn_count()
        if DEBUG:
            print(f"Gpu: {self.gpu}, Reg: {self.reg}, Weights: {self.weights}", file=sys.stderr, flush=True)
            print(f"Advantage: {self.advantage}", file=sys.stderr, flush=True)

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
    
    def normalize_weights(self) -> None:
        """Normalize the weights so they are comparable between games"""
        min_value = min(self.weights.values())
        max_value = max(self.weights.values())
        range_weights = max_value - min_value
        if range_weights != 0:
            self.weights = {move: (weight - min_value) / range_weights for move, weight in self.weights.items()}

    @abstractmethod
    def calculate_advantage(self) -> None:
        """Calculate the advantage/disadvantage against the other players"""
        pass

    def normalize_advantage(self) -> None:
        """Normalize the advantage so its comparable between games"""
        self.advantage = (self.advantage + MAX_ADVANTAGE[self.name]) / (2 * MAX_ADVANTAGE[self.name])

#----------------------------------------------------------------------------------------

class HurdleGame(Minigame):
    def __init__(self, *args, **kwargs) -> None:
        self.name = "Hurdle"
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
                self.weights = { "UP": 2, "LEFT": 1, "DOWN": 2, "RIGHT": -3 }
            elif optimal_move == "UP":
                self.weights = { "UP": 2, "LEFT": -5, "DOWN": -5, "RIGHT": -5 }
            elif optimal_move == "RIGHT":
                self.weights = { "UP": 2, "LEFT": 1, "DOWN": 2, "RIGHT": 3 }
            elif optimal_move == "LEFT":
                self.weights = { "UP": -4, "LEFT": 1, "DOWN": -4, "RIGHT": -4 }
        else:
            self.weights = { "UP": 0, "LEFT": 0, "DOWN": 0, "RIGHT": 0 }

    def calculate_advantage(self) -> None:
        best_other_player = max(self.reg[i] - 2 * self.reg[i + 3]  
                                for i in range(3) if i != self.player_idx)
        self.advantage = (self.current_position - 2 * self.stun_timer) - best_other_player

#----------------------------------------------------------------------------------------

class Coordinates(NamedTuple):
    x: int
    y: int

class Archery(Minigame):
    def __init__(self, *args, **kwargs) -> None:
        self.name =  "Archery"
        self.pos = Coordinates(0, 0)
        super().__init__(*args, **kwargs)

    def obtain_game_specific_parameters(self) -> None:
        self.pos = Coordinates(self.reg[2 * self.player_idx], self.reg[2 * self.player_idx + 1])
        self.turns_left = len(self.gpu)
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
        self.weights = {move: -self.distance2center(new_pos) for move, new_pos in potential_moves.items()}

    def normalize_weights(self) -> None:
        super().normalize_weights()
        if self.turns_left > 10:
            self.weights = { "UP": 0, "LEFT": 0, "DOWN": 0, "RIGHT": 0 }
        elif self.turns_left >= 7:
            self.weights = {move: weight / 2 for move, weight in self.weights.items()}
        elif self.turns_left >= 4:
            self.weights = {move: weight for move, weight in self.weights.items()}
        elif self.turns_left >= 2:
            self.weights = {move: weight * 1.5 for move, weight in self.weights.items()}
        else:
            self.weights = {move: weight * 3 for move, weight in self.weights.items()}

    def calculate_advantage(self) -> None:
        player_positions = [Coordinates(self.reg[2 * i], self.reg[2 * i + 1]) for i in range(3)]
        distances2center = [self.distance2center(pos) for i, pos in enumerate(player_positions) if i != self.player_idx]
        best_other_player = min(distances2center)
        self.advantage = best_other_player - self.distance2center(self.pos)

#----------------------------------------------------------------------------------------

class Diving(Minigame):
    def __init__(self, *args, **kwargs) -> None:
        self.name = "Diving"
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
                self.weights[move] = 0

    def calculate_advantage(self) -> None:
        best_other_player = max(self.reg[i] + self.reg[i + 3] for i in range(3) if i != self.player_idx)
        self.advantage = (self.reg[self.player_idx] + self.combo) - best_other_player

#----------------------------------------------------------------------------------------

class RollerSpeedSkating(Minigame):
    def __init__(self, *args, **kwargs) -> None:
        self.name = "RollerSpeedSkating"
        super().__init__(*args, **kwargs)

    def obtain_game_specific_parameters(self) -> None:
        self.space_travelled = self.reg[self.player_idx]
        self.risk = self.reg[self.player_idx + 3]
        self.turns_left = self.reg[6]

    def calculate_weights(self) -> None:
        if self.turns_left == 1:
            for move in MOVES.values():
                index = self.gpu.find(MAPPING[move])
                if index == 0:
                    self.weights[move] = 1
                elif index == 1:
                    self.weights[move] = 2
                elif index == 2:
                    self.weights[move] = 2
                else:
                    self.weights[move] = 3
        elif self.risk >= 0:
            for move in MOVES.values():
                index = self.gpu.find(MAPPING[move])
                if index == 0:
                    self.weights[move] = 1 + 4/5 if self.risk > 0 else 1
                elif index == 1:
                    self.weights[move] = 2
                elif index == 2:
                    self.weights[move] = 2 - 4/5
                else:
                    if self.risk == 4:
                        self.weights[move] = 3 - 4/5
                    else:
                        self.weights[move] = 3 - 2 * 4/5
        else:
            self.weights = { "UP": 0, "LEFT": 0, "DOWN": 0, "RIGHT": 0 }
    
    def calculate_advantage(self) -> None:
        spaces_travelled = [self.reg[i] - 4/5 * self.reg[i + 3] if self.reg[i + 3] >= 0 else self.reg[i] - 2 * abs(self.reg[i + 3])
                            for i in range(3) if i != self.player_idx]
        self.advantage = (self.space_travelled - 4/5 * self.risk if self.risk >= 0 
                          else self.space_travelled - 2 * abs(self.risk)) - max(spaces_travelled)

#----------------------------------------------------------------------------------------

def read_game_info() -> tuple[int, int]:
    """Reads and returns the basic information about the games"""
    player_idx = int(input())
    nb_games = int(input())
    return player_idx, nb_games

def decide_move(games: list[Minigame], game_modifiers: list[float], games2win: set[int]) -> str:
    """Returns the move to play by calculating the total weigths for all the games"""
    total_weights = {}
    for i, game in enumerate(games):
        game.game_loop()
        for move in game.weights:
            if move not in total_weights:
                total_weights[move] = 0.0
            if i in games2win:
                if game.advantage >= 0:
                    total_weights[move] += game.weights[move] * 1 / (1 + abs(game.advantage)**4) * game_modifiers[i]
                else:
                    total_weights[move] += game.weights[move] * 1 / (1 + abs(game.advantage)**2) * game_modifiers[i]
    if DEBUG:
        print(f"Game modifiers: {game_modifiers}", file=sys.stderr, flush=True)
        print(f"Total weights: {total_weights}", file=sys.stderr, flush=True)
    return max(total_weights, key=lambda move: total_weights[move])

def update_game_modifiers(game_modifiers: list[float], score_info: list[int]) -> None:
    total_points = score_info[0]
    game_points = [3.0 * score_info[3 * i + 1] + score_info[3 * i + 2] for i in range(4)]
    min_points = min(game_points)
    max_points = max(game_points)
    if (max_points - min_points) != 0:
        normalized_points = [(points - min_points) / (max_points - min_points) for points in game_points]
    else:
        normalized_points = game_points
    for i, points in enumerate(normalized_points):
        if total_points > 10:
            game_modifiers[i] = 1 / (1 + points)

#----------------------------------------------------------------------------------------

def main() -> None:
    game_modifiers = [1.0, 1.0, 1.0, 1.0]
    player_idx, nb_games = read_game_info()
    games = [HurdleGame(player_idx, nb_games), Archery(player_idx, nb_games), 
             RollerSpeedSkating(player_idx, nb_games), Diving(player_idx, nb_games)]
    while True:
        for i in range(3):
            if i == player_idx:
                score_info = [int(x) for x in input().split()]
            else:
                _ = input()
        update_game_modifiers(game_modifiers, score_info)
        print(decide_move(games, game_modifiers, {0, 1, 2, 3}))

#----------------------------------------------------------------------------------------

main()