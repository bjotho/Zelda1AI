"""An OpenAI Gym environment for The Legend of Zelda."""
import collections
import os
from nes_py import NESEnv
import numpy as np
import math


# the directory that houses this module
MODULE_DIR = os.path.dirname(os.path.abspath(__file__))


# the path to the Zelda 1 ROM
ROM_PATH = os.path.join(MODULE_DIR, '_roms', 'Zelda_1.nes')


# a mapping of numeric values to cardinal directions
DIRECTIONS = collections.defaultdict(lambda: None, {
    0x08: 'N',
    0x04: 'S',
    0x01: 'E',
    0x02: 'W',
})


# the set of game modes that indicate a scroll is in progress
SCROLL_GAME_MODES = {0x4, 0x6, 0x7}


# a mapping of numeric values to string types for pulse 1
PULSE_1_IM_TYPES = collections.defaultdict(lambda: None, {
    0x80: None, # this value is unknown
    0x40: "1 Heart Warning",
    0x20: "Set Bomb",
    0x10: "Small Heart Pickup",
    0x08: "Key Pickup",
    0x04: "Magic Cast",
    0x02: "Boomerang Stun",
    0x01: "Arrow Deflected",
})


# a mapping of numeric values to string types for pulse 2
PULSE_2_IM_TYPES = collections.defaultdict(lambda: None, {
    0x80: "Death Spiral",
    0x40: "Continue Screen",
    0x20: "Enemy Burst",
    0x10: "Whistle",
    0x08: "Bomb Pickup",
    0x04: "Secret Revealed",
    0x02: "Key Appears",
    0x01: "Rupee Pickup",
})


# a mapping of numeric values to sword types
SWORD_TYPES = collections.defaultdict(lambda: None, {
    0x00: "None",
    0x01: "Sword",
    0x02: "White Sword",
    0x03: "Magical Sword",
})


# the type of arrows in Link's inventory
ARROWS_TYPES = collections.defaultdict(lambda: None, {
    0x00: "None",
    0x01: "Arrow",
    0x02: "Silver Arrow",
})


# the type of candle in Link's inventory
CANDLE_TYPES = collections.defaultdict(lambda: None, {
    0x00: "None",
    0x01: "Blue Candle",
    0x02: "Red Candle",
})


# the type of potion in Link's inventory
POTION_TYPES = collections.defaultdict(lambda: None, {
    0x00: "None",
    0x01: "Life Potion",
    0x02: "2nd Potion",
})


# the type of ring in Link's inventory
RING_TYPES = collections.defaultdict(lambda: None, {
    0x00: "None",
    0x01: "Blue Ring",
    0x02: "Red Ring",
})

# The type of objective currently assigned
OBJECTIVE_TYPE = collections.defaultdict(lambda: None, {
    0: "Get to location",
    1: "Kill enemies"
})


class Zelda1Env(NESEnv):
    """An environment for playing The Legend of Zelda with OpenAI Gym."""

    # the legal range of rewards for each step
    reward_range = (-15, 15)

    def __init__(self):
        """Initialize a new Zelda 1 environment."""
        super().__init__(ROM_PATH)
        # Define the current objective.
        self._objective = 0
        # Define the highest reached objective for the current episode.
        self._highest_objective = 0
        # Define maximum number of steps for each objective before the environment resets.
        self._max_steps_per_objective = 1000
        self._done_after_objectives_completed = True
        self._start_in_level_1 = False
        self._give_rewards = True

        # Define list containing tuples representing objective locations and goals.
        # (OBJECTIVE_TYPE, _x_pixel, _y_pixel, _map_location, goal). (goal = -1 means get to the specified location).
        # Also define list containing properties to check against goals set in _objective_list.
        if self._start_in_level_1:
            self._objective_list = [(OBJECTIVE_TYPE[0],0,141,115,114), (OBJECTIVE_TYPE[1],114,3), (OBJECTIVE_TYPE[0],152,182,114,1), (OBJECTIVE_TYPE[0],224,141,114,115)]
            self._objective_goals = ["self._map_location", "self._killed_enemy_count", "self._number_of_keys"]
        else:
            # self._objective_list = [(OBJECTIVE_TYPE[0],64,77,119,0), (OBJECTIVE_TYPE[0],121,149,119,SWORD_TYPES[1]), (OBJECTIVE_TYPE[0],121,221,119,1), (OBJECTIVE_TYPE[0],120,61,119,103), (OBJECTIVE_TYPE[0],240,140,103,104)]
            # self._objective_goals = ["self._song_type_currently_active", "self._sword", "self._song_type_currently_active", "self._map_location", "self._map_location"]
            self._objective_list = [(OBJECTIVE_TYPE[0],240,141,119,120), (OBJECTIVE_TYPE[0],48,61,120,104), (OBJECTIVE_TYPE[0],0,157,104,103), (OBJECTIVE_TYPE[0],120,221,103,119)]
            self._objective_goals = []
        self._map_location_last = 119
        self._health_last = 3.
        self._rupees_last = 0
        self._killed_enemy_count_last = 0
        self._closest_enemy_distance_last = 0
        self._x_pixel_last = 0
        self._y_pixel_last = 0
        self._target_distance_last = 0
        self._sword_last = SWORD_TYPES[0]
        # Define an offset for the split between the menu and the game screen.
        self._y_offset = 64
        self._loitering_period = 0
        # Counter to keep track of number of steps in the current episode.
        self._steps = 0
        # Initialize a list to keep track of where the agent has explored.
        self._explored_area = []
        # Create a list to connect _map_location to entry in _explored_area list.
        self._explored_area_codes = []
        # reset the emulator, skip the start screen, and create a backup state
        self.reset()
        self._skip_start_screen()
        self._backup()

    # MARK: Memory access

    @property
    def _memory_testing(self):
        """Return value of the specified RAM address."""
        return self.ram[0x0621]

    @property
    def _is_screen_scrolling(self):
        """Return True if the screen is scrolling, False otherwise."""
        return self.ram[0x12] in SCROLL_GAME_MODES

    @property
    def _current_level(self):
        """Return the current level Link is in."""
        return self.ram[0x10]

    @property
    def _frame_counter(self):
        """Returns the number of frames Link has spent in the current map location."""
        return self.ram[0x15]

    @property
    def _current_save_slot(self):
        """Return the current save slot being played on."""
        return self.ram[0x16]

    @property
    def _x_pixel(self):
        """Return the current x pixel of Link's location."""
        return self.ram[0x70]

    @property
    def _enemy_1_x_pixel(self):
        """Return the current x pixel of enemy 1's location."""
        return self.ram[0x71]

    @property
    def _enemy_2_x_pixel(self):
        """Return the current x pixel of enemy 2's location."""
        return self.ram[0x72]

    @property
    def _enemy_3_x_pixel(self):
        """Return the current x pixel of enemy 3's location."""
        return self.ram[0x73]

    @property
    def _enemy_4_x_pixel(self):
        """Return the current x pixel of enemy 4's location."""
        return self.ram[0x74]

    @property
    def _enemy_5_x_pixel(self):
        """Return the current x pixel of enemy 5's location."""
        return self.ram[0x75]

    @property
    def _enemy_6_x_pixel(self):
        """Return the current x pixel of enemy 6's location."""
        return self.ram[0x76]

    @property
    def _y_pixel(self):
        """Return the current y pixel of Link's location."""
        return self.ram[0x84]

    @property
    def _enemy_1_y_pixel(self):
        """Return the current y pixel of enemy 1's location."""
        return self.ram[0x85]

    @property
    def _enemy_2_y_pixel(self):
        """Return the current y pixel of enemy 2's location."""
        return self.ram[0x86]

    @property
    def _enemy_3_y_pixel(self):
        """Return the current y pixel of enemy 3's location."""
        return self.ram[0x87]

    @property
    def _enemy_4_y_pixel(self):
        """Return the current y pixel of enemy 4's location."""
        return self.ram[0x88]

    @property
    def _enemy_5_y_pixel(self):
        """Return the current y pixel of enemy 5's location."""
        return self.ram[0x89]

    @property
    def _enemy_6_y_pixel(self):
        """Return the current y pixel of enemy 6's location."""
        return self.ram[0x8A]

    @property
    def _direction(self):
        """Return the current direction that Link is facing."""
        return DIRECTIONS[self.ram[0x98]]

    @property
    def _player_1_buttons(self):
        """Returns button presses of player 1 last frame."""
        return self.ram[0x0248]

    @property
    def _game_is_paused(self):
        """Returns boolean corresponding to whether the game is paused or not."""
        # return self.ram[0xE0]
        return self.ram[0x0248] == 248

    @property
    def _map_location(self):
        """Return the current map location"""
        return self.ram[0xEB]

    @property
    def _has_candled(self):
        """Return True if Link has used a candle in the current room"""
        return bool(self.ram[0x0513])

    @property
    def _pulse_1_IM_type(self):
        """Return the IM type of pulse 1."""
        # TODO: gives "Small Heart" when text is blitting?
        return PULSE_1_IM_TYPES[self.ram[0x0605]]

    @property
    def _pulse_2_IM_type(self):
        """Return the IM type of pulse 2."""
        # TODO: gives "Bomb" when initial sword is picked up?
        return PULSE_2_IM_TYPES[self.ram[0x0607]]

    @property
    def _song_type_currently_active(self):
        """Returns currently active song type.
        $80 = Title,
        $40 = Dungeon,
        $20 = Level,
        $10 = Ending,
        $08 = Item,
        $04 = Triforce,
        $02 = Ganon,
        $01 = Overworld"""
        return self.ram[0x0609]

    @property
    def _killed_enemy_count(self):
        """Return the number of enemies killed on the current screen."""
        return self.ram[0x0627]

    @property
    def _number_of_deaths(self):
        """Return the number of times Link has died (for slot 1)."""
        # 0630    Number of deaths            save slot 1
        # 0631    Number of deaths            save slot 2
        # 0632    Number of deaths            save slot 3
        return self.ram[0x0630]

    @property
    def _sword(self):
        """Return the sword Link has."""
        return SWORD_TYPES[self.ram[0x0657]]

    @property
    def _number_of_bombs(self):
        """Return the number of bombs in inventory."""
        return self.ram[0x0658]

    @property
    def _arrows_type(self):
        """Return the type of arrows Link has."""
        return ARROWS_TYPES[self.ram[0x0659]]

    @property
    def _is_bow_in_inventory(self):
        """Return True if the bow is in Link's inventory."""
        return bool(self.ram[0x065A])

    @property
    def _candle_type(self):
        """Return the status of the candle Link has."""
        return CANDLE_TYPES[self.ram[0x065B]]

    @property
    def _is_whistle_in_inventory(self):
        """Return True if the candle is in Link's inventory."""
        return bool(self.ram[0x065C])

    @property
    def _is_food_in_inventory(self):
        """Return True if food is in Link's inventory."""
        return bool(self.ram[0x065D])

    @property
    def _potion_type(self):
        """Return True if potion is in Link's inventory."""
        return POTION_TYPES[self.ram[0x065E]]

    @property
    def _is_magic_rod_in_inventory(self):
        """Return True if the magic rod is in Link's inventory."""
        return bool(self.ram[0x065F])

    @property
    def _is_raft_in_inventory(self):
        """Return True if the raft is in Link's inventory."""
        return bool(self.ram[0x0660])

    @property
    def _is_magic_book_in_inventory(self):
        """Return True if the magic book is in Link's inventory."""
        return bool(self.ram[0x0661])

    @property
    def _ring_type(self):
        """Return True if the ring is in Link's inventory."""
        return RING_TYPES[self.ram[0x0662]]

    @property
    def _is_step_ladder_in_inventory(self):
        """Return True if the ladder is in Link's inventory."""
        return bool(self.ram[0x0663])

    @property
    def _is_magical_key_in_inventory(self):
        """Return True if the magic key is in Link's inventory."""
        return bool(self.ram[0x0664])

    @property
    def _is_power_bracelet_in_inventory(self):
        """Return True if the power bracelet is in Link's inventory."""
        return bool(self.ram[0x0665])

    @property
    def _is_letter_in_inventory(self):
        """Return True if the letter is in Link's inventory."""
        return bool(self.ram[0x0666])

    @property
    def _number_of_keys(self):
        """Return the number of keys in Link's inventory."""
        return self.ram[0x66E]

    @property
    def _compass(self):
        """Return the mapping of which compasses are collected."""
        # 0667    Compass in Inventory        One bit per level
        # 0669    Compass in Inventory        (Level 9)

    @property
    def _map(self):
        """Return the mapping of which maps are collected."""
        # 0668    Map in Inventory            One bit per level
        # 066A    Map in Inventory            (Level 9)

    @property
    def _is_clock_possessed(self):
        """Return True if the clock is possessed."""
        return bool(self.ram[0x066C])

    @property
    def _number_of_rupees(self):
        """Return the number of rupees Link has."""
        return self.ram[0x066D]

    @property
    def _number_of_keys(self):
        """Return the number of keys Link has."""
        return self.ram[0x066E]

    @property
    def _number_of_heart_containers(self):
        """Return the number of total heart containers."""
        return (self.ram[0x066F] >> 4) + 1

    @property
    def _full_hearts_remaining(self):
        """Return the number of remaining hearts."""
        return 0x0F & self.ram[0x066F]

    @property
    def _partial_heart_remaining(self):
        """Return the amount of the partial heart remaining (percentage)."""
        return self.ram[0x0670] / 255

    @property
    def _hearts_remaining(self):
        """Return the amount of floating point remaining hears."""
        return self._full_hearts_remaining + self._partial_heart_remaining

    @property
    def _triforce_pieces(self):
        """Return the triforce pieces collected."""
        # 0671 Triforce pieces. One bit per piece

    @property
    def _is_boomerang_in_inventory(self):
        """Return True if the boomerang is in Link's inventory."""
        return bool(self.ram[0x0674])

    @property
    def _is_magic_boomerang_in_inventory(self):
        """Return True if the magic boomerang is in Link's inventory."""
        return bool(self.ram[0x0675])

    @property
    def _is_magic_shield_in_inventory(self):
        """Return True if the magic shield is in Link's inventory."""
        return bool(self.ram[0x0676])

    @property
    def _max_number_of_bombs(self):
        """Return the max number of bombs that Link can carry."""
        return self.ram[0x067C]

    # MARK: RAM Hacks

    def _skip_start_screen(self):
        """Press and release start to skip the start screen."""
        # press and release the start button 21 times
        # - kill 21 frames to get to registration
        # - kill 10 frames to get to player 1 registration
        for _ in range(31):
            self._frame_advance(8)
            self._frame_advance(0)
        # select the letter A and kill 6 frames
        for _ in range(6):
            self._frame_advance(1)
            self._frame_advance(0)
        # move the cursor to the register button
        for _ in range(3):
            self._frame_advance(4)
            self._frame_advance(0)
        # press select to register the profile and subsequently start the game
        # by killing some frames and pressing select again
        for _ in range(9):
            self._frame_advance(8)
            self._frame_advance(0)
        # skip the opening screen animation
        while self._direction is None or bool(self.ram[0x007C]):
            self._frame_advance(0)

    def _wait_for_hearts(self):
        """Skip the death animation when Link dies."""
        while self._hearts_remaining <= 0:
            self._frame_advance(8)
            self._frame_advance(0)

    def _wait_for_scroll(self):
        """Wait for the screen to stop scrolling."""
        while self._is_screen_scrolling:
            self._frame_advance(8)
            self._frame_advance(0)

    def _skip_boring_actions(self):
        """Skip actions that the agent will find boring."""
        # displaying text
        while self.ram[0x0605] == 0x10:
            # each character takes 6 frames to draw
            for _ in range(6):
                self._frame_advance(0)
        # entering / exiting cave
        while self.ram[0x0606] == 0x08:
            self._frame_advance(0)

    def _skip_inventory_scroll(self):
        """Skip the scrolling action when showing / hiding inventory."""
        while 65 < self.ram[0xFC]:
            self._frame_advance(0)

    # MARK: Reward Function

    def _check_objective_completed(self):
        """Used to check whether the current objective has been completed."""
        _function_name = ""
        if self._objective >= len(self._objective_goals):
            _function_name = "self._map_location"
        else:
            _function_name = self._objective_goals[self._objective]
        if _function_name != "":
            if eval(_function_name) == self._objective_list[self._objective][-1]:
                return True
        else:
            if self._get_target_distance() < 1:
                return True

        return False

    def _get_target_distance(self, _x_i = None, _y_i = None):
        """Returns the absolute distance between Link and the current target, or a specified target."""
        if _x_i == None:
            _x_i = self._objective_list[self._objective][1]
        if _y_i == None:
            _y_i = self._objective_list[self._objective][2]

        return ((self._x_pixel - _x_i)**2 + (self._y_pixel - _y_i)**2)**.5

    def _objective_cleared(self):
        """Return reward for a cleared objective."""
        self._target_distance_last = self._get_target_distance()
        self._objective += 1
        self._highest_objective += 1
        if self._objective == self._highest_objective:
            print("Objective", self._objective, "completed!")
            print("(", self._x_pixel, ",", self._y_pixel, ")")

        return 10

    def _map_location_penalty(self):
        if self._objective_list[self._objective][-2] != self._map_location:
            return -10

        return 0

    def _objective_reward(self):
        """Return the reward based on the progress towards the current objective."""
        if self._objective_list[self._objective][0] == OBJECTIVE_TYPE[0]:
            if self._objective >= len(self._objective_list):
                return 0

            if self._check_objective_completed():
                return self._objective_cleared()
            else:
                if self._objective > 0 and self._objective_list[self._objective-1][-2] == self._map_location:
                    self._objective -= 1
                    return -10

                if self._map_location_penalty() < 0:
                    return 0

                _target_distance = self._get_target_distance()
                _difference = _target_distance - self._target_distance_last
                _reward = 3 * math.atan(_difference)
                self._target_distance_last = _target_distance
                if abs(_difference) > 10:
                    return 0

                return _reward

        elif self._objective_list[self._objective][0] == OBJECTIVE_TYPE[1]:
            if self._objective_list[self._objective][-1] == self._killed_enemy_count:
                return self._objective_cleared()
            _closest_enemy_distance = 0
            for i in range(self._objective_list[self._objective][-1]):
                _function_name_x = "self._enemy_" + str(i+1) + "_x_pixel"
                _function_name_y = "self._enemy_" + str(i+1) + "_y_pixel"
                _enemy_distance = self._get_target_distance(eval(_function_name_x), eval(_function_name_y))
                if _enemy_distance < _closest_enemy_distance:
                    _closest_enemy_distance = _enemy_distance

            _difference = self._closest_enemy_distance_last - _closest_enemy_distance
            self._closest_enemy_distance_last = _closest_enemy_distance
            if self._killed_enemy_count_last == self._killed_enemy_count:
                return math.atan(_difference)

        return 0

    def _distance_penalty(self):
        _penalty = -self._get_target_distance()/50
        return _penalty

    # def _exploration_reward(self):
    #     """Return the reward for exploring the map."""
    #     # The area where Link can be is approximately 255*175 pixels (x:0-255, y:64-239).
    #     # If we divide these dimensions by 16, we get a (16, 11) matrix which will represent each position Link can be in.
    #     _reward = 0
    #     _height = 10
    #     _width = 15
    #     if self._map_location not in self._explored_area_codes:
    #         self._explored_area_codes.append(self._map_location)
    #         self._explored_area.append([])
    #         for i in range(_height):
    #             self._explored_area[-1].append([])
    #             for j in range(_width):
    #                 self._explored_area[-1][-1].append(0)
    #
    #         _reward = 1
    #
    #     _map_loc = self._explored_area[self._explored_area_codes.index(self._map_location)]
    #     _x_i = self._x_pixel // 16
    #     _y_i = (self._y_pixel - self._y_offset) // 16
    #     try:
    #         _grid_loc = _map_loc[_y_i][_x_i]
    #     except:
    #         _grid_loc = -1
    #
    #     if _grid_loc == -1:
    #         return 0
    #     elif _grid_loc == 0:
    #         _map_loc[_y_i][_x_i] = 1
    #         _reward = 1
    #         # Printing.
    #         print("\n")
    #         print("(", _x_i, ",", _y_i, "), map square", self._map_location)
    #         for i in _map_loc:
    #             print(i)
    #
    #     return _reward

    def _kill_reward(self):
        """Return the reward for slaying monsters."""
        if self._killed_enemy_count < self._killed_enemy_count_last:
            return 0

        _reward = 10 * (self._killed_enemy_count - self._killed_enemy_count_last)
        self._killed_enemy_count_last = self._killed_enemy_count
        return _reward

    def _health_reward(self):
        """Return the reward based on the difference in health between steps."""
        _reward = 4 * (self._hearts_remaining - self._health_last)
        self._health_last = self._hearts_remaining
        return _reward

    def _rupee_reward(self):
        """Return reward for collecting/spending rupees."""
        _reward = 1 * (self._number_of_rupees - self._rupees_last)
        if self._number_of_rupees > self._rupees_last:
            return _reward

        return 0

    def _pause_penalty(self):
        """Return the reward earned while the game is paused."""
        if self._game_is_paused:
            return -0.1

        return 0

    def _loitering_penalty(self):
        """Return the reward earned by loitering."""
        _reward = 0
        if (self._x_pixel == self._x_pixel_last) and (self._y_pixel == self._y_pixel_last):
            if self._loitering_period < 120:
                self._loitering_period += 1

            _reward = -math.exp(0.05*self._loitering_period) / 100
        else:
            self._loitering_period = 0

        self._x_pixel_last = self._x_pixel
        self._y_pixel_last = self._y_pixel
        return _reward if _reward > -5 else -5

    def _death_penalty(self):
        """Return the reward earned by dying."""
        if self._hearts_remaining == 0:
            return -25

        return 0

    # MARK: nes-py API calls

    def _will_reset(self):
        """Handle and RAM hacking before a reset occurs."""
        self._objective = 0
        self._highest_objective = 0
        self._health_last = 3.
        self._rupees_last = 0
        self._killed_enemy_count_last = 0
        self._closest_enemy_distance_last = 0
        self._x_pixel_last = 0
        self._y_pixel_last = 0
        self._sword_last = SWORD_TYPES[0]
        self._loitering_period = 0
        self._steps = 0
        self._explored_area = []
        self._explored_area_codes = []

    def _did_reset(self):
        """Handle any RAM hacking after a reset occurs."""
        self._x_pixel_last = self._x_pixel
        self._y_pixel_last = self._y_pixel
        if self._give_rewards:
            self._target_distance_last = self._get_target_distance()
        if self._start_in_level_1:
            self.ram[0xEB] = 55
            self.ram[0x0657] = 1

    def _did_step(self, done):
        """
        Handle any RAM hacking after a step occurs.

        Args:
            done: whether the done flag is set to true

        Returns:
            None

        """
        if done:
            return

        self._wait_for_hearts()
        self._wait_for_scroll()
        self._skip_boring_actions()
        self._skip_inventory_scroll()
        self._steps += 1
        if self._map_location != self._map_location_last:
            self._killed_enemy_count_last = 0
            self._map_location_last = self._map_location

    def _get_reward(self):
        """Return the reward after a step occurs."""
        if self._give_rewards:
            return self._objective_reward() +\
                   self._health_reward() +\
                   self._rupee_reward() +\
                   self._kill_reward() +\
                   self._loitering_penalty() +\
                   self._map_location_penalty() +\
                   self._distance_penalty() +\
                   self._death_penalty()
        else:
            return 0

    def _get_done(self):
        """Return True if the episode is over, False otherwise."""
        if not self._done_after_objectives_completed:
            return False

        if self._objective == len(self._objective_list):
            return True

        if self._steps >= self._max_steps_per_objective * (self._objective + 1) + self._max_steps_per_objective * self._objective:
            return True

        return False

    def _get_info(self):
        """Return the info after a step occurs"""
        return dict(
            memory_testing=self._memory_testing,
            objective=self._objective,
            target_distance=self._target_distance_last,
            current_level=self._current_level,
            x_pos=self._x_pixel,
            y_pos=self._y_pixel,
            direction=self._direction,
            map_location=self._map_location,
            game_paused=self._game_is_paused,
            has_candled=self._has_candled,
            pulse_1=self._pulse_1_IM_type,
            pulse_2=self._pulse_2_IM_type,
            killed_enemies=self._killed_enemy_count,
            number_of_deaths=self._number_of_deaths,
            sword=self._sword,
            number_of_bombs=self._number_of_bombs,
            arrows_type=self._arrows_type,
            has_bow=self._is_bow_in_inventory,
            candle_type=self._candle_type,
            has_whistle=self._is_whistle_in_inventory,
            has_food=self._is_food_in_inventory,
            potion_type=self._potion_type,
            has_magic_rod=self._is_magic_rod_in_inventory,
            has_raft=self._is_raft_in_inventory,
            has_magic_book=self._is_magic_book_in_inventory,
            ring_type=self._ring_type,
            has_step_ladder=self._is_step_ladder_in_inventory,
            has_magic_key=self._is_magical_key_in_inventory,
            has_power_bracelet=self._is_power_bracelet_in_inventory,
            has_letter=self._is_letter_in_inventory,
            is_clock_possessed=self._is_clock_possessed,
            rupees=self._number_of_rupees,
            keys=self._number_of_keys,
            heart_containers=self._number_of_heart_containers,
            hearts=self._hearts_remaining,
            has_boomerang=self._is_boomerang_in_inventory,
            has_magic_boomerang=self._is_magic_boomerang_in_inventory,
            has_magic_shield=self._is_magic_shield_in_inventory,
            max_number_of_bombs=self._max_number_of_bombs,
        )


# explicitly define the outward facing API of this module
__all__ = [Zelda1Env.__name__]
