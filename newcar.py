# This Code is Heavily Inspired By The YouTuber: Cheesy AI
# Code Changed, Optimized And Commented By: NeuralNine (Florian Dedov)

from ast import Return
import math
import random
import sys
import os
import pandas as pd
import numpy as np

import neat
import pygame

import pickle

# Constants
# WIDTH = 1600
# HEIGHT = 880

WIDTH = 1000
HEIGHT = 1000

CAR_SIZE_X = 60    
CAR_SIZE_Y = 60

BORDER_COLOR = (255, 255, 255, 255) # Color To Crash on Hit

current_generation = 0 # Generation counter

# class Car:

#     def __init__(self):
#         # Load Car Sprite and Rotate
#         self.sprite = pygame.image.load('car.png').convert() # Convert Speeds Up A Lot
#         self.sprite = pygame.transform.scale(self.sprite, (CAR_SIZE_X, CAR_SIZE_Y))
#         self.rotated_sprite = self.sprite 

#         # self.position = [690, 740] # Starting Position
#         self.position = [830, 920] # Starting Position
#         self.angle = 0
#         self.speed = 0

#         self.speed_set = False # Flag For Default Speed Later on

#         self.center = [self.position[0] + CAR_SIZE_X / 2, self.position[1] + CAR_SIZE_Y / 2] # Calculate Center

#         self.radars = [] # List For Sensors / Radars
#         self.drawing_radars = [] # Radars To Be Drawn

#         self.alive = True # Boolean To Check If Car is Crashed

#         self.distance = 0 # Distance Driven
#         self.time = 0 # Time Passed

#     def draw(self, screen):
#         screen.blit(self.rotated_sprite, self.position) # Draw Sprite
#         self.draw_radar(screen) #OPTIONAL FOR SENSORS

#     def draw_radar(self, screen):
#         # Optionally Draw All Sensors / Radars
#         for radar in self.radars:
#             position = radar[0]
#             pygame.draw.line(screen, (0, 255, 0), self.center, position, 1)
#             pygame.draw.circle(screen, (0, 255, 0), position, 5)

#     def check_collision(self, game_map):
#         self.alive = True
#         for point in self.corners:
#             # If Any Corner Touches Border Color -> Crash
#             # Assumes Rectangle
#             if game_map.get_at((int(point[0]), int(point[1]))) == BORDER_COLOR:
#                 self.alive = False
#                 break

#     def check_radar(self, degree, game_map):
#         length = 0
#         x = int(self.center[0] + math.cos(math.radians(360 - (self.angle + degree))) * length)
#         y = int(self.center[1] + math.sin(math.radians(360 - (self.angle + degree))) * length)

#         # While We Don't Hit BORDER_COLOR AND length < 600 (just a max) -> go further and further
#         while not game_map.get_at((x, y)) == BORDER_COLOR and length < 600:
#             length = length + 1
#             x = int(self.center[0] + math.cos(math.radians(360 - (self.angle + degree))) * length)
#             y = int(self.center[1] + math.sin(math.radians(360 - (self.angle + degree))) * length)

#         # Calculate Distance To Border And Append To Radars List
#         dist = int(math.sqrt(math.pow(x - self.center[0], 2) + math.pow(y - self.center[1], 2)))
#         self.radars.append([(x, y), dist])
    
#     def update(self, game_map):
#         # Set The Speed To 20 For The First Time
#         # Only When Having 4 Output Nodes With Speed Up and Down
#         if not self.speed_set:
#             self.speed = 20
#             self.speed_set = True

#         # Get Rotated Sprite And Move Into The Right X-Direction
#         # Don't Let The Car Go Closer Than 20px To The Edge
#         self.rotated_sprite = self.rotate_center(self.sprite, self.angle)
#         self.position[0] += math.cos(math.radians(360 - self.angle)) * self.speed
#         self.position[0] = max(self.position[0], 20)
#         self.position[0] = min(self.position[0], WIDTH - 120)

#         # Increase Distance and Time
#         self.distance += self.speed
#         self.time += 1
        
#         # Same For Y-Position
#         self.position[1] += math.sin(math.radians(360 - self.angle)) * self.speed
#         self.position[1] = max(self.position[1], 20)
#         self.position[1] = min(self.position[1], WIDTH - 120)

#         # Calculate New Center
#         self.center = [int(self.position[0]) + CAR_SIZE_X / 2, int(self.position[1]) + CAR_SIZE_Y / 2]

#         # Calculate Four Corners
#         # Length Is Half The Side
#         length = 0.5 * CAR_SIZE_X
#         left_top = [self.center[0] + math.cos(math.radians(360 - (self.angle + 30))) * length, self.center[1] + math.sin(math.radians(360 - (self.angle + 30))) * length]
#         right_top = [self.center[0] + math.cos(math.radians(360 - (self.angle + 150))) * length, self.center[1] + math.sin(math.radians(360 - (self.angle + 150))) * length]
#         left_bottom = [self.center[0] + math.cos(math.radians(360 - (self.angle + 210))) * length, self.center[1] + math.sin(math.radians(360 - (self.angle + 210))) * length]
#         right_bottom = [self.center[0] + math.cos(math.radians(360 - (self.angle + 330))) * length, self.center[1] + math.sin(math.radians(360 - (self.angle + 330))) * length]
#         self.corners = [left_top, right_top, left_bottom, right_bottom]

#         # Check Collisions And Clear Radars
#         self.check_collision(game_map)
#         self.radars.clear()

#         # From -90 To 120 With Step-Size 45 Check Radar
#         for d in range(-90, 120, 20):
#             self.check_radar(d, game_map)

#     def get_data(self):
#         # Get Distances To Border
#         radars = self.radars
#         return_values = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#         for i, radar in enumerate(radars):
#             return_values[i] = int(radar[1] / 30)

#         return return_values

#     def is_alive(self):
#         # Basic Alive Function
#         return self.alive

#     def get_reward(self):
#         # Calculate Reward (Maybe Change?)
#         # return self.distance / 50.0
#         return self.distance / (CAR_SIZE_X / 2)

#     def rotate_center(self, image, angle):
#         # Rotate The Rectangle
#         rectangle = image.get_rect()
#         rotated_image = pygame.transform.rotate(image, angle)
#         rotated_rectangle = rectangle.copy()
#         rotated_rectangle.center = rotated_image.get_rect().center
#         rotated_image = rotated_image.subsurface(rotated_rectangle).copy()
#         return rotated_image


class Grid:

    def __init__(self):
        self.btc = pd.read_csv('btc.csv')
        self.eth = pd.read_csv('eth.csv')
        self.data = pd.merge(self.btc, self.eth, how='inner', on='snapped_at')
        pass

    def get_data(self):
        return self.data.drop('snapped_at', 'columns').to_numpy()

class Trade:
    def __init__(self, open_price, quantity, target_price):
        self.open_price = open_price
        self.quantity = quantity
        self.target_price = target_price
        self.status = 'Active'
        self.unrealized_profit = 0
    
    def update(self, current_price):
        self.current_price = current_price
        self.unrealized_profit = (self.open_price * self.quantity) - (self.open_price * self.quantity) 
        if current_price >= self.target_price and self.status == 'Active':
            self.close()
            return self
        return self
    
    def close(self):
        self.profit = self.unrealized_profit
        self.status = 'Closed'  
        return self.profit  
    

class Trader:
    def __init__(self,x,y):
        self.alive = True
        self.index_to_trade = 0
        self.current_price = 0
        self.balance = 1000
        self.position_size = self.balance * 0.03
        self.active_trades = []
        self.profit = 0
        self.closed_trades = []
        self.center = (x, y*20)
        pass

    def is_alive(self):
        return self.alive

    def draw(self):
        return

    def sell(self):
        if len(self.active_trades) == 0:
            return
        biggest_winner = self.active_trades.pop(np.argmax([trade.unrealized_profit for trade in self.active_trades]))
        self.profit += biggest_winner.close()
        self.closed_trades.append(biggest_winner)


    def buy(self):
        quantity = self.current_price * self.position_size
        self.balance -= self.position_size
        target_price = self.current_price + self.current_price*0.05
        self.active_trades.append(Trade(self.current_price, quantity, target_price))
        return

    def update(self, data):
        self.unrealized_profit = 0
        self.equity = self.balance
        self.current_price = data[self.index_to_trade]
        
        trades_to_remove = []
        for i, trade in enumerate(self.active_trades):
            trade.update(self.current_price)
            self.unrealized_profit += trade.unrealized_profit
            if trade.status == 'Closed':
                self.profit += trade.profit
                self.balance += self.position_size
                trades_to_remove.append(i)

            self.equity += trade.unrealized_profit

        if len(trades_to_remove) > 0:
            if len(trades_to_remove) > 0:
                trades_to_remove.reverse()
            for i in trades_to_remove:
                trade = self.active_trades.pop(i) # Remove from last to first
                self.closed_trades.append(trade)
        
        if self.equity <= 0:
            self.alive = False

    def get_reward(self):
        return self.equity

    def draw(self, screen):
        pygame.draw.line(screen, (0, 255, 0), (self.center[0],self.center[1]), (self.center[0]+np.abs(self.unrealized_profit),self.center[1]), 15)



def run_simulation(genomes, config):
    
    # Empty Collections For Nets and Cars
    nets = []
    traders = []
    grid = Grid()

    # Initialize PyGame And The Display
    pygame.init()
    # screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.FULLSCREEN)
    screen = pygame.display.set_mode((WIDTH, HEIGHT))

    # For All Genomes Passed Create A New Neural Network
    for i, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        g.fitness = 0
        traders.append(Trader(WIDTH/2, i))

    # Clock Settings
    # Font Settings & Loading Map
    clock = pygame.time.Clock()
    generation_font = pygame.font.SysFont("Arial", 30)
    alive_font = pygame.font.SysFont("Arial", 20)
    game_map = pygame.image.load('map3.png').convert() # Convert Speeds Up A Lot

    global current_generation
    current_generation += 1

    # Simple Counter To Roughly Limit Time (Not Good Practice)
    counter = 0

    

    for i, data in enumerate(grid.get_data()):
        # Exit On Quit Event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit(0)

        # For Each Car Get The Acton It Takes
        for i, trader in enumerate(traders):
            trader.update(data)
            output = nets[i].activate(data)
            choice = output.index(max(output))
            if choice == 0:
                trader.buy() # Left
            elif choice == 1:
                trader.sell() # Right
            elif choice == 2:
                continue
            else:
                continue
        
        # Check If Car Is Still Alive
        # Increase Fitness If Yes And Break Loop If Not
        still_alive = 0
        for i, trader in enumerate(traders):
            if trader.is_alive():
                still_alive += 1
                genomes[i][1].fitness += trader.get_reward()

        if still_alive == 0:
            break

        counter += 1
        if counter == 30 * 40: # Stop After About 20 Seconds
            break

        # Draw Map And All traders That Are Alive
        screen.blit(game_map, (0, 0))
        print('printing map')
        for trader in traders:
            if trader.is_alive():
                trader.draw(screen)
        
        # Display Info
        text = generation_font.render("Generation: " + str(current_generation), True, (204,0,34))
        text_rect = text.get_rect()
        text_rect.center = (900, 450)
        screen.blit(text, text_rect)

        text = alive_font.render("Still Alive: " + str(still_alive), True, (204,0,34))
        text_rect = text.get_rect()
        text_rect.center = (900, 490)
        screen.blit(text, text_rect)

        pygame.display.flip()
        clock.tick(60) # 60 FPS

if __name__ == "__main__":
    
    # Load Config
    config_path = "./config.txt"
    config = neat.config.Config(neat.DefaultGenome,
                                neat.DefaultReproduction,
                                neat.DefaultSpeciesSet,
                                neat.DefaultStagnation,
                                config_path)

    # Create Population And Add Reporters
    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    
    # Run Simulation For A Maximum of 30 Generations
    winner = population.run(run_simulation, 30)	
    print('Winner has emerged!')
    print('Fitness ', winner.fitness)

    # Save the model in a pickle file
    filename = 'best_nn'
    outfile = open(filename,'wb')
    pickle.dump(winner, outfile)
    outfile.close()



