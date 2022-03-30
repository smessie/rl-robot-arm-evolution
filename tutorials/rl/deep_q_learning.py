import torch
import random
from itertools import count
from collections import deque
from qtable import QTable
from random import randint
from skip import Skip
from termcolor import colored

N_ACTIONS = 13*4 + 1 
SKIP = (0,0)
START = (3,0)

output_to_action_mapping = [ SKIP ] + [ (rank, amount) for rank in range(3,16) for amount in range(1,5) ]

class PresidentNetwork(torch.nn.Module):
    def __init__(self, hidden_nodes):
        super(PresidentNetwork, self).__init__()
        self.linear1 = torch.nn.Linear(16, hidden_nodes)
        self.linear2 = torch.nn.Linear(hidden_nodes, hidden_nodes)
        self.linear3 = torch.nn.Linear(hidden_nodes, N_ACTIONS)
    
    def forward(self, x):
        x = torch.nn.functional.relu(self.linear1(x))
        x = torch.nn.functional.relu(self.linear2(x))
        return self.linear3(x)


class DeepQLearningAgent(Player):
    '''
    Player class that implements a deep Q learning agent
    '''
    def __init__(self, name, train = False, network_path=None):
        super().__init__(name)
        self.training = train
        self.name = name
        self.BATCH_SIZE = 64 
        self.MEM_SIZE = 1000
        self.GAMMA = 0.5
        self.EPS_END = 0.05
        self.eps = 1.0
        self.EPS_DECAY = 0.9999
        self.N_ACTIONS = N_ACTIONS 
        if network_path:
            self.network = torch.load(network_path)
            self.network.eval()
        else:
            self.network = PresidentNetwork(64)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=1e-4)
        self.memory = deque(maxlen=self.MEM_SIZE)
        self.last_action = None
        self.last_action_illegal = False
        self.last_state = None
        self.done = False
        self.plays_this_game = 0
        self.cards_played_this_round = 0
        self.plays_this_round = 0
        self.normalized = True
    
    def play(self, last_move):
        '''
        Overwriting parent method
        '''
        possible_moves = MoveGenerator().generate_possible_moves(self.cards, last_move)
        possible_moves.append(Skip())
        self.possible_moves = possible_moves
                                           
        state = self.get_state(last_move)

        # update plays counter
        self.plays_this_game += 1
        self.plays_this_round += 1

        # depending on if we want to train the agent or not, use different methods
        if not self.training:
            # get action
            action = self.optimal_play(state)
            # transform action to move
            next_move = self.action_to_move(action, possible_moves)
            # check if move is a valid move
            if next_move is None or next_move is Skip():
                return Skip()
            else:
                self.cards = list(filter(lambda card: card not in next_move.cards, self.cards))
                return next_move 


        
        output = self.train_play(state)
        #safe this action and state so we can use them when we get the new state, also reset last_action_illegal
        self.last_action = output
        self.last_state = state
        self.last_action_illegal = False
        next_move = self.action_to_move(self.output_to_action(output), possible_moves)

        # if move is impossible, let move be a skip and remember we played an illegal action
        if next_move is None:
            next_move = Skip()
            self.last_action_illegal = True

        if not next_move is Skip():
            self.cards_played_this_round += next_move.amount
            self.cards = list(filter(lambda card: card not in next_move.cards, self.cards))
        
        return next_move 


    def train_play(self, _state):
        '''
        Method that will update the network and get a action from the network 
        Parameters:
            _state: [int]
        Returns:
            output: network_output
        '''
        # _state is de state dus [ .. ] maar nog niet in een tensor!

        # if we didn't do anything yet, generate a random move
        # hoe groot kiezen we de eps?
        if self.last_action == None: 
            return self.select_action(torch.tensor([_state]).float(), 1)

        self.update(_state)

        return self.select_action(torch.tensor([_state]).float(), self.eps)

    def optimal_play(self, _state):
        '''
        Method that will get the optimal play from the network
        Parameters:
            _state: [int]
        Returns:
            action: (amount, rank)
        '''
        state = torch.tensor([_state]).float()
        output = self.select_action(state, 0)
        action = self.output_to_action(output)
        return action

    def update(self, _state):
        '''
        Method for updating the network
        Parameters: 
            _state: [int]
        '''
            
        # Select and perform an action
        action = self.last_action 
        done = self.done #if the game is over
        reward = self.get_reward(_state)
        state = torch.tensor([self.last_state])
        next_state = torch.tensor([_state])

        # Store the transition in memory
        self.memory.append((state, action, reward, next_state, int(done)))

        # Move to the next state
        state = next_state

        # Experience replay
        if len(self.memory) >= self.BATCH_SIZE:
            batch = random.sample(self.memory, self.BATCH_SIZE)
            states, actions, rewards, n_states, dones = zip(*batch)
            state_batch = torch.cat(states,0)
            action_batch = torch.tensor(actions)
            reward_batch = torch.tensor(rewards)
            n_states = torch.cat(n_states)
            dones = torch.tensor(dones)
            
            # EXPERIENCE REPLAY
            
            # Bereken de Q-values voor de gegeven toestanden
            curr_Q = self.network(state_batch.float()).gather(1, action_batch.unsqueeze(1))
            curr_Q = curr_Q.squeeze(1)
                        
            # Bereken de Q-values voor de volgende toestanden (n_states)
            max_next_Q = (1-dones) * self.network(n_states.float()).max(1)[0].detach()

            # Gebruik deze Q-values om targets te berekenen
            targets = reward_batch + (self.GAMMA*max_next_Q)
            
            # Bereken de loss
            loss_fn = torch.nn.MSELoss()
            loss = loss_fn(curr_Q, targets)
            self.optimizer.zero_grad()
            loss.backward()
            
            # Voer een optimalisatiestap uit
            self.optimizer.step()

            # Decay exploration rate
            self.eps *= self.EPS_DECAY
            self.eps = max(self.EPS_END, self.eps)
                

    def select_action(self, state, eps):
        '''
        Method that selects a action to play
        Parameters:
            state: tensor([[int]])
            eps: float
        Returns:
            action: network_output
        '''
        if random.random() >= eps:
            with torch.no_grad():

                x = self.network(state)
                probs, indices = torch.topk(x, 53, sorted=True)
                move = None
                for action in indices[0]:
                    move = self.action_to_move(self.output_to_action(action.item()), self.possible_moves)
                    if not move is None:
                        return action.item()

#                return self.network(state).argmax().item()
        else:
            # kies random actie zonder te kijken naar state
            # uiteindelijk zal de ai leren welke acties wel en niet mogen door rewards ?
            return random.randrange(self.N_ACTIONS)    

    def get_reward(self, state):
        '''
        Method that calculates the rewards for a given state
        Parameters:
            state: [int]
        Returns:
            reward: float
        '''
        start_score = self.get_hand_score(self.last_state) 
        current_score = self.get_hand_score(state)

        longer_game_penalty = min((1.2)**(self.plays_this_game - 5), 4)
        longer_game_penalty = 0
        skip_penalty = 1.2**(self.plays_this_round - 2)
        #print(f"skip_penalty: {skip_penalty}")
        rel_hand_score = current_score/start_score
        
        #print(round(current_score, 2), round(start_score, 2), round(rel_hand_score, 2))

        # you won
        if current_score == 0:
            return 10 
        # you skipped
        if current_score == start_score:
            return -1.5 + skip_penalty 
        ## others skipped, you won round
        #if state[13:15] == [0,0] and not self.done:
        #    return 3  
        return rel_hand_score - longer_game_penalty

    def get_hand_score(self, state):
        '''
        Method to calculate the score of a hand
        Parameters:
            state: [int]
        '''
        if self.normalized:
            state = [ int(x*2 + 2) for x in state ]

        if not sum(state[:13]):
            return 0

        def score(rank, amount):
            mapping = [4,2,1,0.5]
            score = 0
            for i in range(amount):
                score += rank*mapping[i]
            return score

        return sum([score(i,state[i-1]) for i in range(1,14)]) / sum(state[:13])


    def get_state(self, move):
        '''
        Method 
        Parameters:
            move: Move
        Returns:
            move: [ amount_3 amount_4 ... value_last amount_last ]
        '''
        if self.normalized:
            norm_cards = list(map(lambda x: (x-2)/2, self.cards_to_list(self.cards)))
            if move is Skip():
                return  norm_cards +  [ 0, 0 ]+ [self.plays_this_game]
            if move.is_round_start():
                return  norm_cards + [ 3, 0 ]+ [self.plays_this_game]
            return norm_cards + [move.rank, move.amount]+ [self.plays_this_game]
        else:
            if move is Skip():
                return self.cards_to_list(self.cards) + [ 0, 0 ] + [self.plays_this_game]
            if move.is_round_start():
                return self.cards_to_list(self.cards) + [ 3, 0 ] + [self.plays_this_game]
            return self.cards_to_list(self.cards) + [move.rank, move.amount] + [self.plays_this_game]

    def cards_to_list(self, cards):
        '''
        Method that transforms a hand
        Parameters:
            cards: [Card]
        Returns:
            cards_int: [int]
        '''
        card_count = 13
        cards_ints = [ 0 for _ in range(card_count) ]

        for card in cards:
            if card.rank == 2:
                cards_ints[card_count-1] += 1
            else:
                cards_ints[card.rank-3] += 1
        
        return cards_ints

    def output_to_action(self, output):
        '''
        Method that transforms a output of the network to a action
        Parameters:
            output: int
        Returns:
            action: (rank, amount) | Skip
        '''
        return output_to_action_mapping[output]

    def move_to_action(self, move):
        '''
        Method that transforms a Move to a tuple used for indexing the QTable
        Parameters:
            move: Move
        Returns:
            action: (rank, amount)
        '''
        if move is Skip():
            return  SKIP
        return (move.rank, move.amount)

    def action_to_move(self, action, possible_moves):
        '''
        Method that seeks the move corresponding to a action
        Parameters:
            action: (rank, amount)
            possible_moves: [Move]
        Returns:
            move: Move
        '''
        l = list(filter(lambda move: self.move_to_action(move) == action, possible_moves))
        if not l: 
            return None 
        return l[0]

    def notify_round_end(self):
        '''
        Overwriting parent method
        '''
        if self.training:
            self.update(self.get_state(Skip()))
        self.cards_played_this_round = 0
        self.plays_this_round = 0

    def notify_game_end(self, rank):
        '''
        Overwriting parent method
        '''
        self.done = True
        if self.training:
            self.update(self.get_state(Skip()))
        self.plays_this_game = 0
        

    def notify_game_start(self):
        '''
        Overwriting parent method
        '''
        self.done = False

    def stop_training(self):
        '''
        Method that stops the training
        '''
        self.training = False