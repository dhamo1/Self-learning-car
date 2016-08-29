import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
from pandas import DataFrame as df

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.state = None
        self.gamma = 0.9
        self.policy = {}
        self.q = df(columns=['state','actions','policy_value','mode','step'])
        self.step = 0 
        self.alpha = 1.0
        self.epsilon = 0.5
        self.current_policy_value = 0.0
        self.qsa_hat = 0.0
        self.qsa = 0.0
        self.reward_sum = 0.0
        self.max_policy = 0.0
        
    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.reward_sum = 0.0
        self.state = None
        self.max_policy = 0

    def update(self, t):  
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        #state = self.env.agent_states[self]
        self.state = (inputs['light'],inputs['oncoming'],inputs['right'],inputs['left'],self.next_waypoint)
        	
        # TODO: Select action according to your policy   
        #action = self.next_waypoint
        
        action = self.find_best_action(self.state, self.epsilon)
        self.epsilon *= 0.99

        # Execute action and get reward
        
        reward = self.env.act(self, action)
        self.reward_sum += reward
        
        # TODO: Learn policy based on state, action, reward
        
        ##### STUDENT UPDATE STARTS 
        #calculate next_state
        self.step += 1
        gps_waypoint = self.planner.next_waypoint()
        sensor = self.env.sense(self)
        next_state = (sensor['light'],sensor['oncoming'],sensor['right'],sensor['left'],gps_waypoint)
        
        self.policy = {}
        self.policy['state'] = str(self.state)
        self.policy['actions'] = str(action)  
        self.policy['mode'] = 1
        self.policy['step'] = self.step
        # Populate current state action value from q table if not available in the table default to 10.0
        try:
            self.current_policy_value = self.q[(self.q.state == repr(self.state)) & (self.q.actions == str(action))]['policy_value'].item()
        except ValueError:
        	self.current_policy_value = 10.0
        	
        #  Q(state, action) = R(state, action) + Gamma * Max[Q(next state, all actions)]
        self.qsa = reward + self.gamma * self.max_next_state_action(next_state)

        # v <-- (1-alpha)*v+alpha * x
        self.qsa_hat = ((1 - self.alpha) * self.current_policy_value) + (self.alpha * self.qsa)
        self.policy['policy_value'] = self.qsa_hat

        self.alpha = 1.0 / self.step
        
        #update q table if the state is already populated else append q table
        print '=====================================================================================================================' 
        print 'next_state ',next_state
        print 'next value ',self.qsa
        print 'curr state ',self.policy
        print 'current policy ',self.current_policy_value
        print 'alpha ',self.alpha
        print 'qsa-hat ',self.qsa_hat
        
        if len(self.q[(self.q.state == repr(self.state)) & (self.q.actions == str(action))]) == 1:
        	self.q.loc[(self.q.state == repr(self.state)) & (self.q.actions == str(action)), ('policy_value','actions','mode','step')] = \
        	    (self.policy['policy_value'], str(action),2,self.step)
        	print 'step, update ',self.step
        elif len(self.q[(self.q.state == repr(self.state)) & (self.q.actions == str(action))]) == 0:
        	self.q = self.q.append(self.policy,ignore_index=True)
        	print 'step, append ',self.step
        
       ### STUDENT UPDATE ENDS
        
        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]
        
    	if self.env.agent_states[self]['location'] == self.env.agent_states[self]['destination']:
            print 'reward for this trip ',self.reward_sum
            try:
    	        self.q.to_csv('smartcar.csv', sep='\t')
    	        print 'smartcar.csv created '
            except:
                print self.q
                
    def max_next_state_action(self, state):
    	# populate the maximum policy value from the q table for the given state
    	# if the state is not yet populated assign 10 (optimism when unknown)
    	try:
    		return  max(self.q[self.q.state == repr(state)]['policy_value'])
    	except ValueError:
    		return  10.0
    		
    def find_best_action(self, state, epsilon):
        if random.random() < epsilon:
            action = random.choice(Environment.valid_actions)
        else:
            try:
                self.max_policy = max(self.q[self.q.state == repr(self.state)]['policy_value'])
                action = self.q[(self.q.state == repr(self.state)) & (self.q.policy_value == self.max_policy)]['actions'].item()
                print 'non-random,step==>',self.step,' ',action,' ',self.max_policy
                if action == 'None':
                    action = None
            except ValueError:
        	    action = random.choice(Environment.valid_actions)
        return action
    	
def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.001, display=True)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line
    

if __name__ == '__main__':
    run()
