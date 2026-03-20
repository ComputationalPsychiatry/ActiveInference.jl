import h5py
import numpy as np

from pymdp import utils, maths, learning, control, inference
from pymdp.agent import Agent
from pymdp.envs import Env

#############################################
### Loading Generative Model from h5 file ###
#############################################

# Path to file with generative model
file_path_gm = "../../generative_model_creation/gm_data/gm_matrices.h5"

# A-matrix
A_cross = utils.obj_array(4)
for i in range(1, 5):
    with h5py.File(file_path_gm, 'r') as file:
        A_cross[i-1] = file[f"A_cross_{i}"][:]
        
    # Converting the column-major julia indexing into the row-major python indexing
    A_cross[i-1] = np.transpose(A_cross[i-1], (3, 2, 1, 0))

# pA-matrix
pA_cross = utils.obj_array(4)
for i in range(1, 5):
    with h5py.File(file_path_gm, 'r') as file:
        pA_cross[i-1] = file[f"pA_cross_{i}"][:]
        
    # Converting the column-major julia indexing into the row-major python indexing
    pA_cross[i-1] = np.transpose(pA_cross[i-1], (3, 2, 1, 0))

# B-matrix
B_cross = utils.obj_array(3)
for i in range(1, 4):
    with h5py.File(file_path_gm, 'r') as file:
        B_cross[i-1] = file[f"B_cross_{i}"][:]
        
    # Converting the column-major julia indexing into the row-major python indexing
    B_cross[i-1] = np.transpose(B_cross[i-1], (2, 1, 0))

# pB-matrix
pB_cross = utils.obj_array(3)
for i in range(1, 4):
    with h5py.File(file_path_gm, 'r') as file:
        pB_cross[i-1] = file[f"pB_cross_{i}"][:]
        
    # Converting the column-major julia indexing into the row-major python indexing
    pB_cross[i-1] = np.transpose(pB_cross[i-1], (2, 1, 0))

# C_matrix
C_cross = utils.obj_array(4)
for i in range(1, 5):
    with h5py.File(file_path_gm, 'r') as file:
        C_cross[i-1] = file[f"C_cross_{i}"][:]

# D-matrix
D_cross = utils.obj_array(3)
for i in range(1, 4):
    with h5py.File(file_path_gm, 'r') as file:
        D_cross[i-1] = file[f"D_cross_{i}"][:]
        
# pD-matrix
pD_cross = utils.obj_array(3)
for i in range(1, 4):
    with h5py.File(file_path_gm, 'r') as file:
        pD_cross[i-1] = file[f"pD_cross_{i}"][:]
        
################################
### Creating cross val agent ###
################################

cross_agent = Agent(A = A_cross, B = B_cross, C = C_cross, D = D_cross, pA = pA_cross, pB = pB_cross, pD = pD_cross, policy_len = 4, action_selection="deterministic", lr_pA = 0.5, lr_pB = 0.5, lr_pD = 0.5, use_states_info_gain=True, use_param_info_gain=True, save_belief_hist=True)

#############################################
### Creating and initialising environment ###
#############################################

grid_dims = [5, 7]
num_grid_points = np.prod(grid_dims) 

grid = np.arange(num_grid_points).reshape(grid_dims, order='F')

it = np.nditer(grid, flags=["multi_index"])

loc_list = []
while not it.finished:
    loc_list.append(it.multi_index)
    it.iternext()
    
cue1_location = (2, 0)

cue2_loc_names = ['L1', 'L2', 'L3', 'L4']
cue2_locations = [(0, 2), (1, 3), (3, 3), (4, 2)]

cue1_names = ['Null'] + cue2_loc_names
cue2_names = ['Null', 'reward_on_top', 'reward_on_bottom']
reward_names = ['Null', 'Cheese', 'Shock']

reward_conditions = ["TOP", "BOTTOM"]
reward_locations = [(1, 5), (3, 5)]

actions = ["UP", "DOWN", "LEFT", "RIGHT", "STAY"]

# Using a custom evironment corresponding to the Julia Epistemic chaining env
class GridWorldEnv():
    
    def __init__(self,starting_loc = (0,0), cue1_loc = (2, 0), cue2 = 'L1', reward_condition = 'TOP'):

        self.init_loc = starting_loc
        self.current_location = self.init_loc

        self.cue1_loc = cue1_loc
        self.cue2_name = cue2
        self.cue2_loc_names = ['L1', 'L2', 'L3', 'L4']
        self.cue2_loc = cue2_locations[self.cue2_loc_names.index(self.cue2_name)]

        self.reward_condition = reward_condition
        print(f'Starting location is {self.init_loc}, Reward condition is {self.reward_condition}, cue is located in {self.cue2_name}')
    
    def step(self,action_label):

        (Y, X) = self.current_location

        if action_label == "UP": 
          
          Y_new = Y - 1 if Y > 0 else Y
          X_new = X

        elif action_label == "DOWN": 

          Y_new = Y + 1 if Y < (grid_dims[0]-1) else Y
          X_new = X

        elif action_label == "LEFT": 
          Y_new = Y
          X_new = X - 1 if X > 0 else X

        elif action_label == "RIGHT": 
          Y_new = Y
          X_new = X +1 if X < (grid_dims[1]-1) else X

        elif action_label == "STAY":
          Y_new, X_new = Y, X 
        
        self.current_location = (Y_new, X_new)

        loc_obs = self.current_location 

        if self.current_location == self.cue1_loc:
          cue1_obs = self.cue2_name
        else:
          cue1_obs = 'Null'

        if self.current_location == self.cue2_loc:
          cue2_obs = cue2_names[reward_conditions.index(self.reward_condition)+1]
        else:
          cue2_obs = 'Null'
        
        if self.current_location == reward_locations[0]:
          if self.reward_condition == 'TOP':
            reward_obs = 'Cheese'
          else:
            reward_obs = 'Shock'
        elif self.current_location == reward_locations[1]:
          if self.reward_condition == 'BOTTOM':
            reward_obs = 'Cheese'
          else:
            reward_obs = 'Shock'
        else:
          reward_obs = 'Null'

        return loc_obs, cue1_obs, cue2_obs, reward_obs

    def reset(self):
        self.current_location = self.init_loc
        print(f'Re-initialized location to {self.init_loc}')
        loc_obs = self.current_location
        cue1_obs = 'Null'
        cue2_obs = 'Null'
        reward_obs = 'Null'

        return loc_obs, cue1_obs, cue2_obs, reward_obs

cross_env = GridWorldEnv(starting_loc = (0,0), cue1_loc = (2, 0), cue2 = 'L4', reward_condition = 'BOTTOM')

# Getting initial observation and setting it to correct format
with h5py.File(file_path_gm, 'r') as file:
    obs = file["obs"][:]
obs = obs - 1

obs_obj_array = np.empty(len(obs), dtype=object)
for i, ob in enumerate(obs):
    obs_obj_array[i] = ob
obs = obs_obj_array

obs = obs.tolist()

##########################
### Running simulation ###
##########################

# Time step set to 50 trials
T = 50

# run simulation
for t in range(T):

    qs = cross_agent.infer_states(obs)

    cross_agent.update_A(obs)
    
    if t != 0:
        qs_prev = cross_agent.qs_hist[t-1]
        cross_agent.update_B(qs_prev)
    
    if t == 0:
        qs_t1 = cross_agent.qs
        cross_agent.update_D(qs_t1)
    
    q_pi, G = cross_agent.infer_policies()
    
    chosen_action_id = cross_agent.sample_action()

    movement_id = int(chosen_action_id[0])

    choice_action = actions[movement_id]

    loc_obs, cue1_obs, cue2_obs, reward_obs = cross_env.step(choice_action)

    obs = [loc_list.index(loc_obs), cue1_names.index(cue1_obs), cue2_names.index(cue2_obs), reward_names.index(reward_obs)]


###########################
### Storing the results ###
###########################

# Storing the variables into the comparison hdf5 file
file_path_results = "../../cross_val_results/complete_run_data.h5"

with h5py.File(file_path_results, "w") as hdf:
  # Storing A-matrix results
  hdf.create_dataset("python_A_cross_1", data=cross_agent.A[0])
  hdf.create_dataset("python_A_cross_2", data=cross_agent.A[1])
  hdf.create_dataset("python_A_cross_3", data=cross_agent.A[2])
  hdf.create_dataset("python_A_cross_4", data=cross_agent.A[3])
  
  # Storing B-matrix results
  hdf.create_dataset("python_B_cross_1", data=cross_agent.B[0])
  hdf.create_dataset("python_B_cross_2", data=cross_agent.B[1])
  hdf.create_dataset("python_B_cross_3", data=cross_agent.B[2])
  
  # Storing D-matrix results
  hdf.create_dataset("python_D_cross_1", data=cross_agent.D[0])
  hdf.create_dataset("python_D_cross_2", data=cross_agent.D[1])
  hdf.create_dataset("python_D_cross_3", data=cross_agent.D[2])
  
  # Storing posterior states
  hdf.create_dataset("python_qs_1", data=cross_agent.qs[0])
  hdf.create_dataset("python_qs_2", data=cross_agent.qs[1])
  hdf.create_dataset("python_qs_3", data=cross_agent.qs[2])


